#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step3_blast_remote_and_parse.py - 远程BLAST搜索与结果解析
作者: Kuroneko
日期: 2025.9.04

功能说明:
    1. 读取序列文件（原始或SEG掩蔽后）
    2. 使用 NCBI BLAST Web 服务进行远程同源搜索
    3. 解析 BLAST 结果（E-value, 比对得分, 覆盖度等）
    4. 生成结果报告（CSV格式）
    5. 针对合成序列给出特别提示
    6. 包含隐私警告与用户确认机制

输入输出:
    输入:
        - outputs/seg_masked.fasta（优先）或
        - data/input_sequences.fasta（备选）
    输出:
        - outputs/blast_results.csv（BLAST结果摘要）
        - outputs/blast_详细结果_{timestamp}.txt（详细结果）

运行示例:
    python step3_blast_remote_and_parse.py

设计决策:
    - 使用 Biopython NCBIWWW 模块进行远程 BLAST
    - 默认使用 blastp（蛋白）或 blastn（核酸）
    - 限制返回结果数量以节省时间
    - 对每个序列单独提交（避免批量提交超时）
    - 合成序列需要额外确认才能上传
"""

import os
import sys
import time
import csv
from pathlib import Path
from datetime import datetime

# 导入工具模块
try:
    from step7_utils import (
        setup_logger, get_user_input, confirm_action,
        print_section_header, read_fasta, check_sequence_validity,
        privacy_warning_blast, DATA_DIR, OUTPUT_DIR, format_timestamp
    )
except ImportError:
    print("错误: 无法导入 step7_utils 模块")
    print("请确保 step7_utils.py 与本脚本在同一目录下")
    sys.exit(1)


def detect_sequence_source(seq_id: str) -> str:
    """
    从序列ID检测来源类型
    
    参数:
        seq_id: 序列标识符
    
    返回:
        'public', 'synthetic', 或 'user'
    """
    seq_id_upper = seq_id.upper()
    if 'SYNTH' in seq_id_upper or 'SYNTHETIC' in seq_id_upper:
        return 'synthetic'
    elif any(prefix in seq_id_upper for prefix in ['SP|', 'TR|', 'NM_', 'NP_', 'XM_', 'XP_']):
        return 'public'
    else:
        return 'user'


def select_representative_sequences(records: list, logger) -> list:
    """
    智能选择代表性序列进行BLAST
    
    选择策略：
    1. 优先选择通过验证的序列
    2. 选择不同长度的序列（短、中、长）
    3. 避免合成序列和包含特殊字符的序列
    4. 最多选择3条序列
    
    参数:
        records: 序列记录列表
        logger: 日志记录器
    
    返回:
        选中的序列列表
    """
    logger.info("开始智能选择代表性序列...")
    
    # 按长度分类
    short_seqs = []    # < 100 aa
    medium_seqs = []   # 100-300 aa  
    long_seqs = []     # > 300 aa
    
    for record in records:
        seq_len = len(record['sequence'])
        source_type = detect_sequence_source(record['id'])
        
        # 跳过合成序列和包含特殊字符的序列
        if source_type == 'synthetic':
            continue
        if any(char in record['sequence'] for char in '!@#$%^&*()_+-=[]{}|;\':",./<>?`~'):
            continue
        
        if seq_len < 100:
            short_seqs.append(record)
        elif seq_len <= 300:
            medium_seqs.append(record)
        else:
            long_seqs.append(record)
    
    selected = []
    
    # 优先选择中等长度序列（最有可能有同源序列）
    if medium_seqs:
        selected.append(medium_seqs[0])
        logger.info(f"选择中等长度序列: {medium_seqs[0]['id']} ({len(medium_seqs[0]['sequence'])} aa)")
    
    # 选择一条短序列（如果存在）
    if short_seqs and len(selected) < 3:
        selected.append(short_seqs[0])
        logger.info(f"选择短序列: {short_seqs[0]['id']} ({len(short_seqs[0]['sequence'])} aa)")
    
    # 选择一条长序列（如果存在）
    if long_seqs and len(selected) < 3:
        selected.append(long_seqs[0])
        logger.info(f"选择长序列: {long_seqs[0]['id']} ({len(long_seqs[0]['sequence'])} aa)")
    
    # 如果还不够3条，从剩余序列中选择
    if len(selected) < 3:
        remaining = [r for r in records if r not in selected and detect_sequence_source(r['id']) != 'synthetic']
        for record in remaining[:3-len(selected)]:
            selected.append(record)
            logger.info(f"选择额外序列: {record['id']} ({len(record['sequence'])} aa)")
    
    logger.info(f"智能选择完成，共选择 {len(selected)} 条序列")
    return selected


def manual_select_sequences(records: list) -> list:
    """
    手动选择序列进行BLAST
    
    参数:
        records: 序列记录列表
    
    返回:
        选中的序列列表
    """
    print("\n可用序列列表:")
    for i, record in enumerate(records, 1):
        seq_len = len(record['sequence'])
        source_type = detect_sequence_source(record['id'])
        print(f"  {i}. {record['id']} ({seq_len} aa, 来源: {source_type})")
    
    print("\n请输入要BLAST的序列编号（用逗号分隔，如: 1,3,5）:")
    user_input = input("序列编号: ").strip()
    
    try:
        indices = [int(x.strip()) - 1 for x in user_input.split(',')]
        selected = []
        for idx in indices:
            if 0 <= idx < len(records):
                selected.append(records[idx])
            else:
                print(f"警告: 序列编号 {idx+1} 无效，跳过")
        
        return selected
    except ValueError:
        print("输入格式错误，将使用前3条序列")
        return records[:3]


def run_blast_search(sequence: str, seq_id: str, blast_program: str, database: str, 
                     max_hits: int, logger) -> dict:
    """
    执行远程 BLAST 搜索
    
    参数:
        sequence: 查询序列
        seq_id: 序列标识符
        blast_program: BLAST程序 ('blastp', 'blastn' 等)
        database: 数据库 ('nr', 'nt', 'pdb' 等)
        max_hits: 最多返回命中数
        logger: 日志记录器
    
    返回:
        BLAST结果字典
    """
    try:
        from Bio.Blast import NCBIWWW, NCBIXML
    except ImportError:
        logger.error("需要安装 Biopython: pip install biopython")
        raise ImportError("Biopython not installed")
    
    logger.info(f"提交 BLAST 查询: {seq_id}")
    logger.info(f"  程序: {blast_program}, 数据库: {database}, 序列长度: {len(sequence)}")
    
    try:
        # 提交 BLAST 查询
        print(f"  正在提交 {seq_id} 到 NCBI BLAST（这可能需要几分钟）...")
        result_handle = NCBIWWW.qblast(
            program=blast_program,
            database=database,
            sequence=sequence,
            hitlist_size=max_hits,
            expect=10.0
        )
        
        # 解析结果
        blast_record = NCBIXML.read(result_handle)
        result_handle.close()
        
        logger.info(f"  完成，找到 {len(blast_record.alignments)} 个比对")
        
        return {
            'success': True,
            'record': blast_record,
            'error': None
        }
        
    except Exception as e:
        logger.error(f"  BLAST 查询失败: {e}")
        return {
            'success': False,
            'record': None,
            'error': str(e)
        }


def parse_blast_results(blast_record, seq_id: str, logger) -> list:
    """
    解析 BLAST 结果
    
    参数:
        blast_record: BioPython BLAST 记录对象
        seq_id: 查询序列ID
        logger: 日志记录器
    
    返回:
        结果列表（字典）
    """
    results = []
    
    if not blast_record or not blast_record.alignments:
        logger.info(f"  {seq_id}: 无显著比对结果")
        return results
    
    for alignment in blast_record.alignments:
        # 取第一个HSP（高分片段对）
        if alignment.hsps:
            hsp = alignment.hsps[0]
            
            # 计算覆盖度
            query_coverage = hsp.align_length / blast_record.query_length * 100
            identity_percent = hsp.identities / hsp.align_length * 100 if hsp.align_length > 0 else 0
            
            result = {
                'query_id': seq_id,
                'hit_id': alignment.hit_id,
                'hit_def': alignment.hit_def[:100],  # 限制长度
                'e_value': f"{hsp.expect:.2e}",
                'bit_score': hsp.bits,
                'identity': hsp.identities,
                'align_length': hsp.align_length,
                'query_coverage': f"{query_coverage:.1f}",
                'identity_percent': f"{identity_percent:.1f}",
                'gaps': hsp.gaps
            }
            
            results.append(result)
            
            # 只记录前5个最显著的结果到日志
            if len(results) <= 5:
                logger.info(f"    命中: {alignment.hit_def[:60]}... E={hsp.expect:.2e}")
    
    return results


def save_blast_details(blast_records: list, output_file: Path):
    """
    保存 BLAST 详细结果
    
    参数:
        blast_records: BLAST记录列表
        output_file: 输出文件
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("BLAST 搜索详细结果\n")
        f.write("作者: Kuroneko\n")
        f.write("日期: 2025.10.04\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*70 + "\n\n")
        
        for seq_id, record, success in blast_records:
            f.write(f"查询序列: {seq_id}\n")
            f.write("-"*70 + "\n")
            
            if not success:
                f.write("状态: 查询失败\n\n")
                continue
            
            if not record or not record.alignments:
                f.write("状态: 无显著比对结果\n\n")
                continue
            
            f.write(f"数据库: {record.database}\n")
            f.write(f"查询长度: {record.query_length}\n")
            f.write(f"找到 {len(record.alignments)} 个比对\n\n")
            
            for i, alignment in enumerate(record.alignments[:10], 1):  # 只保存前10个
                f.write(f"  [{i}] {alignment.hit_def}\n")
                f.write(f"      ID: {alignment.hit_id}\n")
                f.write(f"      长度: {alignment.length}\n")
                
                if alignment.hsps:
                    hsp = alignment.hsps[0]
                    f.write(f"      E-value: {hsp.expect:.2e}\n")
                    f.write(f"      Score: {hsp.score} bits ({hsp.bits})\n")
                    f.write(f"      Identities: {hsp.identities}/{hsp.align_length} "
                           f"({hsp.identities/hsp.align_length*100:.1f}%)\n")
                    f.write(f"      Gaps: {hsp.gaps}/{hsp.align_length}\n")
                f.write("\n")
            
            f.write("\n" + "="*70 + "\n\n")


def main():
    """
    主函数
    """
    print_section_header("步骤 3: 远程 BLAST 搜索与结果解析")
    print("作者: Kuroneko")
    print("日期: 2025.10.04\n")
    
    # 设置日志
    logger = setup_logger('step3_blast')
    
    # 隐私警告
    if not privacy_warning_blast():
        logger.info("用户拒绝上传序列，退出")
        print("\n已取消 BLAST 搜索")
        sys.exit(0)
    
    # 查找输入文件
    print_section_header("加载输入序列")
    
    input_candidates = [
        OUTPUT_DIR / "seg_masked.fasta",
        DATA_DIR / "input_sequences.fasta",
        DATA_DIR / "example_public_uniprot.fasta",
        DATA_DIR / "example_synthetic.fasta"
    ]
    
    input_file = None
    for candidate in input_candidates:
        if candidate.exists() and candidate.stat().st_size > 0:
            input_file = candidate
            break
    
    if input_file is None:
        logger.error("未找到有效的输入序列文件")
        print("\n错误: 未找到输入序列文件")
        print("请先运行 step2_seq_check_and_seg.py")
        sys.exit(1)
    
    logger.info(f"使用输入文件: {input_file}")
    print(f"输入文件: {input_file.name}")
    
    # 读取序列
    try:
        records = read_fasta(input_file)
        logger.info(f"成功读取 {len(records)} 条序列")
        print(f"读取到 {len(records)} 条序列\n")
    except Exception as e:
        logger.error(f"读取文件失败: {e}")
        print(f"\n错误: {e}")
        sys.exit(1)
    
    # 检测合成序列
    synthetic_count = sum(1 for r in records if detect_sequence_source(r['id']) == 'synthetic')
    if synthetic_count > 0:
        print(f"\n注意: 检测到 {synthetic_count} 条合成序列")
        print("合成序列是计算机生成的，不太可能在公共数据库中找到同源序列。")
        
        if not confirm_action("是否继续对合成序列进行 BLAST 搜索？", default_yes=False):
            logger.info("用户选择跳过合成序列的 BLAST")
            # 过滤掉合成序列
            records = [r for r in records if detect_sequence_source(r['id']) != 'synthetic']
            if len(records) == 0:
                print("\n没有非合成序列可供搜索，退出")
                sys.exit(0)
            print(f"\n将对剩余 {len(records)} 条序列进行 BLAST 搜索")
    
    # BLAST 参数配置
    print_section_header("BLAST 参数配置")
    
    # 检测序列类型（蛋白或核酸）
    first_seq_check = check_sequence_validity(records[0]['sequence'], 'auto')
    is_protein = first_seq_check['type'] == 'protein'
    
    default_program = 'blastp' if is_protein else 'blastn'
    default_database = 'nr' if is_protein else 'nt'
    
    print(f"\n检测到序列类型: {'蛋白质' if is_protein else '核酸'}")
    
    blast_program = get_user_input(
        f"BLAST 程序 (blastp/blastn/blastx等)",
        default=default_program
    )
    
    database = get_user_input(
        f"数据库 (nr/nt/pdb/swissprot等)",
        default=default_database
    )
    
    max_hits = int(get_user_input(
        "最多返回命中数",
        default="10"
    ))
    
    # 智能选择序列进行BLAST
    print_section_header("序列选择策略")
    print("\n检测到多条序列，请选择BLAST策略：")
    print("1. 全部序列分别BLAST（耗时较长，结果最全面）")
    print("2. 智能选择代表性序列（推荐，平衡效率与覆盖度）")
    print("3. 仅选择前3条序列（快速测试）")
    print("4. 手动选择特定序列")
    
    strategy = get_user_input("请选择策略", default="2", valid_options=['1', '2', '3', '4'])
    
    if strategy == '1':
        # 全部序列
        selected_records = records
        print(f"\n将对所有 {len(records)} 条序列进行BLAST搜索")
    elif strategy == '2':
        # 智能选择
        selected_records = select_representative_sequences(records, logger)
        print(f"\n智能选择了 {len(selected_records)} 条代表性序列")
    elif strategy == '3':
        # 前3条
        selected_records = records[:3]
        print(f"\n将对前3条序列进行BLAST搜索")
    else:
        # 手动选择
        selected_records = manual_select_sequences(records)
        print(f"\n手动选择了 {len(selected_records)} 条序列")
    
    records = selected_records
    
    # 执行 BLAST 搜索
    print_section_header(f"执行 BLAST 搜索（共 {len(records)} 条序列）")
    print("提示: BLAST 远程搜索可能需要较长时间，请耐心等待...\n")
    
    all_results = []
    blast_details = []
    
    for i, record in enumerate(records, 1):
        seq_id = record['id']
        sequence = record['sequence']
        source_type = detect_sequence_source(seq_id)
        
        print(f"[{i}/{len(records)}] 查询序列: {seq_id} (来源: {source_type})")
        
        # 运行 BLAST
        blast_result = run_blast_search(
            sequence, seq_id, blast_program, database, max_hits, logger
        )
        
        if blast_result['success']:
            # 解析结果
            parsed_results = parse_blast_results(blast_result['record'], seq_id, logger)
            all_results.extend(parsed_results)
            
            blast_details.append((seq_id, blast_result['record'], True))
            
            if len(parsed_results) == 0:
                print(f"  结果: 无显著命中\n")
            else:
                print(f"  结果: 找到 {len(parsed_results)} 个命中\n")
        else:
            logger.error(f"  {seq_id} BLAST 失败: {blast_result['error']}")
            print(f"  结果: 查询失败 - {blast_result['error']}\n")
            blast_details.append((seq_id, None, False))
        
        # 礼貌延迟（避免频繁请求）
        if i < len(records):
            wait_time = 3
            print(f"  等待 {wait_time} 秒后继续下一个查询...")
            time.sleep(wait_time)
    
    # 保存结果
    print_section_header("保存结果")
    
    # 保存CSV摘要
    if all_results:
        csv_file = OUTPUT_DIR / "blast_results.csv"
        logger.info(f"保存结果到: {csv_file}")
        
        with open(csv_file, 'w', encoding='utf-8', newline='') as f:
            fieldnames = ['query_id', 'hit_id', 'hit_def', 'e_value', 'bit_score',
                          'identity', 'align_length', 'query_coverage', 
                          'identity_percent', 'gaps']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        
        print(f"结果摘要已保存: {csv_file}")
        print(f"  共 {len(all_results)} 条BLAST命中记录")
    else:
        print("所有查询均无显著结果")
    
    # 保存详细结果
    details_file = OUTPUT_DIR / f"blast_details_{format_timestamp()}.txt"
    save_blast_details(blast_details, details_file)
    logger.info(f"详细结果已保存: {details_file}")
    print(f"详细结果已保存: {details_file}")
    
    # 完成
    print_section_header("BLAST 搜索完成")
    
    print("\n下一步:")
    print("  运行 step4_gff_genbank_parse.py 进行注释文件解析")
    print("  命令: python scripts/step4_gff_genbank_parse.py")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断操作")
        sys.exit(0)
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


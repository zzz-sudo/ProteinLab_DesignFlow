#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step2_seq_check_and_seg.py - 序列质量检查与SEG掩蔽
作者: Kuroneko
日期: 2025.10.04

功能说明:
    1. 读取 FASTA 格式序列文件
    2. 检查序列质量：非标准字符、长度、低复杂度区域
    3. 执行 SEG 算法掩蔽低复杂度区域（可选）
    4. 生成序列检查报告（CSV格式）
    5. 输出掩蔽后的序列（FASTA格式）
    6. 记录数据来源类型（公开/合成/用户）

输入输出:
    输入:
        - data/input_sequences.fasta（主要）
        - data/example_public_uniprot.fasta（备选）
        - data/example_synthetic.fasta（备选）
    输出:
        - outputs/seq_check.csv（序列检查报告）
        - outputs/seg_masked.fasta（SEG掩蔽后序列）
        - outputs/sample_seg_examples.txt（掩蔽示例）

运行示例:
    python step2_seq_check_and_seg.py

设计决策:
    - SEG 算法使用简化版本（检测单一氨基酸高频区域）
    - 标准 SEG 需要外部程序，这里实现纯 Python 版本
    - 备选方案：可安装 seg 程序或使用 BioPython 的 SEG 封装
"""

import os
import sys
import csv
from pathlib import Path
from collections import Counter

# 导入工具模块
try:
    from step7_utils import (
        setup_logger, get_user_input, confirm_action,
        print_section_header, check_sequence_validity,
        read_fasta, write_fasta, DATA_DIR, OUTPUT_DIR
    )
except ImportError:
    print("错误: 无法导入 step7_utils 模块")
    print("请确保 step7_utils.py 与本脚本在同一目录下")
    sys.exit(1)


def detect_source_type(fasta_file: Path) -> str:
    """
    检测序列来源类型
    
    参数:
        fasta_file: FASTA 文件路径
    
    返回:
        'public', 'synthetic', 或 'user'
    """
    if 'public' in fasta_file.name.lower():
        return 'public'
    elif 'synthetic' in fasta_file.name.lower() or 'synth' in fasta_file.name.lower():
        return 'synthetic'
    else:
        return 'user'


def simple_seg_mask(sequence: str, window: int = 12, threshold: float = 0.5, 
                    mask_char: str = 'X') -> tuple:
    """
    简化的 SEG 算法实现 - 掩蔽低复杂度区域
    
    标准 SEG 算法基于序列局部的组成复杂度，这里实现一个简化版本：
    在滑动窗口内，如果某个氨基酸频率超过阈值，则标记为低复杂度
    
    参数:
        sequence: 蛋白序列
        window: 滑动窗口大小
        threshold: 单一氨基酸频率阈值
        mask_char: 掩蔽字符
    
    返回:
        (masked_sequence, mask_positions, mask_count)
    """
    seq_upper = sequence.upper()
    seq_list = list(seq_upper)
    mask_positions = []
    
    # 滑动窗口检测
    for i in range(len(seq_upper) - window + 1):
        window_seq = seq_upper[i:i+window]
        counter = Counter(window_seq)
        most_common_aa, count = counter.most_common(1)[0]
        
        if count / window >= threshold:
            # 标记整个窗口为低复杂度
            for j in range(i, i + window):
                if j not in mask_positions:
                    mask_positions.append(j)
                    seq_list[j] = mask_char.lower()  # 使用小写表示掩蔽
    
    masked_seq = ''.join(seq_list)
    return masked_seq, sorted(mask_positions), len(mask_positions)


def calculate_composition_complexity(sequence: str, window: int = 12) -> float:
    """
    计算序列复杂度（基于Shannon熵的简化版本）
    
    参数:
        sequence: 序列
        window: 窗口大小
    
    返回:
        平均复杂度分数（0-1，越高越复杂）
    """
    import math
    
    if len(sequence) == 0:
        return 0.0
    
    if len(sequence) < window:
        window = len(sequence)
    
    if window <= 0:
        return 0.0
    
    complexities = []
    for i in range(len(sequence) - window + 1):
        window_seq = sequence[i:i+window]
        counter = Counter(window_seq)
        
        # Shannon熵计算
        entropy = 0
        for count in counter.values():
            if count > 0:
                p = count / window
                entropy -= p * math.log2(p)
        
        # 归一化（最大熵为 log2(min(20, window)) 对于蛋白质）
        max_entropy = math.log2(min(20, window)) if min(20, window) > 0 else 1.0
        normalized = entropy / max_entropy if max_entropy > 0 else 0
        complexities.append(normalized)
    
    return sum(complexities) / len(complexities) if complexities else 0.0


def check_sequences(records: list, source_type: str, logger) -> list:
    """
    检查所有序列并生成报告，支持多序列和特殊字符处理
    
    参数:
        records: 序列记录列表
        source_type: 数据来源类型
        logger: 日志记录器
    
    返回:
        检查结果列表（字典）
    """
    results = []
    
    logger.info(f"开始检查 {len(records)} 条序列...")
    
    for i, record in enumerate(records, 1):
        seq_id = record['id']
        sequence = record['sequence']
        
        logger.info(f"检查序列 {i}/{len(records)}: {seq_id}")
        
        # 基本验证
        validation = check_sequence_validity(sequence, 'auto')
        
        # 复杂度分析
        complexity = calculate_composition_complexity(validation['sequence'])
        
        # 统计信息
        aa_counter = Counter(validation['sequence'])
        most_common_aa = aa_counter.most_common(1)[0] if aa_counter else ('N/A', 0)
        most_common_ratio = most_common_aa[1] / validation['length'] if validation['length'] > 0 else 0
        
        # 特殊字符统计
        special_chars_str = ', '.join(sorted(validation.get('special_chars', set()))) if validation.get('special_chars') else 'None'
        
        # 组装结果
        result = {
            'sequence_id': seq_id,
            'source_type': source_type,
            'length': validation['length'],
            'original_length': validation.get('original_length', validation['length']),
            'detected_type': validation['type'],
            'is_valid': validation['valid'],
            'issues': '; '.join(validation['issues']) if validation['issues'] else 'None',
            'complexity_score': f"{complexity:.3f}",
            'most_common_aa': most_common_aa[0],
            'most_common_ratio': f"{most_common_ratio:.3f}",
            'special_chars': special_chars_str,
            'description': record.get('description', '')
        }
        
        results.append(result)
        
        # 输出简要信息
        status = "通过" if validation['valid'] else "警告"
        logger.info(f"  状态: {status}, 长度: {validation['length']}, 复杂度: {complexity:.3f}")
        if validation['issues']:
            for issue in validation['issues']:
                logger.warning(f"    问题: {issue}")
        if validation.get('special_chars'):
            logger.info(f"    特殊字符: {special_chars_str}")
    
    # 统计汇总
    valid_count = sum(1 for r in results if r['is_valid'])
    logger.info(f"\n序列检查完成: {valid_count}/{len(results)} 条序列通过验证")
    
    return results


def apply_seg_masking(records: list, logger) -> tuple:
    """
    对所有序列应用 SEG 掩蔽
    
    参数:
        records: 序列记录列表
        logger: 日志记录器
    
    返回:
        (masked_records, masking_stats)
    """
    masked_records = []
    masking_stats = []
    
    for record in records:
        seq_id = record['id']
        original_seq = record['sequence']
        
        # 应用 SEG 掩蔽
        masked_seq, mask_positions, mask_count = simple_seg_mask(original_seq)
        
        masked_record = {
            'id': seq_id,
            'description': record['description'] + ' [SEG-masked]',
            'sequence': masked_seq
        }
        masked_records.append(masked_record)
        
        # 统计信息
        mask_ratio = mask_count / len(original_seq) if len(original_seq) > 0 else 0
        stat = {
            'sequence_id': seq_id,
            'original_length': len(original_seq),
            'masked_positions': mask_count,
            'mask_ratio': mask_ratio
        }
        masking_stats.append(stat)
        
        logger.info(f"SEG 掩蔽 {seq_id}: {mask_count} 位置 ({mask_ratio*100:.1f}%)")
    
    return masked_records, masking_stats


def save_seg_examples(records: list, masked_records: list, output_file: Path, max_examples: int = 3):
    """
    保存 SEG 掩蔽前后对比示例
    
    参数:
        records: 原始序列
        masked_records: 掩蔽后序列
        output_file: 输出文件
        max_examples: 最多保存几个示例
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("SEG 掩蔽前后对比示例\n")
        f.write("作者: Kuroneko\n")
        f.write("日期: 2025.10.04\n")
        f.write("="*70 + "\n\n")
        
        f.write("说明:\n")
        f.write("  - SEG 算法用于检测并掩蔽蛋白序列中的低复杂度区域\n")
        f.write("  - 低复杂度区域通常富含某种氨基酸（如 poly-A, proline-rich）\n")
        f.write("  - 掩蔽字符：'x'（小写）表示低复杂度位置\n")
        f.write("  - 本实现为简化版 SEG，检测滑动窗口内单一氨基酸高频区域\n\n")
        
        for i, (orig, masked) in enumerate(zip(records[:max_examples], masked_records[:max_examples]), 1):
            f.write(f"示例 {i}: {orig['id']}\n")
            f.write("-"*70 + "\n")
            f.write(f"描述: {orig['description']}\n")
            f.write(f"长度: {len(orig['sequence'])} aa\n\n")
            
            # 显示前100个字符的对比
            display_len = min(100, len(orig['sequence']))
            f.write("原始序列（前100个残基）:\n")
            f.write(f"{orig['sequence'][:display_len]}\n\n")
            f.write("掩蔽后序列（前100个残基）:\n")
            f.write(f"{masked['sequence'][:display_len]}\n\n")
            
            # 统计掩蔽情况
            mask_count = sum(1 for c in masked['sequence'] if c.islower())
            mask_ratio = mask_count / len(masked['sequence']) * 100
            f.write(f"掩蔽统计: {mask_count}/{len(masked['sequence'])} 位置 ({mask_ratio:.1f}%)\n")
            f.write("\n" + "="*70 + "\n\n")


def main():
    """
    主函数
    """
    print_section_header("步骤 2: 序列质量检查与 SEG 掩蔽")
    print("作者: Kuroneko")
    print("日期: 2025.10.04\n")
    
    # 设置日志
    logger = setup_logger('step2_seq_check')
    
    # 查找输入文件
    print_section_header("加载输入序列")
    
    input_candidates = [
        DATA_DIR / "input_sequences.fasta",
        DATA_DIR / "example_multi_sequences.fasta",
        DATA_DIR / "example_public_uniprot.fasta",
        DATA_DIR / "example_synthetic.fasta"
    ]
    
    input_file = None
    for candidate in input_candidates:
        if candidate.exists() and candidate.stat().st_size > 0:
            # 检查是否为空文件或只有注释
            with open(candidate, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content and not content.startswith('#'):
                    input_file = candidate
                    break
    
    if input_file is None:
        logger.error("未找到有效的输入序列文件")
        print("\n错误: 未找到包含序列的输入文件")
        print("请先运行 step1_setup_and_download.py 准备数据")
        sys.exit(1)
    
    logger.info(f"使用输入文件: {input_file}")
    print(f"输入文件: {input_file.name}")
    
    # 读取序列
    try:
        records = read_fasta(input_file)
        logger.info(f"成功读取 {len(records)} 条序列")
        print(f"读取到 {len(records)} 条序列\n")
    except Exception as e:
        logger.error(f"读取 FASTA 文件失败: {e}")
        print(f"\n错误: {e}")
        sys.exit(1)
    
    if len(records) == 0:
        logger.error("文件中没有有效序列")
        print("错误: 文件中没有有效序列")
        sys.exit(1)
    
    # 检测数据来源
    source_type = detect_source_type(input_file)
    logger.info(f"数据来源类型: {source_type}")
    
    # 序列质量检查
    print_section_header("序列质量检查")
    check_results = check_sequences(records, source_type, logger)
    
    # 保存检查报告
    report_file = OUTPUT_DIR / "seq_check.csv"
    logger.info(f"保存检查报告到: {report_file}")
    
    with open(report_file, 'w', encoding='utf-8', newline='') as f:
        fieldnames = ['sequence_id', 'source_type', 'length', 'original_length', 'detected_type', 
                      'is_valid', 'issues', 'complexity_score', 
                      'most_common_aa', 'most_common_ratio', 'special_chars', 'description']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(check_results)
    
    print(f"\n检查报告已保存: {report_file}")
    
    # 询问是否执行 SEG 掩蔽
    print_section_header("SEG 掩蔽")
    print("\nSEG 算法用于检测和掩蔽蛋白序列中的低复杂度区域。")
    print("低复杂度区域可能影响序列比对和数据库搜索的准确性。")
    
    perform_seg = confirm_action("\n是否执行 SEG 掩蔽？", default_yes=True)
    
    if perform_seg:
        # 执行 SEG 掩蔽
        masked_records, masking_stats = apply_seg_masking(records, logger)
        
        # 保存掩蔽后的序列
        masked_file = OUTPUT_DIR / "seg_masked.fasta"
        write_fasta(masked_records, masked_file)
        logger.info(f"掩蔽序列已保存: {masked_file}")
        print(f"\n掩蔽序列已保存: {masked_file}")
        
        # 保存掩蔽示例
        examples_file = OUTPUT_DIR / "sample_seg_examples.txt"
        save_seg_examples(records, masked_records, examples_file)
        logger.info(f"掩蔽示例已保存: {examples_file}")
        print(f"掩蔽示例已保存: {examples_file}")
        
        # 显示统计
        print("\nSEG 掩蔽统计:")
        for stat in masking_stats:
            print(f"  {stat['sequence_id']}: {stat['masked_positions']} 位置掩蔽 "
                  f"({stat['mask_ratio']*100:.1f}%)")
    else:
        logger.info("用户选择跳过 SEG 掩蔽")
        print("已跳过 SEG 掩蔽")
    
    # 完成
    print_section_header("序列检查完成")
    
    print("\n生成文件:")
    print(f"  - {report_file}")
    if perform_seg:
        print(f"  - {OUTPUT_DIR / 'seg_masked.fasta'}")
        print(f"  - {OUTPUT_DIR / 'sample_seg_examples.txt'}")
    
    print("\n下一步:")
    print("  运行 step3_blast_remote_and_parse.py 进行 BLAST 搜索")
    print("  命令: python scripts/step3_blast_remote_and_parse.py")


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


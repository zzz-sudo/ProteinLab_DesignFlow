#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step4_gff_genbank_parse.py - GFF和GenBank注释文件解析
作者: Kuroneko
日期: 2025.9.06

功能说明:
    1. 解析 GFF3 格式基因注释文件
    2. 解析 GenBank 格式记录文件
    3. 提取基因特征（CDS, mRNA, exon等）
    4. 输出注释信息摘要（CSV格式）
    5. 支持多种来源的注释文件

输入输出:
    输入:
        - data/example_public_genbank.gb（若存在）
        - data/*.gff3（用户提供的GFF文件）
        - data/*.gb, data/*.gbk（用户提供的GenBank文件）
    输出:
        - outputs/gff_parsed.csv（GFF解析结果）
        - outputs/gbk_parsed.csv（GenBank解析结果）

运行示例:
    python step4_gff_genbank_parse.py

设计决策:
    - 使用 Biopython SeqIO 解析 GenBank 格式
    - GFF3 解析采用纯 Python 实现（无需外部库）
    - 提取主要特征类型：gene, CDS, mRNA, exon, tRNA等
"""

import os
import sys
import csv
from pathlib import Path

# 导入工具模块
try:
    from step7_utils import (
        setup_logger, get_user_input, confirm_action,
        print_section_header, DATA_DIR, OUTPUT_DIR
    )
except ImportError:
    print("错误: 无法导入 step7_utils 模块")
    print("请确保 step7_utils.py 与本脚本在同一目录下")
    sys.exit(1)


def parse_gff3_line(line: str) -> dict:
    """
    解析 GFF3 格式的一行
    
    参数:
        line: GFF3 行文本
    
    返回:
        特征字典或 None（如果是注释行）
    """
    line = line.strip()
    
    # 跳过注释和空行
    if not line or line.startswith('#'):
        return None
    
    # GFF3 格式: seqid source type start end score strand phase attributes
    parts = line.split('\t')
    if len(parts) != 9:
        return None
    
    # 解析属性
    attributes = {}
    for attr in parts[8].split(';'):
        if '=' in attr:
            key, value = attr.split('=', 1)
            attributes[key] = value
    
    return {
        'seqid': parts[0],
        'source': parts[1],
        'type': parts[2],
        'start': int(parts[3]) if parts[3].isdigit() else 0,
        'end': int(parts[4]) if parts[4].isdigit() else 0,
        'score': parts[5],
        'strand': parts[6],
        'phase': parts[7],
        'attributes': attributes
    }


def parse_gff3_file(gff_file: Path, logger) -> list:
    """
    解析完整 GFF3 文件
    
    参数:
        gff_file: GFF3 文件路径
        logger: 日志记录器
    
    返回:
        特征列表
    """
    logger.info(f"解析 GFF3 文件: {gff_file}")
    features = []
    
    try:
        with open(gff_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                feature = parse_gff3_line(line)
                if feature:
                    feature['line_number'] = line_num
                    features.append(feature)
        
        logger.info(f"  成功解析 {len(features)} 个特征")
        return features
        
    except Exception as e:
        logger.error(f"  解析失败: {e}")
        return []


def parse_genbank_file(gbk_file: Path, logger) -> list:
    """
    解析 GenBank 文件
    
    参数:
        gbk_file: GenBank 文件路径
        logger: 日志记录器
    
    返回:
        特征列表
    """
    try:
        from Bio import SeqIO
    except ImportError:
        logger.error("需要安装 Biopython: pip install biopython")
        return []
    
    logger.info(f"解析 GenBank 文件: {gbk_file}")
    all_features = []
    
    try:
        for record in SeqIO.parse(gbk_file, 'genbank'):
            logger.info(f"  记录: {record.id}, 长度: {len(record.seq)} bp")
            
            for feature in record.features:
                # 提取特征信息
                feature_dict = {
                    'record_id': record.id,
                    'record_description': record.description[:100],
                    'type': feature.type,
                    'start': int(feature.location.start) if feature.location else 0,
                    'end': int(feature.location.end) if feature.location else 0,
                    'strand': feature.location.strand if feature.location else 0,
                    'qualifiers': {}
                }
                
                # 提取常用的 qualifiers
                for key in ['gene', 'product', 'protein_id', 'locus_tag', 'note']:
                    if key in feature.qualifiers:
                        value = feature.qualifiers[key]
                        feature_dict['qualifiers'][key] = value[0] if isinstance(value, list) else value
                
                all_features.append(feature_dict)
        
        logger.info(f"  成功解析 {len(all_features)} 个特征")
        return all_features
        
    except Exception as e:
        logger.error(f"  解析失败: {e}")
        return []


def save_gff_results(features: list, output_file: Path, logger):
    """
    保存 GFF 解析结果为 CSV
    
    参数:
        features: 特征列表
        output_file: 输出文件
        logger: 日志记录器
    """
    if not features:
        logger.warning("没有 GFF 特征可保存")
        return
    
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        fieldnames = ['seqid', 'source', 'type', 'start', 'end', 'strand', 
                      'score', 'phase', 'ID', 'Name', 'Parent', 'gene', 'product']
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        
        for feature in features:
            row = {
                'seqid': feature['seqid'],
                'source': feature['source'],
                'type': feature['type'],
                'start': feature['start'],
                'end': feature['end'],
                'strand': feature['strand'],
                'score': feature['score'],
                'phase': feature['phase'],
            }
            
            # 添加属性
            attrs = feature.get('attributes', {})
            row['ID'] = attrs.get('ID', '')
            row['Name'] = attrs.get('Name', '')
            row['Parent'] = attrs.get('Parent', '')
            row['gene'] = attrs.get('gene', '')
            row['product'] = attrs.get('product', '')
            
            writer.writerow(row)
    
    logger.info(f"GFF 结果已保存: {output_file}")


def save_genbank_results(features: list, output_file: Path, logger):
    """
    保存 GenBank 解析结果为 CSV
    
    参数:
        features: 特征列表
        output_file: 输出文件
        logger: 日志记录器
    """
    if not features:
        logger.warning("没有 GenBank 特征可保存")
        return
    
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        fieldnames = ['record_id', 'record_description', 'type', 'start', 'end', 
                      'strand', 'gene', 'product', 'protein_id', 'locus_tag', 'note']
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        
        for feature in features:
            row = {
                'record_id': feature['record_id'],
                'record_description': feature['record_description'],
                'type': feature['type'],
                'start': feature['start'],
                'end': feature['end'],
                'strand': feature['strand'],
            }
            
            # 添加 qualifiers
            qualifiers = feature.get('qualifiers', {})
            row['gene'] = qualifiers.get('gene', '')
            row['product'] = qualifiers.get('product', '')
            row['protein_id'] = qualifiers.get('protein_id', '')
            row['locus_tag'] = qualifiers.get('locus_tag', '')
            row['note'] = qualifiers.get('note', '')
            
            writer.writerow(row)
    
    logger.info(f"GenBank 结果已保存: {output_file}")


def save_genbank_readable_report(features: list, output_file: Path, logger):
    """
    保存 GenBank 易读报告
    
    参数:
        features: 特征列表
        output_file: 输出文件
        logger: 日志记录器
    """
    if not features:
        logger.warning("没有 GenBank 特征可保存")
        return
    
    # 按类型分组特征
    features_by_type = {}
    for feature in features:
        ftype = feature['type']
        if ftype not in features_by_type:
            features_by_type[ftype] = []
        features_by_type[ftype].append(feature)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("GenBank 注释文件易读报告\n")
        f.write("作者: Kuroneko\n")
        f.write("日期: 2025.10.04\n")
        f.write("="*70 + "\n\n")
        
        # 基本信息
        if features:
            first_feature = features[0]
            f.write("基本信息\n")
            f.write("-"*70 + "\n")
            f.write(f"记录ID: {first_feature['record_id']}\n")
            f.write(f"描述: {first_feature['record_description']}\n")
            f.write(f"总特征数: {len(features)}\n\n")
        
        # 按重要性排序特征类型
        important_types = ['gene', 'CDS', 'mRNA', 'exon', 'tRNA', 'rRNA']
        other_types = [t for t in features_by_type.keys() if t not in important_types]
        ordered_types = important_types + other_types
        
        for ftype in ordered_types:
            if ftype not in features_by_type:
                continue
                
            type_features = features_by_type[ftype]
            f.write(f"{ftype.upper()} 特征 ({len(type_features)} 个)\n")
            f.write("-"*70 + "\n")
            
            for i, feature in enumerate(type_features, 1):
                f.write(f"\n[{i}] 位置: {feature['start']}-{feature['end']} (链: {feature['strand']})\n")
                
                qualifiers = feature.get('qualifiers', {})
                if qualifiers.get('gene'):
                    f.write(f"    基因: {qualifiers['gene']}\n")
                if qualifiers.get('product'):
                    f.write(f"    产物: {qualifiers['product']}\n")
                if qualifiers.get('protein_id'):
                    f.write(f"    蛋白ID: {qualifiers['protein_id']}\n")
                if qualifiers.get('locus_tag'):
                    f.write(f"    位点标签: {qualifiers['locus_tag']}\n")
                if qualifiers.get('note'):
                    note = qualifiers['note']
                    if len(note) > 100:
                        note = note[:100] + "..."
                    f.write(f"    注释: {note}\n")
            
            f.write("\n" + "="*70 + "\n\n")
    
    logger.info(f"GenBank 易读报告已保存: {output_file}")


def save_gff_readable_report(features: list, output_file: Path, logger):
    """
    保存 GFF 易读报告
    
    参数:
        features: 特征列表
        output_file: 输出文件
        logger: 日志记录器
    """
    if not features:
        logger.warning("没有 GFF 特征可保存")
        return
    
    # 按类型分组特征
    features_by_type = {}
    for feature in features:
        ftype = feature['type']
        if ftype not in features_by_type:
            features_by_type[ftype] = []
        features_by_type[ftype].append(feature)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("GFF 注释文件易读报告\n")
        f.write("作者: Kuroneko\n")
        f.write("日期: 2025.10.04\n")
        f.write("="*70 + "\n\n")
        
        # 基本信息
        f.write("基本信息\n")
        f.write("-"*70 + "\n")
        f.write(f"总特征数: {len(features)}\n")
        f.write(f"序列ID: {features[0]['seqid'] if features else 'N/A'}\n\n")
        
        # 按重要性排序特征类型
        important_types = ['gene', 'CDS', 'mRNA', 'exon', 'tRNA', 'rRNA']
        other_types = [t for t in features_by_type.keys() if t not in important_types]
        ordered_types = important_types + other_types
        
        for ftype in ordered_types:
            if ftype not in features_by_type:
                continue
                
            type_features = features_by_type[ftype]
            f.write(f"{ftype.upper()} 特征 ({len(type_features)} 个)\n")
            f.write("-"*70 + "\n")
            
            for i, feature in enumerate(type_features, 1):
                f.write(f"\n[{i}] 位置: {feature['start']}-{feature['end']} (链: {feature['strand']})\n")
                f.write(f"    来源: {feature['source']}\n")
                f.write(f"    得分: {feature['score']}\n")
                f.write(f"    相位: {feature['phase']}\n")
                
                attrs = feature.get('attributes', {})
                if attrs.get('ID'):
                    f.write(f"    ID: {attrs['ID']}\n")
                if attrs.get('Name'):
                    f.write(f"    名称: {attrs['Name']}\n")
                if attrs.get('Parent'):
                    f.write(f"    父级: {attrs['Parent']}\n")
                if attrs.get('gene'):
                    f.write(f"    基因: {attrs['gene']}\n")
                if attrs.get('product'):
                    f.write(f"    产物: {attrs['product']}\n")
            
            f.write("\n" + "="*70 + "\n\n")
    
    logger.info(f"GFF 易读报告已保存: {output_file}")


def main():
    """
    主函数
    """
    print_section_header("步骤 4: GFF 和 GenBank 注释文件解析")
    print("作者: Kuroneko")
    print("日期: 2025.10.04\n")
    
    # 设置日志
    logger = setup_logger('step4_annotation_parse')
    
    # 查找 GFF 文件
    print_section_header("查找注释文件")
    
    gff_files = list(DATA_DIR.glob("*.gff3")) + list(DATA_DIR.glob("*.gff"))
    gbk_files = list(DATA_DIR.glob("*.gb")) + list(DATA_DIR.glob("*.gbk"))
    
    print(f"找到 {len(gff_files)} 个 GFF 文件")
    print(f"找到 {len(gbk_files)} 个 GenBank 文件")
    
    if len(gff_files) == 0 and len(gbk_files) == 0:
        logger.warning("未找到注释文件")
        print("\n注意: 未在 data/ 目录找到 GFF 或 GenBank 文件")
        print("请将注释文件放置在 data/ 目录下")
        print("  - GFF 文件: *.gff 或 *.gff3")
        print("  - GenBank 文件: *.gb 或 *.gbk")
        
        if not confirm_action("\n是否创建示例 GenBank 文件用于测试？", default_yes=True):
            print("退出")
            sys.exit(0)
        
        # 创建简单示例
        example_gbk = DATA_DIR / "example_minimal.gb"
        with open(example_gbk, 'w', encoding='utf-8') as f:
            f.write("""LOCUS       EXAMPLE_GENE            500 bp    DNA     linear   SYN 04-OCT-2025
DEFINITION  Synthetic example for testing.
ACCESSION   EXAMPLE001
VERSION     EXAMPLE001.1
KEYWORDS    .
SOURCE      synthetic construct
  ORGANISM  synthetic construct
            other sequences; artificial sequences.
FEATURES             Location/Qualifiers
     source          1..500
                     /organism="synthetic construct"
                     /mol_type="genomic DNA"
     gene            100..400
                     /gene="exampleGene"
     CDS             100..400
                     /gene="exampleGene"
                     /product="example protein"
                     /translation="MKKLLILSLL"
ORIGIN      
        1 atgatgatga tgatgatgat gatgatgatg atgatgatga tgatgatgat gatgatgatg
       61 atgatgatga tgatgatgat gatgatgatg atgatgatga tgatgatgat gatgatgatg
      121 atgatgatga tgatgatgat gatgatgatg atgatgatga tgatgatgat gatgatgatg
      181 atgatgatga tgatgatgat gatgatgatg atgatgatga tgatgatgat gatgatgatg
      241 atgatgatga tgatgatgat gatgatgatg atgatgatga tgatgatgat gatgatgatg
      301 atgatgatga tgatgatgat gatgatgatg atgatgatga tgatgatgat gatgatgatg
      361 atgatgatga tgatgatgat gatgatgatg atgatgatga tgatgatgat gatgatgatg
      421 atgatgatga tgatgatgat gatgatgatg atgatgatga tgatgatgat gatgatgatg
      481 atgatgatga tgatgatgat
//
""")
        logger.info(f"创建示例文件: {example_gbk}")
        gbk_files = [example_gbk]
    
    # 解析 GFF 文件
    all_gff_features = []
    if gff_files:
        print_section_header("解析 GFF 文件")
        for gff_file in gff_files:
            print(f"\n处理: {gff_file.name}")
            features = parse_gff3_file(gff_file, logger)
            all_gff_features.extend(features)
        
        if all_gff_features:
            # 保存CSV格式
            csv_file = OUTPUT_DIR / "gff_parsed.csv"
            save_gff_results(all_gff_features, csv_file, logger)
            
            # 保存易读报告
            readable_file = OUTPUT_DIR / "gff_readable_report.txt"
            save_gff_readable_report(all_gff_features, readable_file, logger)
            
            print(f"\nGFF 解析完成，共 {len(all_gff_features)} 个特征")
            print(f"CSV结果已保存: {csv_file}")
            print(f"易读报告已保存: {readable_file}")
    
    # 解析 GenBank 文件
    all_gbk_features = []
    if gbk_files:
        print_section_header("解析 GenBank 文件")
        for gbk_file in gbk_files:
            print(f"\n处理: {gbk_file.name}")
            features = parse_genbank_file(gbk_file, logger)
            all_gbk_features.extend(features)
        
        if all_gbk_features:
            # 保存CSV格式
            csv_file = OUTPUT_DIR / "gbk_parsed.csv"
            save_genbank_results(all_gbk_features, csv_file, logger)
            
            # 保存易读报告
            readable_file = OUTPUT_DIR / "gbk_readable_report.txt"
            save_genbank_readable_report(all_gbk_features, readable_file, logger)
            
            print(f"\nGenBank 解析完成，共 {len(all_gbk_features)} 个特征")
            print(f"CSV结果已保存: {csv_file}")
            print(f"易读报告已保存: {readable_file}")
            
            # 统计特征类型
            from collections import Counter
            type_counts = Counter(f['type'] for f in all_gbk_features)
            print("\n特征类型统计:")
            for ftype, count in type_counts.most_common(10):
                print(f"  {ftype}: {count}")
    
    # 完成
    print_section_header("注释解析完成")
    
    print("\n下一步:")
    print("  运行 step5_pdb_processing.py 进行 PDB 结构处理")
    print("  命令: python scripts/step5_pdb_processing.py")


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

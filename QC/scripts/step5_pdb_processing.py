#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step5_pdb_processing.py - PDB蛋白结构文件处理
作者: Kuroneko
日期: 2025.10.04

功能说明:
    1. 读取 PDB 格式蛋白结构文件
    2. 提取结构信息（链、残基、原子统计）
    3. 计算结构特征（二级结构、几何中心、B因子统计）
    4. 生成结构摘要报告（CSV格式）
    5. 支持多模型PDB文件

输入输出:
    输入:
        - data/example_public_structure.pdb（若存在）
        - data/*.pdb（用户提供的PDB文件）
    输出:
        - outputs/pdb_summary.csv（结构摘要）
        - outputs/pdb_details_{timestamp}.txt（详细信息）

运行示例:
    python step5_pdb_processing.py

设计决策:
    - 使用 Biopython Bio.PDB 模块解析 PDB 文件
    - 计算简单的结构统计（不涉及复杂的结构分析）
    - 备选方案：可使用 DSSP 计算二级结构（需要外部程序）
"""

import os
import sys
import csv
from pathlib import Path
from datetime import datetime

# 导入工具模块
try:
    from step7_utils import (
        setup_logger, get_user_input, confirm_action,
        print_section_header, DATA_DIR, OUTPUT_DIR, format_timestamp
    )
except ImportError:
    print("错误: 无法导入 step7_utils 模块")
    print("请确保 step7_utils.py 与本脚本在同一目录下")
    sys.exit(1)


def parse_pdb_file(pdb_file: Path, logger) -> dict:
    """
    解析 PDB 文件
    
    参数:
        pdb_file: PDB 文件路径
        logger: 日志记录器
    
    返回:
        结构信息字典
    """
    try:
        from Bio.PDB import PDBParser
        from Bio.PDB.DSSP import DSSP
    except ImportError:
        logger.error("需要安装 Biopython: pip install biopython")
        return None
    
    logger.info(f"解析 PDB 文件: {pdb_file}")
    
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(pdb_file.stem, str(pdb_file))
        
        # 基本统计
        model_count = len(structure)
        chain_count = sum(1 for _ in structure.get_chains())
        residue_count = sum(1 for _ in structure.get_residues())
        atom_count = sum(1 for _ in structure.get_atoms())
        
        # 链信息
        chains_info = []
        for chain in structure.get_chains():
            chain_residues = list(chain.get_residues())
            chains_info.append({
                'chain_id': chain.id,
                'residue_count': len(chain_residues)
            })
        
        # B 因子统计
        b_factors = [atom.get_bfactor() for atom in structure.get_atoms()]
        avg_b_factor = sum(b_factors) / len(b_factors) if b_factors else 0
        min_b_factor = min(b_factors) if b_factors else 0
        max_b_factor = max(b_factors) if b_factors else 0
        
        # 几何中心
        coords = [atom.get_coord() for atom in structure.get_atoms()]
        if coords:
            import numpy as np
            center = np.mean(coords, axis=0)
            center_str = f"({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})"
        else:
            center_str = "N/A"
        
        info = {
            'filename': pdb_file.name,
            'structure_id': structure.id,
            'model_count': model_count,
            'chain_count': chain_count,
            'residue_count': residue_count,
            'atom_count': atom_count,
            'chains': chains_info,
            'avg_b_factor': avg_b_factor,
            'min_b_factor': min_b_factor,
            'max_b_factor': max_b_factor,
            'geometric_center': center_str
        }
        
        logger.info(f"  结构: {structure.id}")
        logger.info(f"  模型数: {model_count}, 链数: {chain_count}")
        logger.info(f"  残基数: {residue_count}, 原子数: {atom_count}")
        
        return info
        
    except Exception as e:
        logger.error(f"  解析失败: {e}")
        return None


def save_pdb_summary(pdb_infos: list, output_file: Path, logger):
    """
    保存 PDB 摘要为 CSV
    
    参数:
        pdb_infos: PDB 信息列表
        output_file: 输出文件
        logger: 日志记录器
    """
    if not pdb_infos:
        logger.warning("没有 PDB 信息可保存")
        return
    
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        fieldnames = ['filename', 'structure_id', 'model_count', 'chain_count',
                      'residue_count', 'atom_count', 'avg_b_factor', 
                      'min_b_factor', 'max_b_factor', 'geometric_center']
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(pdb_infos)
    
    logger.info(f"PDB 摘要已保存: {output_file}")


def save_pdb_details(pdb_infos: list, output_file: Path):
    """
    保存 PDB 详细信息
    
    参数:
        pdb_infos: PDB 信息列表
        output_file: 输出文件
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("PDB 结构详细信息\n")
        f.write("作者: Kuroneko\n")
        f.write("日期: 2025.10.04\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*70 + "\n\n")
        
        for info in pdb_infos:
            f.write(f"文件: {info['filename']}\n")
            f.write("-"*70 + "\n")
            f.write(f"结构ID: {info['structure_id']}\n")
            f.write(f"模型数: {info['model_count']}\n")
            f.write(f"链数: {info['chain_count']}\n")
            f.write(f"残基数: {info['residue_count']}\n")
            f.write(f"原子数: {info['atom_count']}\n")
            f.write(f"B因子: 平均={info['avg_b_factor']:.2f}, "
                   f"最小={info['min_b_factor']:.2f}, "
                   f"最大={info['max_b_factor']:.2f}\n")
            f.write(f"几何中心: {info['geometric_center']}\n\n")
            
            f.write("链信息:\n")
            for chain in info.get('chains', []):
                f.write(f"  链 {chain['chain_id']}: {chain['residue_count']} 个残基\n")
            
            f.write("\n" + "="*70 + "\n\n")


def main():
    """
    主函数
    """
    print_section_header("步骤 5: PDB 蛋白结构处理")
    print("作者: Kuroneko")
    print("日期: 2025.10.04\n")
    
    # 设置日志
    logger = setup_logger('step5_pdb_processing')
    
    # 查找 PDB 文件
    print_section_header("查找 PDB 文件")
    
    pdb_files = list(DATA_DIR.glob("*.pdb"))
    
    print(f"找到 {len(pdb_files)} 个 PDB 文件")
    
    if len(pdb_files) == 0:
        logger.warning("未找到 PDB 文件")
        print("\n注意: 未在 data/ 目录找到 PDB 文件")
        print("请将 PDB 结构文件放置在 data/ 目录下")
        print("  - PDB 文件: *.pdb")
        print("\n提示: 可以运行 step1_setup_and_download.py 下载示例 PDB 文件")
        sys.exit(0)
    
    # 解析 PDB 文件
    print_section_header("解析 PDB 文件")
    
    pdb_infos = []
    for pdb_file in pdb_files:
        print(f"\n处理: {pdb_file.name}")
        info = parse_pdb_file(pdb_file, logger)
        if info:
            pdb_infos.append(info)
    
    if not pdb_infos:
        logger.error("所有 PDB 文件解析失败")
        print("\n错误: 无法解析任何 PDB 文件")
        sys.exit(1)
    
    # 保存结果
    print_section_header("保存结果")
    
    csv_file = OUTPUT_DIR / "pdb_summary.csv"
    save_pdb_summary(pdb_infos, csv_file, logger)
    print(f"结构摘要已保存: {csv_file}")
    
    details_file = OUTPUT_DIR / f"pdb_details_{format_timestamp()}.txt"
    save_pdb_details(pdb_infos, details_file)
    logger.info(f"详细信息已保存: {details_file}")
    print(f"详细信息已保存: {details_file}")
    
    # 统计
    print(f"\nPDB 处理完成，共 {len(pdb_infos)} 个结构")
    total_chains = sum(info['chain_count'] for info in pdb_infos)
    total_residues = sum(info['residue_count'] for info in pdb_infos)
    total_atoms = sum(info['atom_count'] for info in pdb_infos)
    
    print(f"  总链数: {total_chains}")
    print(f"  总残基数: {total_residues}")
    print(f"  总原子数: {total_atoms}")
    
    # 完成
    print_section_header("PDB 处理完成")
    
    print("\n下一步:")
    print("  运行 step6_pipeline_driver.py 执行完整流程")
    print("  命令: python scripts/step6_pipeline_driver.py")


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


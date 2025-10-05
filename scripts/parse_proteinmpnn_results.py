#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
解析ProteinMPNN设计结果
从outputs目录的FASTA文件中提取序列信息，更新proteinmpnn_sequences.json

作者: Kuroneko
日期: 2025.10.01
"""

import os
import sys
import json
import re
from pathlib import Path
from typing import Dict, List, Any

def parse_fasta_header(header: str) -> Dict[str, Any]:
    """解析FASTA头部信息"""
    # 示例: >design:0 n:0|mpnn:1.295|plddt:0.833|ptm:0.642|pae:6.301|rmsd:2.302
    
    # 提取设计信息
    design_match = re.search(r'design:(\d+) n:(\d+)', header)
    design_id = int(design_match.group(1)) if design_match else 0
    sequence_num = int(design_match.group(2)) if design_match else 0
    
    # 提取评分信息
    mpnn_match = re.search(r'mpnn:([\d\.]+)', header)
    plddt_match = re.search(r'plddt:([\d\.]+)', header)
    ptm_match = re.search(r'ptm:([\d\.]+)', header)
    pae_match = re.search(r'pae:([\d\.]+)', header)
    rmsd_match = re.search(r'rmsd:([\d\.]+)', header)
    
    return {
        "design_id": design_id,
        "sequence_num": sequence_num,
        "mpnn_score": float(mpnn_match.group(1)) if mpnn_match else 0.0,
        "plddt": float(plddt_match.group(1)) if plddt_match else 0.0,
        "ptm": float(ptm_match.group(1)) if ptm_match else 0.0,
        "pae": float(pae_match.group(1)) if pae_match else 0.0,
        "rmsd": float(rmsd_match.group(1)) if rmsd_match else 0.0
    }

def parse_design_fasta(fasta_file: Path) -> List[Dict[str, Any]]:
    """解析设计FASTA文件"""
    sequences = []
    
    try:
        with open(fasta_file, 'r', encoding='utf-8') as f:
            lines = f.read().strip().split('\n')
        
        i = 0
        while i < len(lines):
            if lines[i].startswith('>'):
                header = lines[i][1:]  # 去掉'>'
                sequence = lines[i + 1] if i + 1 < len(lines) else ""
                
                # 解析头部信息
                header_info = parse_fasta_header(header)
                
                # 创建序列信息
                seq_info = {
                    "sequence_id": f"backbone_{fasta_file.parent.name}_proteinmpnn_{header_info['sequence_num']+1:03d}",
                    "sequence": sequence,
                    "length": len(sequence),
                    "method": "proteinmpnn",
                    "design_id": header_info["design_id"],
                    "sequence_num": header_info["sequence_num"],
                    "mpnn_score": header_info["mpnn_score"],
                    "plddt": header_info["plddt"],
                    "ptm": header_info["ptm"],
                    "pae": header_info["pae"],
                    "rmsd": header_info["rmsd"]
                }
                
                sequences.append(seq_info)
                i += 2
            else:
                i += 1
                
    except Exception as e:
        print(f"解析FASTA文件失败 {fasta_file}: {e}")
    
    return sequences

def update_proteinmpnn_sequences():
    """更新ProteinMPNN序列文件"""
    print("=" * 60)
    print("解析ProteinMPNN设计结果")
    print("=" * 60)
    
    # 路径设置
    outputs_dir = Path("designs/proteinmpnn_design_results/outputs")
    sequences_file = Path("designs/iter1/proteinmpnn_sequences.json")
    
    if not outputs_dir.exists():
        print(f"错误: 未找到outputs目录: {outputs_dir}")
        return False
    
    # 加载现有的序列文件
    sequences_data = {}
    if sequences_file.exists():
        with open(sequences_file, 'r', encoding='utf-8') as f:
            sequences_data = json.load(f)
        print(f"✓ 已加载现有序列文件: {sequences_file}")
    else:
        # 检查是否有从proteinmpnn_design_results生成的序列文件
        old_file = Path("designs/proteinmpnn_design_results/proteinmpnn_sequences.json")
        if old_file.exists():
            with open(old_file, 'r', encoding='utf-8') as f:
                sequences_data = json.load(f)
            print(f"✓ 从 {old_file} 加载序列数据")
        else:
            print(f"未找到现有序列文件，将创建新文件")
    
    # 处理每个backbone目录
    processed_count = 0
    total_sequences = 0
    
    for backbone_dir in sorted(outputs_dir.glob("backbone_*")):
        if not backbone_dir.is_dir():
            continue
        
        backbone_id = backbone_dir.name
        fasta_file = backbone_dir / "design.fasta"
        
        if not fasta_file.exists():
            print(f"警告: 未找到FASTA文件: {fasta_file}")
            continue
        
        print(f"处理: {backbone_id}")
        
        # 解析FASTA文件
        sequences = parse_design_fasta(fasta_file)
        
        if sequences:
            # 更新序列数据
            if backbone_id in sequences_data:
                sequences_data[backbone_id]["sequences"] = sequences
                sequences_data[backbone_id]["sequence_count"] = len(sequences)
                processed_count += 1
                total_sequences += len(sequences)
                print(f"  ✓ 找到 {len(sequences)} 个序列")
            else:
                print(f"  警告: 未找到对应的backbone数据: {backbone_id}")
        else:
            print(f"  ✗ 未找到序列")
    
    # 保存更新后的文件
    with open(sequences_file, 'w', encoding='utf-8') as f:
        json.dump(sequences_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n" + "="*60)
    print("解析完成!")
    print("="*60)
    print(f"处理backbone数量: {processed_count}")
    print(f"总序列数量: {total_sequences}")
    print(f"更新文件: {sequences_file}")
    
    return True

def main():
    """主函数"""
    try:
        success = update_proteinmpnn_sequences()
        return success
    except Exception as e:
        print(f"错误: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
随机采样ProteinMPNN序列
从proteinmpnn_sequences.json中随机选择80个序列进行AlphaFold2预测

作者: Kuroneko
日期: 2025.10.01
"""

import os
import sys
import json
import random
from pathlib import Path
from typing import Dict, List, Any

def get_project_root():
    """获取项目根目录"""
    return Path(__file__).parent

def load_proteinmpnn_sequences() -> Dict[str, Any]:
    """加载ProteinMPNN序列数据"""
    # 尝试多个可能的路径
    possible_paths = [
        get_project_root() / "designs" / "proteinmpnn_sequences.json",
        get_project_root() / "designs" / "iter1" / "proteinmpnn_sequences.json",
        get_project_root() / "designs" / "iter1" / "proteinmpnn_sequences.json"
    ]
    
    sequences_file = None
    for path in possible_paths:
        if path.exists():
            sequences_file = path
            break
    
    if not sequences_file:
        print(f"错误: 未找到proteinmpnn_sequences.json文件")
        print("尝试的路径:")
        for path in possible_paths:
            print(f"  - {path}")
        return {}
    
    try:
        with open(sequences_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✓ 成功加载ProteinMPNN序列文件: {sequences_file}")
        return data
    except Exception as e:
        print(f"错误: 加载文件失败 {e}")
        return {}

def extract_all_sequences(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """提取所有序列"""
    all_sequences = []
    
    for backbone_id, backbone_data in data.items():
        if "sequences" in backbone_data:
            for seq in backbone_data["sequences"]:
                seq["backbone_id"] = backbone_id
                all_sequences.append(seq)
    
    return all_sequences

def random_sample_sequences(sequences: List[Dict[str, Any]], sample_size: int = 80) -> List[Dict[str, Any]]:
    """随机采样序列"""
    if len(sequences) <= sample_size:
        print(f"警告: 总序列数({len(sequences)})小于采样数({sample_size})，返回所有序列")
        return sequences
    
    # 设置随机种子以确保可重复性
    random.seed(42)
    
    # 随机采样
    sampled_sequences = random.sample(sequences, sample_size)
    
    print(f"✓ 随机采样完成: {len(sampled_sequences)} 个序列")
    return sampled_sequences

def create_sampled_json(sampled_sequences: List[Dict[str, Any]]) -> Dict[str, Any]:
    """创建采样后的JSON结构"""
    sampled_data = {}
    
    for seq in sampled_sequences:
        backbone_id = seq["backbone_id"]
        
        if backbone_id not in sampled_data:
            sampled_data[backbone_id] = {
                "backbone_file": f"{backbone_id}.pdb",
                "backbone_id": backbone_id,
                "design_method": "proteinmpnn",
                "parameters": {
                    "num_sequences": 1,
                    "sampling_temp": 0.1,
                    "remove_aa": "C",
                    "num_recycles": 1
                },
                "timestamp": "2025-10-01 18:30:00",
                "sequences": []
            }
        
        # 移除backbone_id字段，避免重复
        seq_copy = seq.copy()
        seq_copy.pop("backbone_id", None)
        
        sampled_data[backbone_id]["sequences"].append(seq_copy)
    
    return sampled_data

def save_sampled_json(sampled_data: Dict[str, Any], output_file: Path):
    """保存采样后的JSON文件"""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(sampled_data, f, indent=2, ensure_ascii=False)
        print(f"✓ 采样结果已保存: {output_file}")
        return True
    except Exception as e:
        print(f"错误: 保存文件失败 {e}")
        return False

def print_statistics(original_data: Dict[str, Any], sampled_data: Dict[str, Any]):
    """打印统计信息"""
    original_sequences = sum(len(backbone_data.get("sequences", [])) 
                           for backbone_data in original_data.values())
    sampled_sequences = sum(len(backbone_data.get("sequences", [])) 
                           for backbone_data in sampled_data.values())
    
    original_backbones = len(original_data)
    sampled_backbones = len(sampled_data)
    
    print("\n" + "="*60)
    print("采样统计信息")
    print("="*60)
    print(f"原始数据:")
    print(f"  - 骨架数量: {original_backbones}")
    print(f"  - 序列数量: {original_sequences}")
    print(f"采样数据:")
    print(f"  - 骨架数量: {sampled_backbones}")
    print(f"  - 序列数量: {sampled_sequences}")
    print(f"采样比例: {sampled_sequences/original_sequences*100:.1f}%")
    print("="*60)

def main():
    """主函数"""
    print("=" * 60)
    print("ProteinMPNN序列随机采样")
    print("作者: Kuroneko | 日期: 2025.10.01")
    print("=" * 60)
    
    # 获取采样数量
    sample_size = input("请输入采样数量 (默认: 80): ").strip()
    if not sample_size:
        sample_size = 80
    else:
        sample_size = int(sample_size)
    
    print(f"开始随机采样 {sample_size} 个ProteinMPNN序列...")
    
    # 加载原始数据
    original_data = load_proteinmpnn_sequences()
    if not original_data:
        return False
    
    # 提取所有序列
    all_sequences = extract_all_sequences(original_data)
    print(f"✓ 提取到 {len(all_sequences)} 个序列")
    
    # 随机采样
    sampled_sequences = random_sample_sequences(all_sequences, sample_size)
    
    # 创建采样后的JSON结构
    sampled_data = create_sampled_json(sampled_sequences)
    
    # 保存采样结果
    output_file = get_project_root() / "proteinmpnn_sampled_80.json"
    if not save_sampled_json(sampled_data, output_file):
        return False
    
    # 打印统计信息
    print_statistics(original_data, sampled_data)
    
    print(f"\n✓ 采样完成!")
    print(f"输出文件: {output_file}")
    print(f"可用于AlphaFold2预测的序列数量: {len(sampled_sequences)}")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
序列提取脚本 - 从第四步结果提取序列用于第五步
作者: Kuroneko
日期: 2025.6.26
描述: 从第四步ESMFold预测结果中提取序列，转换为第五步ColabFold可以识别的JSON格式
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_merged_sequences(merged_file: str) -> List[Dict[str, Any]]:
    """从merged_input_sequences.json加载所有序列"""
    logger.info(f"加载合并序列文件: {merged_file}")
    
    try:
        with open(merged_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        sequences = data.get('sequences', [])
        logger.info(f"  找到 {len(sequences)} 个序列")
        return sequences
        
    except Exception as e:
        logger.error(f"加载失败: {e}")
        return []

def load_metrics_from_folders(prediction_dir: str) -> Dict[str, Dict[str, Any]]:
    """从预测结果文件夹加载指标数据"""
    logger.info(f"扫描预测结果目录: {prediction_dir}")
    
    metrics_data = {}
    prediction_path = Path(prediction_dir)
    
    if not prediction_path.exists():
        logger.warning(f"预测结果目录不存在: {prediction_dir}")
        return metrics_data
    
    # 扫描所有序列文件夹
    for seq_dir in prediction_path.iterdir():
        if seq_dir.is_dir() and seq_dir.name.startswith('protein_backbone_'):
            metrics_file = seq_dir / 'metrics.json'
            if metrics_file.exists():
                try:
                    with open(metrics_file, 'r', encoding='utf-8') as f:
                        metrics = json.load(f)
                    metrics_data[seq_dir.name] = metrics
                    logger.debug(f"  加载指标: {seq_dir.name}")
                except Exception as e:
                    logger.warning(f"  加载指标失败 {seq_dir.name}: {e}")
    
    logger.info(f"  成功加载 {len(metrics_data)} 个序列的指标")
    return metrics_data

def filter_sequences_by_quality(sequences: List[Dict[str, Any]], 
                               metrics_data: Dict[str, Dict[str, Any]],
                               min_plddt: float = 50.0,
                               min_ptm: float = 0.5) -> List[Dict[str, Any]]:
    """根据质量指标筛选序列"""
    logger.info(f"开始质量筛选 (pLDDT >= {min_plddt}, PTM >= {min_ptm})")
    
    filtered_sequences = []
    
    for seq in sequences:
        seq_id = seq['id']
        
        # 检查是否有对应的指标数据
        if seq_id not in metrics_data:
            logger.warning(f"  序列 {seq_id} 没有找到指标数据，跳过")
            continue
        
        metrics = metrics_data[seq_id]
        plddt_mean = metrics.get('plddt_mean', 0)
        ptm = metrics.get('ptm', 0)
        
        # 应用筛选条件
        if plddt_mean >= min_plddt and ptm >= min_ptm:
            # 添加质量指标到序列数据
            seq_with_metrics = seq.copy()
            seq_with_metrics.update({
                'plddt_mean': plddt_mean,
                'ptm': ptm,
                'quality_score': (plddt_mean + ptm * 100) / 2,  # 综合评分
                'prediction_status': 'completed'
            })
            filtered_sequences.append(seq_with_metrics)
            logger.info(f"  ✓ 通过筛选: {seq_id} (pLDDT: {plddt_mean:.1f}, PTM: {ptm:.3f})")
        else:
            logger.info(f"  ✗ 未通过筛选: {seq_id} (pLDDT: {plddt_mean:.1f}, PTM: {ptm:.3f})")
    
    logger.info(f"筛选完成: {len(filtered_sequences)}/{len(sequences)} 个序列通过")
    return filtered_sequences

def create_step5_input_format(sequences: List[Dict[str, Any]], 
                             output_file: str,
                             method: str = 'esm_if') -> None:
    """创建第五步可以识别的输入格式"""
    logger.info(f"创建第五步输入文件: {output_file}")
    
    # 转换为第五步期望的格式
    step5_data = {
        "design_method": method,
        "timestamp": "2025-10-03",
        "source": "step4_esmfold_results",
        "total_sequences": len(sequences),
        "results": {}
    }
    
    # 按backbone_id分组
    backbone_groups = {}
    for seq in sequences:
        backbone_id = seq.get('backbone_id', 'unknown')
        if backbone_id not in backbone_groups:
            backbone_groups[backbone_id] = {
                "backbone_id": backbone_id,
                "sequences": []
            }
        
        # 转换为第五步期望的序列格式
        seq_data = {
            "sequence_id": seq['id'],
            "sequence": seq['sequence'],
            "length": seq['length'],
            "method": seq.get('method', method),
            "plddt_mean": seq.get('plddt_mean', 0),
            "ptm": seq.get('ptm', 0),
            "quality_score": seq.get('quality_score', 0),
            "prediction_status": seq.get('prediction_status', 'unknown')
        }
        
        backbone_groups[backbone_id]["sequences"].append(seq_data)
    
    step5_data["results"] = backbone_groups
    
    # 保存文件
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(step5_data, f, indent=2, ensure_ascii=False)
        logger.info(f"  成功保存 {len(sequences)} 个序列到 {output_file}")
    except Exception as e:
        logger.error(f"  保存失败: {e}")

def create_simple_list_format(sequences: List[Dict[str, Any]], 
                             output_file: str) -> None:
    """创建简单的序列列表格式（备用）"""
    logger.info(f"创建简单列表格式: {output_file}")
    
    simple_sequences = []
    for seq in sequences:
        simple_sequences.append({
            "id": seq['id'],
            "sequence": seq['sequence'],
            "length": seq['length'],
            "method": seq.get('method', 'esm_if'),
            "plddt_mean": seq.get('plddt_mean', 0),
            "ptm": seq.get('ptm', 0),
            "quality_score": seq.get('quality_score', 0)
        })
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(simple_sequences, f, indent=2, ensure_ascii=False)
        logger.info(f"  成功保存 {len(simple_sequences)} 个序列到 {output_file}")
    except Exception as e:
        logger.error(f"  保存失败: {e}")

def main():
    """主函数"""
    print("=" * 60)
    print("序列提取脚本 - 从第四步结果提取序列用于第五步")
    print("作者: Kuroneko | 日期: 2025.10.3")
    print("=" * 60)
    
    # 配置路径
    base_dir = Path("F:/Project/蛋白质设计")
    prediction_dir = base_dir / "output" / "prediction_results"
    merged_file = prediction_dir / "merged_input_sequences.json"
    
    # 输出文件
    output_dir = base_dir / "output"
    output_dir.mkdir(exist_ok=True)
    
    step5_format_file = output_dir / "step5_input_sequences.json"
    simple_format_file = output_dir / "step5_simple_sequences.json"
    
    print(f"\n[配置] 输入目录: {prediction_dir}")
    print(f"[配置] 输出目录: {output_dir}")
    
    # 1. 加载合并序列
    print(f"\n[步骤1] 加载合并序列...")
    sequences = load_merged_sequences(str(merged_file))
    if not sequences:
        print("❌ 没有找到序列数据，请检查文件路径")
        return
    
    # 2. 加载指标数据
    print(f"\n[步骤2] 加载预测指标...")
    metrics_data = load_metrics_from_folders(str(prediction_dir))
    
    # 3. 质量筛选
    print(f"\n[步骤3] 质量筛选...")
    print("请选择筛选策略:")
    print("1. 严格筛选 (pLDDT >= 70, PTM >= 0.7)")
    print("2. 中等筛选 (pLDDT >= 50, PTM >= 0.5)")
    print("3. 宽松筛选 (pLDDT >= 30, PTM >= 0.3)")
    print("4. 不筛选 (使用所有序列)")
    
    choice = input("请选择 (1-4): ").strip()
    
    if choice == "1":
        min_plddt, min_ptm = 70.0, 0.7
    elif choice == "2":
        min_plddt, min_ptm = 50.0, 0.5
    elif choice == "3":
        min_plddt, min_ptm = 30.0, 0.3
    else:
        min_plddt, min_ptm = 0.0, 0.0
    
    if min_plddt > 0 or min_ptm > 0:
        filtered_sequences = filter_sequences_by_quality(sequences, metrics_data, min_plddt, min_ptm)
    else:
        # 不筛选，但添加可用的指标数据
        filtered_sequences = []
        for seq in sequences:
            seq_with_metrics = seq.copy()
            if seq['id'] in metrics_data:
                metrics = metrics_data[seq['id']]
                seq_with_metrics.update({
                    'plddt_mean': metrics.get('plddt_mean', 0),
                    'ptm': metrics.get('ptm', 0),
                    'quality_score': (metrics.get('plddt_mean', 0) + metrics.get('ptm', 0) * 100) / 2,
                    'prediction_status': 'completed'
                })
            else:
                seq_with_metrics.update({
                    'plddt_mean': 0,
                    'ptm': 0,
                    'quality_score': 0,
                    'prediction_status': 'no_metrics'
                })
            filtered_sequences.append(seq_with_metrics)
    
    if not filtered_sequences:
        print("❌ 没有序列通过筛选，请降低筛选标准")
        return
    
    # 4. 创建输出文件
    print(f"\n[步骤4] 创建输出文件...")
    create_step5_input_format(filtered_sequences, str(step5_format_file))
    create_simple_list_format(filtered_sequences, str(simple_format_file))
    
    # 5. 统计信息
    print(f"\n[完成] 序列提取完成!")
    print(f"  总序列数: {len(sequences)}")
    print(f"  通过筛选: {len(filtered_sequences)}")
    print(f"  输出文件:")
    print(f"    - 第五步格式: {step5_format_file}")
    print(f"    - 简单格式: {simple_format_file}")
    
    # 显示质量统计
    if filtered_sequences:
        plddt_scores = [seq.get('plddt_mean', 0) for seq in filtered_sequences]
        ptm_scores = [seq.get('ptm', 0) for seq in filtered_sequences]
        
        print(f"\n[质量统计]")
        print(f"  pLDDT 平均: {sum(plddt_scores)/len(plddt_scores):.1f}")
        print(f"  pLDDT 范围: {min(plddt_scores):.1f} - {max(plddt_scores):.1f}")
        print(f"  PTM 平均: {sum(ptm_scores)/len(ptm_scores):.3f}")
        print(f"  PTM 范围: {min(ptm_scores):.3f} - {max(ptm_scores):.3f}")
    
    print(f"\n[下一步] 现在可以使用以下文件作为第五步的输入:")
    print(f"  - {step5_format_file} (推荐，完整格式)")
    print(f"  - {simple_format_file} (备用，简单格式)")

if __name__ == "__main__":
    main()

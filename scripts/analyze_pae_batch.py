#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PAE 批量分析脚本 - 分析蛋白质结构预测的 PAE 矩阵
作者: Kuroneko
日期: 2025.10.3
描述: 批量分析预测结果文件夹中的 PAE 文件，生成质量评估和功能分析报告
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体（避免中文显示问题）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PAEAnalyzer:
    """PAE 分析器类"""
    
    def __init__(self, output_dir: str = "pae_analysis_results"):
        """
        初始化 PAE 分析器
        
        Args:
            output_dir: 输出目录路径
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 创建子目录
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        self.reports_dir = self.output_dir / "reports"
        self.reports_dir.mkdir(exist_ok=True)
        
    def load_pae_file(self, pae_file: str) -> np.ndarray:
        """
        加载 PAE 文件
        
        Args:
            pae_file: PAE 文件路径
            
        Returns:
            PAE 矩阵（numpy数组）
        """
        try:
            # 尝试加载为 numpy 数组（.npy 格式）
            if pae_file.endswith('.npy'):
                return np.load(pae_file)
            
            # 尝试加载为文本文件
            elif pae_file.endswith('.txt'):
                with open(pae_file, 'r') as f:
                    lines = f.readlines()
                    pae_matrix = []
                    for line in lines:
                        # 去除空行和注释
                        line = line.strip()
                        if line and not line.startswith('#'):
                            # 尝试解析数字
                            values = [float(x) for x in line.split()]
                            if values:
                                pae_matrix.append(values)
                    
                    if pae_matrix:
                        return np.array(pae_matrix)
                    else:
                        logger.warning(f"PAE 文件为空: {pae_file}")
                        return None
            
            # 尝试加载为 JSON 格式
            elif pae_file.endswith('.json'):
                with open(pae_file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return np.array(data)
                    elif 'pae' in data:
                        return np.array(data['pae'])
                    else:
                        logger.warning(f"无法解析 JSON PAE 文件: {pae_file}")
                        return None
            
            else:
                logger.warning(f"不支持的文件格式: {pae_file}")
                return None
                
        except Exception as e:
            logger.error(f"加载 PAE 文件失败: {pae_file}, 错误: {e}")
            return None
    
    def calculate_quality_metrics(self, pae_matrix: np.ndarray) -> Dict[str, float]:
        """
        计算质量指标
        
        Args:
            pae_matrix: PAE 矩阵
            
        Returns:
            质量指标字典
        """
        n = len(pae_matrix)
        
        # 整体平均 PAE
        avg_pae = np.mean(pae_matrix)
        
        # 对角线附近的 PAE（局部结构质量）
        local_pae = []
        for i in range(n):
            for j in range(max(0, i-5), min(n, i+6)):
                if i != j:
                    local_pae.append(pae_matrix[i][j])
        avg_local_pae = np.mean(local_pae) if local_pae else 0
        
        # 不同距离范围的 PAE
        short_range = []  # 1-5 残基
        medium_range = []  # 6-15 残基
        long_range = []  # 16+ 残基
        
        for i in range(n):
            for j in range(i+1, n):
                distance = abs(i - j)
                pae_value = pae_matrix[i][j]
                
                if distance <= 5:
                    short_range.append(pae_value)
                elif distance <= 15:
                    medium_range.append(pae_value)
                else:
                    long_range.append(pae_value)
        
        # 置信度水平统计
        total_pairs = n * n
        high_conf = np.sum(pae_matrix < 5) / total_pairs * 100  # < 5 Å
        medium_conf = np.sum((pae_matrix >= 5) & (pae_matrix < 10)) / total_pairs * 100  # 5-10 Å
        low_conf = np.sum((pae_matrix >= 10) & (pae_matrix < 15)) / total_pairs * 100  # 10-15 Å
        very_low_conf = np.sum(pae_matrix >= 15) / total_pairs * 100  # >= 15 Å
        
        return {
            'avg_pae': avg_pae,
            'avg_local_pae': avg_local_pae,
            'short_range_pae': np.mean(short_range) if short_range else 0,
            'medium_range_pae': np.mean(medium_range) if medium_range else 0,
            'long_range_pae': np.mean(long_range) if long_range else 0,
            'high_confidence_%': high_conf,
            'medium_confidence_%': medium_conf,
            'low_confidence_%': low_conf,
            'very_low_confidence_%': very_low_conf,
            'min_pae': np.min(pae_matrix),
            'max_pae': np.max(pae_matrix),
            'std_pae': np.std(pae_matrix)
        }
    
    def assess_quality(self, metrics: Dict[str, float]) -> Tuple[str, str]:
        """
        评估结构质量
        
        Args:
            metrics: 质量指标
            
        Returns:
            (质量等级, 质量说明)
        """
        avg_pae = metrics['avg_pae']
        high_conf = metrics['high_confidence_%']
        
        if avg_pae < 5 and high_conf > 50:
            quality = "优秀"
            description = "结构预测质量非常高，大部分残基间距离预测准确，可以直接用于后续分析"
        elif avg_pae < 10 and high_conf > 30:
            quality = "良好"
            description = "结构预测质量较好，主要区域的距离预测可靠，适合用于功能分析"
        elif avg_pae < 15:
            quality = "中等"
            description = "结构预测质量一般，部分区域可能不可靠，需要结合其他证据进行分析"
        elif avg_pae < 20:
            quality = "较差"
            description = "结构预测质量较差，多数区域预测不可靠，建议谨慎使用或重新预测"
        else:
            quality = "差"
            description = "结构预测质量很差，大部分区域不可靠，不建议使用该预测结果"
        
        return quality, description
    
    def identify_domains(self, pae_matrix: np.ndarray, threshold: float = 12) -> List[Dict]:
        """
        识别结构域
        
        Args:
            pae_matrix: PAE 矩阵
            threshold: PAE 阈值（Å）
            
        Returns:
            结构域列表
        """
        n = len(pae_matrix)
        domains = []
        
        # 简化的域识别：寻找对角块状的低 PAE 区域
        window_size = 20  # 窗口大小
        
        for i in range(0, n - window_size, 10):  # 步长为10
            for j in range(i, n - window_size, 10):
                # 计算窗口内的平均 PAE
                window = pae_matrix[i:i+window_size, j:j+window_size]
                avg_pae = np.mean(window)
                
                # 如果是低 PAE 区域（对角块）
                if avg_pae < threshold and abs(i - j) < 30:
                    domains.append({
                        'start': i + 1,
                        'end': i + window_size,
                        'avg_pae': avg_pae,
                        'type': 'structured_domain' if avg_pae < 8 else 'possible_domain'
                    })
        
        # 合并重叠的域
        if domains:
            merged_domains = [domains[0]]
            for domain in domains[1:]:
                last_domain = merged_domains[-1]
                if domain['start'] <= last_domain['end'] + 10:
                    # 合并
                    last_domain['end'] = max(last_domain['end'], domain['end'])
                    last_domain['avg_pae'] = min(last_domain['avg_pae'], domain['avg_pae'])
                else:
                    merged_domains.append(domain)
            
            return merged_domains
        
        return domains
    
    def identify_functional_sites(self, pae_matrix: np.ndarray) -> List[Dict]:
        """
        识别功能位点
        
        Args:
            pae_matrix: PAE 矩阵
            
        Returns:
            功能位点列表
        """
        n = len(pae_matrix)
        functional_sites = []
        
        for i in range(n):
            # 计算残基 i 与其他残基的平均 PAE
            avg_pae_to_others = np.mean([pae_matrix[i][j] for j in range(n) if i != j])
            
            # 计算局部置信度（周围±10个残基）
            local_range = range(max(0, i-10), min(n, i+11))
            local_pae = np.mean([pae_matrix[i][j] for j in local_range if i != j])
            
            # 如果局部 PAE 很低，可能是重要的结构化区域
            if local_pae < 8:
                functional_sites.append({
                    'residue': i + 1,
                    'avg_pae': avg_pae_to_others,
                    'local_pae': local_pae,
                    'importance': 'high' if local_pae < 5 else 'medium',
                    'type': 'structured_core' if avg_pae_to_others < 10 else 'local_structure'
                })
        
        # 按局部 PAE 排序
        functional_sites.sort(key=lambda x: x['local_pae'])
        
        return functional_sites[:50]  # 返回前50个最重要的位点
    
    def analyze_interactions(self, pae_matrix: np.ndarray, threshold: float = 10) -> List[Dict]:
        """
        分析残基间相互作用
        
        Args:
            pae_matrix: PAE 矩阵
            threshold: PAE 阈值（Å）
            
        Returns:
            相互作用列表
        """
        n = len(pae_matrix)
        interactions = []
        
        for i in range(n):
            for j in range(i+5, n):  # 跳过近邻残基（至少相距5个残基）
                pae_value = pae_matrix[i][j]
                
                if pae_value < threshold:
                    interactions.append({
                        'residue_i': i + 1,
                        'residue_j': j + 1,
                        'distance': abs(i - j),
                        'pae': pae_value,
                        'confidence': 'high' if pae_value < 5 else 'medium'
                    })
        
        # 按 PAE 排序
        interactions.sort(key=lambda x: x['pae'])
        
        return interactions[:100]  # 返回前100个最强的相互作用
    
    def plot_pae_matrix(self, pae_matrix: np.ndarray, title: str, save_path: str):
        """
        绘制 PAE 矩阵热图
        
        Args:
            pae_matrix: PAE 矩阵
            title: 图表标题
            save_path: 保存路径
        """
        plt.figure(figsize=(10, 8))
        
        # 使用 imshow 绘制热图
        im = plt.imshow(pae_matrix, cmap='viridis_r', origin='lower', 
                       vmin=0, vmax=min(30, np.max(pae_matrix)))
        
        plt.colorbar(im, label='PAE (Angstrom)')
        plt.xlabel('Residue Index')
        plt.ylabel('Residue Index')
        plt.title(title)
        
        # 添加网格
        plt.grid(False)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  保存 PAE 热图: {save_path}")
    
    def plot_pae_distribution(self, pae_matrix: np.ndarray, title: str, save_path: str):
        """
        绘制 PAE 值分布图
        
        Args:
            pae_matrix: PAE 矩阵
            title: 图表标题
            save_path: 保存路径
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # 直方图
        pae_values = pae_matrix.flatten()
        ax1.hist(pae_values, bins=50, edgecolor='black', alpha=0.7)
        ax1.axvline(x=5, color='green', linestyle='--', label='High Confidence (5A)')
        ax1.axvline(x=10, color='orange', linestyle='--', label='Medium Confidence (10A)')
        ax1.axvline(x=15, color='red', linestyle='--', label='Low Confidence (15A)')
        ax1.set_xlabel('PAE (Angstrom)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('PAE Distribution')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # 箱线图（按距离分组）
        n = len(pae_matrix)
        distance_groups = {
            'Short (1-5)': [],
            'Medium (6-15)': [],
            'Long (16+)': []
        }
        
        for i in range(n):
            for j in range(i+1, n):
                distance = abs(i - j)
                pae_value = pae_matrix[i][j]
                
                if distance <= 5:
                    distance_groups['Short (1-5)'].append(pae_value)
                elif distance <= 15:
                    distance_groups['Medium (6-15)'].append(pae_value)
                else:
                    distance_groups['Long (16+)'].append(pae_value)
        
        ax2.boxplot(distance_groups.values(), labels=distance_groups.keys())
        ax2.set_ylabel('PAE (Angstrom)')
        ax2.set_xlabel('Residue Distance')
        ax2.set_title('PAE by Residue Distance')
        ax2.grid(alpha=0.3)
        
        plt.suptitle(title, fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  保存 PAE 分布图: {save_path}")
    
    def plot_per_residue_confidence(self, pae_matrix: np.ndarray, title: str, save_path: str):
        """
        绘制每个残基的置信度曲线
        
        Args:
            pae_matrix: PAE 矩阵
            title: 图表标题
            save_path: 保存路径
        """
        n = len(pae_matrix)
        
        # 计算每个残基的平均 PAE
        avg_pae_per_residue = [np.mean(pae_matrix[i]) for i in range(n)]
        
        plt.figure(figsize=(12, 5))
        plt.plot(range(1, n+1), avg_pae_per_residue, linewidth=1.5)
        plt.axhline(y=5, color='green', linestyle='--', alpha=0.5, label='High Confidence')
        plt.axhline(y=10, color='orange', linestyle='--', alpha=0.5, label='Medium Confidence')
        plt.axhline(y=15, color='red', linestyle='--', alpha=0.5, label='Low Confidence')
        
        plt.xlabel('Residue Index')
        plt.ylabel('Average PAE (Angstrom)')
        plt.title(title)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  保存残基置信度曲线: {save_path}")
    
    def analyze_sequence(self, seq_dir: Path) -> Dict[str, Any]:
        """
        分析单个序列的 PAE 文件
        
        Args:
            seq_dir: 序列目录路径
            
        Returns:
            分析结果字典
        """
        seq_id = seq_dir.name
        logger.info(f"分析序列: {seq_id}")
        
        # 查找 PAE 文件
        pae_files = list(seq_dir.glob("*.pae.txt")) + list(seq_dir.glob("*.pae.json")) + list(seq_dir.glob("*.pae.npy"))
        
        if not pae_files:
            logger.warning(f"  未找到 PAE 文件: {seq_dir}")
            return None
        
        pae_file = pae_files[0]
        logger.info(f"  加载 PAE 文件: {pae_file.name}")
        
        # 加载 PAE 矩阵
        pae_matrix = self.load_pae_file(str(pae_file))
        
        if pae_matrix is None:
            logger.warning(f"  PAE 矩阵加载失败: {seq_id}")
            return None
        
        # 计算质量指标
        metrics = self.calculate_quality_metrics(pae_matrix)
        quality, quality_desc = self.assess_quality(metrics)
        
        # 识别结构域
        domains = self.identify_domains(pae_matrix)
        
        # 识别功能位点
        functional_sites = self.identify_functional_sites(pae_matrix)
        
        # 分析相互作用
        interactions = self.analyze_interactions(pae_matrix)
        
        # 绘图
        seq_plots_dir = self.plots_dir / seq_id
        seq_plots_dir.mkdir(exist_ok=True)
        
        self.plot_pae_matrix(pae_matrix, f"{seq_id} - PAE Matrix", 
                            str(seq_plots_dir / f"{seq_id}_pae_matrix.png"))
        self.plot_pae_distribution(pae_matrix, f"{seq_id} - PAE Distribution",
                                   str(seq_plots_dir / f"{seq_id}_pae_distribution.png"))
        self.plot_per_residue_confidence(pae_matrix, f"{seq_id} - Per-Residue Confidence",
                                         str(seq_plots_dir / f"{seq_id}_residue_confidence.png"))
        
        return {
            'sequence_id': seq_id,
            'sequence_length': len(pae_matrix),
            'quality': quality,
            'quality_description': quality_desc,
            'metrics': metrics,
            'domains': domains,
            'functional_sites': functional_sites,
            'interactions': interactions
        }
    
    def batch_analyze(self, input_dir: str) -> pd.DataFrame:
        """
        批量分析所有序列
        
        Args:
            input_dir: 输入目录路径
            
        Returns:
            分析结果 DataFrame
        """
        input_path = Path(input_dir)
        
        if not input_path.exists():
            logger.error(f"输入目录不存在: {input_dir}")
            return None
        
        # 查找所有子目录
        subdirs = [d for d in input_path.iterdir() if d.is_dir()]
        
        logger.info(f"找到 {len(subdirs)} 个序列目录")
        
        all_results = []
        detailed_reports = []
        
        for seq_dir in subdirs:
            result = self.analyze_sequence(seq_dir)
            
            if result:
                # 汇总结果
                summary = {
                    'sequence_id': result['sequence_id'],
                    'sequence_length': result['sequence_length'],
                    'quality': result['quality'],
                    'avg_pae': result['metrics']['avg_pae'],
                    'avg_local_pae': result['metrics']['avg_local_pae'],
                    'short_range_pae': result['metrics']['short_range_pae'],
                    'medium_range_pae': result['metrics']['medium_range_pae'],
                    'long_range_pae': result['metrics']['long_range_pae'],
                    'high_confidence_%': result['metrics']['high_confidence_%'],
                    'medium_confidence_%': result['metrics']['medium_confidence_%'],
                    'low_confidence_%': result['metrics']['low_confidence_%'],
                    'very_low_confidence_%': result['metrics']['very_low_confidence_%'],
                    'num_domains': len(result['domains']),
                    'num_functional_sites': len(result['functional_sites']),
                    'num_interactions': len(result['interactions']),
                    'quality_description': result['quality_description']
                }
                
                all_results.append(summary)
                detailed_reports.append(result)
        
        # 转换为 DataFrame
        df = pd.DataFrame(all_results)
        
        # 保存详细报告
        self.save_detailed_reports(detailed_reports)
        
        return df
    
    def save_detailed_reports(self, reports: List[Dict]):
        """
        保存详细报告
        
        Args:
            reports: 分析结果列表
        """
        for report in reports:
            seq_id = report['sequence_id']
            report_file = self.reports_dir / f"{seq_id}_detailed_report.txt"
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write(f"序列 PAE 分析详细报告: {seq_id}\n")
                f.write("=" * 80 + "\n\n")
                
                # 质量评估
                f.write(f"【质量评估】\n")
                f.write(f"  质量等级: {report['quality']}\n")
                f.write(f"  质量说明: {report['quality_description']}\n\n")
                
                # 质量指标
                f.write(f"【质量指标】\n")
                for key, value in report['metrics'].items():
                    f.write(f"  {key}: {value:.2f}\n")
                f.write("\n")
                
                # 结构域
                f.write(f"【结构域分析】(共 {len(report['domains'])} 个)\n")
                for i, domain in enumerate(report['domains'][:10], 1):
                    f.write(f"  域 {i}: 残基 {domain['start']}-{domain['end']}, "
                           f"平均PAE={domain['avg_pae']:.2f}Å, 类型={domain['type']}\n")
                f.write("\n")
                
                # 功能位点
                f.write(f"【功能位点分析】(显示前10个最重要的位点)\n")
                for i, site in enumerate(report['functional_sites'][:10], 1):
                    f.write(f"  位点 {i}: 残基 {site['residue']}, "
                           f"局部PAE={site['local_pae']:.2f}Å, "
                           f"重要性={site['importance']}, 类型={site['type']}\n")
                f.write("\n")
                
                # 相互作用
                f.write(f"【残基间相互作用】(显示前10个最强的相互作用)\n")
                for i, inter in enumerate(report['interactions'][:10], 1):
                    f.write(f"  相互作用 {i}: 残基 {inter['residue_i']} <-> {inter['residue_j']}, "
                           f"距离={inter['distance']}残基, PAE={inter['pae']:.2f}Å, "
                           f"置信度={inter['confidence']}\n")
                f.write("\n")
                
                f.write("=" * 80 + "\n")
        
        logger.info(f"保存 {len(reports)} 个详细报告到 {self.reports_dir}")


def main():
    """主函数"""
    print("=" * 80)
    print("PAE 批量分析脚本")
    print("作者: Kuroneko | 日期: 2025.10.3")
    print("=" * 80)
    print()
    
    # 获取用户输入
    input_dir = input("请输入预测结果目录路径: ").strip()
    
    if not input_dir:
        print("使用默认路径: F:/Project/蛋白质设计/output/prediction_results")
        input_dir = "F:/Project/蛋白质设计/output/prediction_results"
    
    output_dir = input("请输入输出目录路径 (默认: pae_analysis_results): ").strip()
    
    if not output_dir:
        output_dir = "pae_analysis_results"
    
    # 创建分析器
    analyzer = PAEAnalyzer(output_dir=output_dir)
    
    print()
    print(f"[开始] 批量分析 PAE 文件")
    print(f"  输入目录: {input_dir}")
    print(f"  输出目录: {output_dir}")
    print()
    
    # 批量分析
    results_df = analyzer.batch_analyze(input_dir)
    
    if results_df is None or len(results_df) == 0:
        print("❌ 没有找到可分析的数据")
        return
    
    # 保存汇总结果
    summary_csv = Path(output_dir) / "pae_analysis_summary.csv"
    results_df.to_csv(summary_csv, index=False, encoding='utf-8-sig')
    logger.info(f"保存汇总结果: {summary_csv}")
    
    # 生成统计报告
    print()
    print("=" * 80)
    print("分析完成！统计摘要:")
    print("=" * 80)
    print(f"  总序列数: {len(results_df)}")
    print(f"  质量等级分布:")
    quality_counts = results_df['quality'].value_counts()
    for quality, count in quality_counts.items():
        print(f"    {quality}: {count} ({count/len(results_df)*100:.1f}%)")
    
    print()
    print(f"  平均 PAE: {results_df['avg_pae'].mean():.2f} ± {results_df['avg_pae'].std():.2f} Å")
    print(f"  平均高置信度比例: {results_df['high_confidence_%'].mean():.1f}%")
    print(f"  平均结构域数量: {results_df['num_domains'].mean():.1f}")
    print(f"  平均功能位点数量: {results_df['num_functional_sites'].mean():.1f}")
    print()
    
    # 输出文件列表
    print("输出文件:")
    print(f"  - 汇总CSV: {summary_csv}")
    print(f"  - 详细报告: {Path(output_dir) / 'reports'}")
    print(f"  - 分析图表: {Path(output_dir) / 'plots'}")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()


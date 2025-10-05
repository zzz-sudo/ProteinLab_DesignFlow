"""
步骤2: MSA保守性分析

作者: Kuroneko
日期: 2025.10.6

输入:
    - 步骤1输出的FASTA文件
    - HHblits生成的A3M格式MSA文件
    
输出:
    - 保守性得分CSV文件保存到 step2_output/
    - 保守性可视化图表保存到 step2_output/
    
功能:
    - 解析A3M格式的MSA文件
    - 计算每个位点的保守性得分
    - 生成保守性分析报告
    - 可视化保守性分布
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import logging
from datetime import datetime

# 创建日志目录
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{log_dir}/step2_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class MSAConservationAnalyzer:
    """MSA保守性分析类"""
    
    def __init__(self, output_dir='step2_output'):
        """
        初始化MSA保守性分析器
        
        参数:
            output_dir: 输出文件存储目录
        """
        self.output_dir = output_dir
        self.csv_dir = os.path.join(output_dir, 'conservation_scores')
        self.plot_dir = os.path.join(output_dir, 'conservation_plots')
        
        # 创建输出目录
        os.makedirs(self.csv_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)
        
        logging.info(f"输出目录已创建: {self.output_dir}")
    
    def parse_a3m(self, a3m_file):
        """
        解析A3M格式的MSA文件
        
        参数:
            a3m_file: A3M文件路径
            
        返回:
            序列列表（查询序列移除gap，但比对序列保留gap）
        """
        logging.info(f"正在解析A3M文件: {a3m_file}")
        
        sequences = []
        current_seq = []
        
        with open(a3m_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    # 保存前一个序列
                    if current_seq:
                        sequences.append(''.join(current_seq))
                        current_seq = []
                elif line:
                    # A3M格式: 小写字母是相对于查询序列的插入
                    # 保留大写字母和gap，移除小写字母
                    filtered_line = ''.join(c for c in line if c.isupper() or c == '-')
                    current_seq.append(filtered_line)
        
        # 添加最后一个序列
        if current_seq:
            sequences.append(''.join(current_seq))
        
        logging.info(f"从A3M文件解析出 {len(sequences)} 条序列")
        
        if len(sequences) == 0:
            raise ValueError("A3M文件中没有找到序列")
        
        return sequences
    
    def calculate_conservation_entropy(self, sequences):
        """
        使用Shannon熵计算保守性得分
        熵越低 = 保守性越高
        
        参数:
            sequences: 比对序列列表
            
        返回:
            保守性得分列表（0-1，越高越保守）
        """
        logging.info("正在使用Shannon熵法计算保守性得分")
        
        # 获取比对长度
        alignment_length = len(sequences[0])
        num_sequences = len(sequences)
        
        conservation_scores = []
        amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        
        for pos in range(alignment_length):
            # 获取该位置的所有氨基酸
            column = [seq[pos] for seq in sequences if pos < len(seq)]
            
            # 统计氨基酸（排除gap）
            aa_counts = Counter([aa for aa in column if aa != '-' and aa in amino_acids])
            
            if not aa_counts:
                # 该位置全是gap
                conservation_scores.append(0.0)
                continue
            
            # 计算频率
            total = sum(aa_counts.values())
            frequencies = [count / total for count in aa_counts.values()]
            
            # 计算Shannon熵
            entropy = -sum(f * np.log2(f) for f in frequencies if f > 0)
            
            # 归一化熵值到0-1范围
            # 20种氨基酸的最大熵值是 log2(20) = 4.32
            max_entropy = np.log2(20)
            normalized_entropy = entropy / max_entropy
            
            # 转换为保守性得分（1 - 归一化熵值）
            conservation = 1.0 - normalized_entropy
            conservation_scores.append(conservation)
        
        logging.info(f"已计算 {len(conservation_scores)} 个位点的保守性")
        return conservation_scores
    
    def calculate_conservation_frequency(self, sequences):
        """
        使用最常见氨基酸的频率计算保守性得分
        
        参数:
            sequences: 比对序列列表
            
        返回:
            保守性得分列表（0-1，越高越保守）
        """
        logging.info("正在使用频率法计算保守性得分")
        
        alignment_length = len(sequences[0])
        conservation_scores = []
        amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        
        for pos in range(alignment_length):
            # 获取该位置的所有氨基酸
            column = [seq[pos] for seq in sequences if pos < len(seq)]
            
            # 统计氨基酸（排除gap）
            aa_counts = Counter([aa for aa in column if aa != '-' and aa in amino_acids])
            
            if not aa_counts:
                conservation_scores.append(0.0)
                continue
            
            # 获取最常见氨基酸的频率
            total = sum(aa_counts.values())
            most_common_freq = aa_counts.most_common(1)[0][1] / total
            
            conservation_scores.append(most_common_freq)
        
        return conservation_scores
    
    def save_conservation_scores(self, conservation_scores, query_sequence, output_name):
        """
        将保守性得分保存为CSV文件
        
        参数:
            conservation_scores: 保守性得分列表
            query_sequence: 查询序列（MSA中的第一条序列）
            output_name: 输出文件基础名称
            
        返回:
            CSV文件路径
        """
        csv_path = os.path.join(self.csv_dir, f"{output_name}_conservation.csv")
        
        # 创建数据框
        df = pd.DataFrame({
            'Position': range(1, len(conservation_scores) + 1),
            'Residue': list(query_sequence),
            'Conservation_Score': conservation_scores,
            'Conserved': ['Yes' if score > 0.7 else 'No' for score in conservation_scores]
        })
        
        # 保存到CSV
        df.to_csv(csv_path, index=False)
        
        logging.info(f"保守性得分已保存到: {csv_path}")
        
        # 打印统计信息
        highly_conserved = sum(1 for score in conservation_scores if score > 0.7)
        moderately_conserved = sum(1 for score in conservation_scores if 0.4 < score <= 0.7)
        variable = sum(1 for score in conservation_scores if score <= 0.4)
        
        logging.info(f"保守性统计:")
        logging.info(f"  高度保守 (>0.7): {highly_conserved} 个位点")
        logging.info(f"  中度保守 (0.4-0.7): {moderately_conserved} 个位点")
        logging.info(f"  可变 (<=0.4): {variable} 个位点")
        
        return csv_path
    
    def plot_conservation(self, conservation_scores, query_sequence, output_name):
        """
        创建保守性可视化图表
        
        参数:
            conservation_scores: 保守性得分列表
            query_sequence: 查询序列
            output_name: 输出文件基础名称
            
        返回:
            图表文件路径
        """
        plot_path = os.path.join(self.plot_dir, f"{output_name}_conservation_plot.png")
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(max(12, len(conservation_scores) * 0.3), 6))
        
        positions = range(1, len(conservation_scores) + 1)
        colors = ['red' if score > 0.7 else 'orange' if score > 0.4 else 'blue' 
                  for score in conservation_scores]
        
        # 绘制柱状图
        bars = ax.bar(positions, conservation_scores, color=colors, alpha=0.7, edgecolor='black')
        
        # 添加阈值线
        ax.axhline(y=0.7, color='red', linestyle='--', linewidth=1, label='High conservation')
        ax.axhline(y=0.4, color='orange', linestyle='--', linewidth=1, label='Moderate conservation')
        
        # 格式化
        ax.set_xlabel('Position', fontsize=12)
        ax.set_ylabel('Conservation Score', fontsize=12)
        ax.set_title(f'Conservation Profile: {output_name}', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # 在x轴添加残基标签
        if len(query_sequence) <= 100:
            ax.set_xticks(positions)
            ax.set_xticklabels([f"{query_sequence[i]}{i+1}" for i in range(len(query_sequence))], 
                               rotation=90, fontsize=8)
        else:
            # 对于长序列，每10个位置显示一个标签
            tick_positions = list(range(0, len(positions), 10))
            ax.set_xticks([positions[i] for i in tick_positions])
            ax.set_xticklabels([f"{query_sequence[i]}{i+1}" for i in tick_positions], 
                              rotation=90, fontsize=8)
        
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"保守性图表已保存到: {plot_path}")
        return plot_path
    
    def analyze(self, a3m_file, output_name, method='entropy'):
        """
        执行完整的保守性分析
        
        参数:
            a3m_file: A3M格式MSA文件路径
            output_name: 输出文件基础名称
            method: 保守性计算方法（'entropy' 或 'frequency'）
            
        返回:
            包含输出文件路径的字典
        """
        logging.info(f"开始保守性分析: {output_name}")
        logging.info(f"使用方法: {method}")
        
        # 解析A3M文件
        sequences = self.parse_a3m(a3m_file)
        query_sequence = sequences[0].replace('-', '')  # 从查询序列移除gap
        
        # 计算保守性
        if method == 'entropy':
            conservation_scores = self.calculate_conservation_entropy(sequences)
        elif method == 'frequency':
            conservation_scores = self.calculate_conservation_frequency(sequences)
        else:
            raise ValueError(f"未知方法: {method}。请使用 'entropy' 或 'frequency'")
        
        # 保存结果
        csv_path = self.save_conservation_scores(conservation_scores, query_sequence, output_name)
        plot_path = self.plot_conservation(conservation_scores, query_sequence, output_name)
        
        logging.info(f"分析完成: {output_name}")
        
        return {
            'csv': csv_path,
            'plot': plot_path,
            'conservation_scores': conservation_scores,
            'query_sequence': query_sequence
        }


def main():
    """
    主执行函数
    
    修改下面的参数来分析你的MSA文件
    """
    # 配置参数
    A3M_FILE = 'step1_output/4kha_A.a3m'
    OUTPUT_NAME = '4KHA_A'
    METHOD = 'entropy'
    OUTPUT_DIR = 'step2_output'
    
    logging.info("="*60)
    logging.info("步骤2: MSA保守性分析")
    logging.info("="*60)
    
    # 检查A3M文件是否存在
    if not os.path.exists(A3M_FILE):
        logging.error(f"未找到A3M文件: {A3M_FILE}")
        logging.error("请先运行HHblits生成MSA")
        logging.error("示例命令:")
        logging.error("hhblits -i input.fasta -d database -oa3m output.a3m -n 3 -cpu 4")
        sys.exit(1)
    
    # 初始化分析器
    analyzer = MSAConservationAnalyzer(output_dir=OUTPUT_DIR)
    
    # 执行分析
    results = analyzer.analyze(A3M_FILE, OUTPUT_NAME, method=METHOD)
    
    # 打印结果摘要
    logging.info("="*60)
    logging.info("结果摘要:")
    logging.info(f"  CSV文件: {results['csv']}")
    logging.info(f"  图表文件: {results['plot']}")
    logging.info(f"  序列长度: {len(results['query_sequence'])}")
    logging.info(f"  平均保守性: {np.mean(results['conservation_scores']):.3f}")
    logging.info("="*60)
    logging.info("步骤2 成功完成")
    
    return results


if __name__ == "__main__":
    try:
        results = main()
    except Exception as e:
        logging.error(f"步骤2 失败，错误: {str(e)}")
        sys.exit(1)


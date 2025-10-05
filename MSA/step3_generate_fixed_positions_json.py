"""
步骤3: 生成ProteinMPNN的固定位点JSON文件

作者: Kuroneko
日期: 2025.10.6

输入:
    - 步骤2输出的保守性得分CSV文件
    - 步骤1下载的PDB文件
    - 固定位点的保守性阈值
    
输出:
    - fixed_positions.json文件保存到 step3_output/
    - 汇总报告保存到 step3_output/
    
功能:
    - 从CSV读取保守性得分
    - 识别高度保守的位点
    - 将序列位置映射到PDB残基编号
    - 生成ProteinMPNN兼容的JSON格式
    - 支持多链PDB文件
"""

import os
import sys
import json
import pandas as pd
from Bio import PDB
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
        logging.FileHandler(f'{log_dir}/step3_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class FixedPositionGenerator:
    """生成ProteinMPNN固定位点JSON的类"""
    
    def __init__(self, output_dir='step3_output'):
        """
        初始化固定位点生成器
        
        参数:
            output_dir: 输出文件存储目录
        """
        self.output_dir = output_dir
        self.json_dir = os.path.join(output_dir, 'json_files')
        self.report_dir = os.path.join(output_dir, 'reports')
        
        # 创建输出目录
        os.makedirs(self.json_dir, exist_ok=True)
        os.makedirs(self.report_dir, exist_ok=True)
        
        logging.info(f"输出目录已创建: {self.output_dir}")
    
    def read_conservation_scores(self, csv_file):
        """
        从CSV文件读取保守性得分
        
        参数:
            csv_file: 步骤2输出的保守性CSV文件路径
            
        返回:
            包含保守性数据的DataFrame
        """
        logging.info(f"正在从以下位置读取保守性得分: {csv_file}")
        
        df = pd.read_csv(csv_file)
        
        # 验证CSV格式
        required_columns = ['Position', 'Residue', 'Conservation_Score']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"缺少必需列: {col}")
        
        logging.info(f"从CSV加载了 {len(df)} 个位点")
        return df
    
    def get_pdb_residue_numbers(self, pdb_file, chain_id):
        """
        从PDB文件获取指定链的残基编号
        
        参数:
            pdb_file: PDB文件路径
            chain_id: 链标识符
            
        返回:
            残基编号列表（PDB编号）
        """
        logging.info(f"正在读取PDB文件: {pdb_file}, 链: {chain_id}")
        
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure('protein', pdb_file)
        
        residue_numbers = []
        
        for model in structure:
            for chain in model:
                if chain.id == chain_id:
                    for residue in chain:
                        # 只处理氨基酸残基
                        if residue.id[0] == ' ':
                            residue_numbers.append(residue.id[1])
        
        if not residue_numbers:
            raise ValueError(f"未找到链 {chain_id} 或该链没有残基")
        
        logging.info(f"在链 {chain_id} 中找到 {len(residue_numbers)} 个残基")
        return residue_numbers
    
    def identify_fixed_positions(self, conservation_df, threshold=0.7):
        """
        根据保守性阈值识别需要固定的位点
        
        参数:
            conservation_df: 包含保守性得分的DataFrame
            threshold: 保守性阈值（默认: 0.7）
            
        返回:
            需要固定的位点列表（1-based索引）
        """
        logging.info(f"正在识别固定位点，阈值: {threshold}")
        
        fixed_positions = conservation_df[
            conservation_df['Conservation_Score'] >= threshold
        ]['Position'].tolist()
        
        logging.info(f"识别出 {len(fixed_positions)} 个固定位点")
        return fixed_positions
    
    def generate_proteinmpnn_json(self, pdb_file, chain_id, fixed_positions, 
                                   pdb_residue_numbers, output_name):
        """
        生成ProteinMPNN兼容的fixed_positions.json文件
        
        参数:
            pdb_file: PDB文件路径
            chain_id: 链标识符
            fixed_positions: 需要固定的序列位置列表（1-based）
            pdb_residue_numbers: PDB残基编号列表
            output_name: 输出文件基础名称
            
        返回:
            JSON文件路径
        """
        logging.info(f"正在为 {output_name} 生成ProteinMPNN JSON")
        
        # 将序列位置映射到PDB残基编号
        pdb_fixed_positions = []
        for seq_pos in fixed_positions:
            if 1 <= seq_pos <= len(pdb_residue_numbers):
                pdb_res_num = pdb_residue_numbers[seq_pos - 1]
                pdb_fixed_positions.append(pdb_res_num)
            else:
                logging.warning(f"序列位置 {seq_pos} 超出范围")
        
        # 创建ProteinMPNN格式的JSON
        pdb_basename = os.path.basename(pdb_file)
        
        fixed_positions_dict = {
            pdb_basename: {
                chain_id: pdb_fixed_positions
            }
        }
        
        # 保存JSON文件
        json_path = os.path.join(self.json_dir, f"{output_name}_fixed_positions.json")
        
        with open(json_path, 'w') as f:
            json.dump(fixed_positions_dict, f, indent=2)
        
        logging.info(f"JSON文件已保存到: {json_path}")
        return json_path
    
    def generate_designed_positions_json(self, pdb_file, chain_id, fixed_positions,
                                          pdb_residue_numbers, output_name):
        """
        生成designed_positions.json（不固定的位点）
        这是fixed_positions的补集
        
        参数:
            pdb_file: PDB文件路径
            chain_id: 链标识符
            fixed_positions: 需要固定的序列位置列表（1-based）
            pdb_residue_numbers: PDB残基编号列表
            output_name: 输出文件基础名称
            
        返回:
            JSON文件路径
        """
        logging.info(f"正在为 {output_name} 生成设计位点JSON")
        
        # 获取所有位点
        all_positions = set(range(1, len(pdb_residue_numbers) + 1))
        fixed_positions_set = set(fixed_positions)
        
        # 设计位点是那些不固定的位点
        designed_positions = sorted(list(all_positions - fixed_positions_set))
        
        # 映射到PDB残基编号
        pdb_designed_positions = []
        for seq_pos in designed_positions:
            if 1 <= seq_pos <= len(pdb_residue_numbers):
                pdb_res_num = pdb_residue_numbers[seq_pos - 1]
                pdb_designed_positions.append(pdb_res_num)
        
        # 创建JSON
        pdb_basename = os.path.basename(pdb_file)
        
        designed_positions_dict = {
            pdb_basename: {
                chain_id: pdb_designed_positions
            }
        }
        
        # 保存JSON文件
        json_path = os.path.join(self.json_dir, f"{output_name}_designed_positions.json")
        
        with open(json_path, 'w') as f:
            json.dump(designed_positions_dict, f, indent=2)
        
        logging.info(f"设计位点JSON已保存到: {json_path}")
        return json_path
    
    def generate_report(self, conservation_df, fixed_positions, pdb_residue_numbers,
                       chain_id, output_name, threshold):
        """
        生成汇总报告
        
        参数:
            conservation_df: 包含保守性得分的DataFrame
            fixed_positions: 固定位点列表
            pdb_residue_numbers: PDB残基编号列表
            chain_id: 链标识符
            output_name: 输出文件基础名称
            threshold: 使用的保守性阈值
            
        返回:
            报告文件路径
        """
        report_path = os.path.join(self.report_dir, f"{output_name}_report.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("固定位点分析报告\n")
            f.write("="*70 + "\n\n")
            f.write(f"作者: Kuroneko\n")
            f.write(f"日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"输出名称: {output_name}\n")
            f.write(f"链: {chain_id}\n")
            f.write(f"保守性阈值: {threshold}\n\n")
            
            f.write(f"总残基数: {len(conservation_df)}\n")
            f.write(f"固定位点数: {len(fixed_positions)}\n")
            f.write(f"设计位点数: {len(conservation_df) - len(fixed_positions)}\n")
            f.write(f"固定百分比: {100 * len(fixed_positions) / len(conservation_df):.1f}%\n\n")
            
            f.write("="*70 + "\n")
            f.write("固定位点详情\n")
            f.write("="*70 + "\n\n")
            f.write(f"{'序列位置':<8} {'PDB位置':<8} {'残基':<8} {'保守性':<15}\n")
            f.write("-"*70 + "\n")
            
            for seq_pos in fixed_positions:
                if 1 <= seq_pos <= len(pdb_residue_numbers):
                    pdb_pos = pdb_residue_numbers[seq_pos - 1]
                    residue = conservation_df.loc[seq_pos - 1, 'Residue']
                    score = conservation_df.loc[seq_pos - 1, 'Conservation_Score']
                    f.write(f"{seq_pos:<8} {pdb_pos:<8} {residue:<8} {score:<15.3f}\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("保守性得分统计\n")
            f.write("="*70 + "\n\n")
            f.write(f"平均值: {conservation_df['Conservation_Score'].mean():.3f}\n")
            f.write(f"中位数: {conservation_df['Conservation_Score'].median():.3f}\n")
            f.write(f"最小值: {conservation_df['Conservation_Score'].min():.3f}\n")
            f.write(f"最大值: {conservation_df['Conservation_Score'].max():.3f}\n")
        
        logging.info(f"报告已保存到: {report_path}")
        return report_path
    
    def process(self, conservation_csv, pdb_file, chain_id, output_name, 
                threshold=0.7, generate_designed=True):
        """
        执行完整流程生成固定位点JSON
        
        参数:
            conservation_csv: 步骤2输出的保守性CSV路径
            pdb_file: PDB文件路径
            chain_id: 链标识符
            output_name: 输出文件基础名称
            threshold: 固定位点的保守性阈值
            generate_designed: 是否生成designed_positions.json
            
        返回:
            包含输出文件路径的字典
        """
        logging.info(f"正在处理 {output_name}")
        logging.info(f"保守性阈值: {threshold}")
        
        # 读取保守性得分
        conservation_df = self.read_conservation_scores(conservation_csv)
        
        # 获取PDB残基编号
        pdb_residue_numbers = self.get_pdb_residue_numbers(pdb_file, chain_id)
        
        # 验证长度是否匹配
        if len(conservation_df) != len(pdb_residue_numbers):
            logging.warning(f"长度不匹配: CSV有 {len(conservation_df)} 个位点, "
                          f"PDB有 {len(pdb_residue_numbers)} 个残基")
            # 使用最小长度
            min_length = min(len(conservation_df), len(pdb_residue_numbers))
            conservation_df = conservation_df.iloc[:min_length]
            pdb_residue_numbers = pdb_residue_numbers[:min_length]
            logging.warning(f"已截断为 {min_length} 个位点")
        
        # 识别固定位点
        fixed_positions = self.identify_fixed_positions(conservation_df, threshold)
        
        # 生成固定位点JSON
        fixed_json_path = self.generate_proteinmpnn_json(
            pdb_file, chain_id, fixed_positions, pdb_residue_numbers, output_name
        )
        
        # 生成设计位点JSON（可选）
        designed_json_path = None
        if generate_designed:
            designed_json_path = self.generate_designed_positions_json(
                pdb_file, chain_id, fixed_positions, pdb_residue_numbers, output_name
            )
        
        # 生成报告
        report_path = self.generate_report(
            conservation_df, fixed_positions, pdb_residue_numbers,
            chain_id, output_name, threshold
        )
        
        logging.info(f"处理完成: {output_name}")
        
        return {
            'fixed_positions_json': fixed_json_path,
            'designed_positions_json': designed_json_path,
            'report': report_path,
            'num_fixed': len(fixed_positions),
            'num_designed': len(conservation_df) - len(fixed_positions)
        }


def main():
    """
    主执行函数
    
    修改下面的参数来生成固定位点JSON
    """
    # 配置参数
    CONSERVATION_CSV = 'step2_output/conservation_scores/4KHA_A_conservation.csv'
    PDB_FILE = 'step1_output/pdb_files/4kha.pdb'
    CHAIN_ID = 'A'
    OUTPUT_NAME = '4KHA_A'
    CONSERVATION_THRESHOLD = 0.7
    GENERATE_DESIGNED = True
    OUTPUT_DIR = 'step3_output'
    
    logging.info("="*60)
    logging.info("步骤3: 生成固定位点JSON文件")
    logging.info("="*60)
    
    # 检查输入文件是否存在
    if not os.path.exists(CONSERVATION_CSV):
        logging.error(f"未找到保守性CSV文件: {CONSERVATION_CSV}")
        logging.error("请先运行步骤2")
        sys.exit(1)
    
    if not os.path.exists(PDB_FILE):
        logging.error(f"未找到PDB文件: {PDB_FILE}")
        logging.error("请先运行步骤1")
        sys.exit(1)
    
    # 初始化生成器
    generator = FixedPositionGenerator(output_dir=OUTPUT_DIR)
    
    # 执行处理
    results = generator.process(
        conservation_csv=CONSERVATION_CSV,
        pdb_file=PDB_FILE,
        chain_id=CHAIN_ID,
        output_name=OUTPUT_NAME,
        threshold=CONSERVATION_THRESHOLD,
        generate_designed=GENERATE_DESIGNED
    )
    
    # 打印结果摘要
    logging.info("="*60)
    logging.info("结果摘要:")
    logging.info(f"  固定位点JSON: {results['fixed_positions_json']}")
    if results['designed_positions_json']:
        logging.info(f"  设计位点JSON: {results['designed_positions_json']}")
    logging.info(f"  报告: {results['report']}")
    logging.info(f"  固定位点数量: {results['num_fixed']}")
    logging.info(f"  设计位点数量: {results['num_designed']}")
    logging.info("="*60)
    logging.info("步骤3 成功完成")
    logging.info("JSON文件已准备好，可用于ProteinMPNN")
    
    return results


if __name__ == "__main__":
    try:
        results = main()
    except Exception as e:
        logging.error(f"步骤3 失败，错误: {str(e)}")
        sys.exit(1)


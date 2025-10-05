"""
步骤1: 下载PDB文件并提取链序列

作者: Kuroneko
日期: 2025.10.6

输入: 
    - UniProt ID 或 PDB ID
    - 需要提取的链列表
    
输出:
    - 下载的PDB文件保存到 step1_output/
    - 每条链的FASTA文件保存到 step1_output/
    
功能:
    - 从RCSB PDB下载PDB文件
    - 提取指定链的氨基酸序列
    - 生成用于MSA分析的FASTA文件
    - 支持多链PDB文件
"""

import os
import sys
import requests
from Bio import PDB
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
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
        logging.FileHandler(f'{log_dir}/step1_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class PDBDownloader:
    """处理PDB文件下载和序列提取的类"""
    
    def __init__(self, output_dir='step1_output'):
        """
        初始化PDB下载器
        
        参数:
            output_dir: 输出文件存储目录
        """
        self.output_dir = output_dir
        self.pdb_dir = os.path.join(output_dir, 'pdb_files')
        self.fasta_dir = os.path.join(output_dir, 'fasta_files')
        
        # 创建输出目录
        os.makedirs(self.pdb_dir, exist_ok=True)
        os.makedirs(self.fasta_dir, exist_ok=True)
        
        logging.info(f"输出目录已创建: {self.output_dir}")
    
    def download_pdb(self, pdb_id):
        """
        从RCSB PDB下载PDB文件
        
        参数:
            pdb_id: 4位字符的PDB ID
            
        返回:
            下载的PDB文件路径
        """
        pdb_id = pdb_id.lower()
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        pdb_path = os.path.join(self.pdb_dir, f"{pdb_id}.pdb")
        
        # 检查文件是否已存在
        if os.path.exists(pdb_path):
            logging.info(f"PDB文件 {pdb_id}.pdb 已存在，跳过下载")
            return pdb_path
        
        # 下载PDB文件
        logging.info(f"正在下载PDB文件: {pdb_id}")
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            with open(pdb_path, 'w') as f:
                f.write(response.text)
            
            logging.info(f"成功下载 {pdb_id}.pdb")
            return pdb_path
            
        except requests.exceptions.RequestException as e:
            logging.error(f"下载PDB {pdb_id} 失败: {str(e)}")
            raise
    
    def extract_chain_sequence(self, pdb_path, chain_id):
        """
        从指定链提取氨基酸序列
        
        参数:
            pdb_path: PDB文件路径
            chain_id: 链标识符
            
        返回:
            序列字符串
        """
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure('protein', pdb_path)
        
        # 三字母到单字母氨基酸代码转换
        three_to_one = {
            'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
            'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
            'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
            'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
            'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
        }
        
        sequence = []
        
        # 遍历模型和链
        for model in structure:
            for chain in model:
                if chain.id == chain_id:
                    for residue in chain:
                        # 只处理氨基酸残基
                        if residue.id[0] == ' ':
                            res_name = residue.get_resname()
                            if res_name in three_to_one:
                                sequence.append(three_to_one[res_name])
                            else:
                                logging.warning(f"未知残基: {res_name} 位置 {residue.id[1]}")
                                sequence.append('X')
                    
                    if sequence:
                        logging.info(f"提取链 {chain_id}: {len(sequence)} 个残基")
                        return ''.join(sequence)
        
        raise ValueError(f"在PDB文件中未找到链 {chain_id}")
    
    def save_fasta(self, sequence, pdb_id, chain_id, description=''):
        """
        将序列保存为FASTA文件
        
        参数:
            sequence: 氨基酸序列
            pdb_id: PDB标识符
            chain_id: 链标识符
            description: 可选描述信息
            
        返回:
            FASTA文件路径
        """
        fasta_path = os.path.join(self.fasta_dir, f"{pdb_id}_{chain_id}.fasta")
        
        # 创建SeqRecord对象
        if not description:
            description = f"{pdb_id} chain {chain_id}"
        
        record = SeqRecord(
            Seq(sequence),
            id=f"{pdb_id}_{chain_id}",
            description=description
        )
        
        # 写入文件
        with open(fasta_path, 'w') as f:
            SeqIO.write(record, f, 'fasta')
        
        logging.info(f"FASTA文件已保存: {fasta_path}")
        return fasta_path
    
    def process_pdb(self, pdb_id, chain_ids):
        """
        处理PDB文件: 下载并提取指定链的序列
        
        参数:
            pdb_id: PDB标识符
            chain_ids: 链标识符列表
            
        返回:
            字典，映射chain_id到FASTA文件路径
        """
        logging.info(f"正在处理PDB: {pdb_id}, 链: {chain_ids}")
        
        # 下载PDB文件
        pdb_path = self.download_pdb(pdb_id)
        
        results = {}
        
        # 提取每条链的序列
        for chain_id in chain_ids:
            try:
                sequence = self.extract_chain_sequence(pdb_path, chain_id)
                fasta_path = self.save_fasta(sequence, pdb_id, chain_id)
                results[chain_id] = fasta_path
                
            except Exception as e:
                logging.error(f"处理链 {chain_id} 失败: {str(e)}")
                results[chain_id] = None
        
        logging.info(f"步骤1 完成，PDB: {pdb_id}")
        return results


def main():
    """
    主执行函数
    
    修改下面的参数来处理你的目标PDB
    """
    # 配置参数
    PDB_ID = '4KHA'
    CHAIN_IDS = ['A', 'B']
    OUTPUT_DIR = 'step1_output'
    
    logging.info("="*60)
    logging.info("步骤1: 下载PDB文件并提取链序列")
    logging.info("="*60)
    
    # 初始化下载器
    downloader = PDBDownloader(output_dir=OUTPUT_DIR)
    
    # 处理PDB
    results = downloader.process_pdb(PDB_ID, CHAIN_IDS)
    
    # 打印结果摘要
    logging.info("="*60)
    logging.info("结果摘要:")
    for chain_id, fasta_path in results.items():
        if fasta_path:
            logging.info(f"  链 {chain_id}: {fasta_path}")
        else:
            logging.info(f"  链 {chain_id}: 失败")
    logging.info("="*60)
    logging.info("步骤1 成功完成")
    
    return results


if __name__ == "__main__":
    try:
        results = main()
    except Exception as e:
        logging.error(f"步骤1 失败，错误: {str(e)}")
        sys.exit(1)


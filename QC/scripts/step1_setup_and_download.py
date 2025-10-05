#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step1_setup_and_download.py - 环境设置与示例数据准备
作者: Kuroneko
日期: 2025.10.04

功能说明:
    1. 创建项目所需的目录结构（data/, outputs/, logs/）
    2. 询问用户是否下载公开真实数据（UniProt、GenBank、PDB）
    3. 若用户同意，从公共数据库下载示例数据并记录来源
    4. 若用户拒绝或网络不可用，生成合成测试序列
    5. 创建或更新数据清单文件（manifest）
    6. 生成初始 README.md 文件

输入输出:
    输入: 用户交互选择（是否下载公开数据、accession/id）
    输出:
        - data/download_manifest.txt（公开数据下载记录）
        - data/synthetic_manifest.txt（合成数据生成记录）
        - data/example_public_uniprot.fasta（若下载）
        - data/example_public_genbank.gb（若下载）
        - data/example_public_structure.pdb（若下载）
        - data/example_synthetic.fasta（若生成）
        - data/input_sequences.fasta（默认输入文件）
        - README.md

运行示例:
    python step1_setup_and_download.py
    
设计决策:
    - 使用 requests 库下载（而非 urllib）以获得更好的错误处理
    - 优先推荐公开数据，提供默认 accession
    - 合成数据使用固定随机种子以确保可复现
    - 所有网络操作包含重试机制（最多3次）
"""

import os
import sys
import random
from pathlib import Path
from datetime import datetime

# 导入工具模块
try:
    from step7_utils import (
        setup_logger, ensure_directories, get_user_input, 
        confirm_action, print_section_header, retry_request,
        PROJECT_ROOT, DATA_DIR, OUTPUT_DIR, LOG_DIR, write_fasta
    )
except ImportError:
    print("错误: 无法导入 step7_utils 模块")
    print("请确保 step7_utils.py 与本脚本在同一目录下")
    sys.exit(1)


# 默认公开数据 accession/id
DEFAULT_UNIPROT_ACC = "P69905"  # 人血红蛋白亚基alpha（小分子，易下载）
DEFAULT_GENBANK_ACC = "NM_000518.5"  # 人HBB基因mRNA（血红蛋白beta链）
DEFAULT_PDB_ID = "1CRN"  # Crambin（小蛋白，46残基）

# 合成数据随机种子
SYNTHETIC_SEED = 42


def download_uniprot_fasta(accession: str, output_file: Path, logger) -> bool:
    """
    从 UniProt 下载 FASTA 格式蛋白序列
    
    参数:
        accession: UniProt accession (如 P69905)
        output_file: 输出文件路径
        logger: 日志记录器
    
    返回:
        True 表示成功，False 表示失败
    """
    try:
        import requests
    except ImportError:
        logger.error("需要安装 requests 库: pip install requests")
        return False
    
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.fasta"
    logger.info(f"正在从 UniProt 下载: {accession}")
    logger.info(f"URL: {url}")
    
    def fetch():
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.text
    
    try:
        content = retry_request(fetch, max_retries=3, delay=2.0)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"成功下载到: {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"下载失败: {e}")
        return False


def download_genbank_record(accession: str, output_file: Path, logger) -> bool:
    """
    从 NCBI 下载 GenBank 记录
    
    参数:
        accession: GenBank accession (如 NM_000518.5)
        output_file: 输出文件路径
        logger: 日志记录器
    
    返回:
        True 表示成功，False 表示失败
    """
    try:
        from Bio import Entrez
    except ImportError:
        logger.error("需要安装 Biopython: pip install biopython")
        return False
    
    # 设置 Entrez email（必需）
    Entrez.email = "kuroneko_bioproject@example.com"
    
    logger.info(f"正在从 NCBI GenBank 下载: {accession}")
    
    try:
        def fetch():
            handle = Entrez.efetch(db="nucleotide", id=accession, rettype="gb", retmode="text")
            return handle.read()
        
        content = retry_request(fetch, max_retries=3, delay=2.0)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"成功下载到: {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"下载失败: {e}")
        return False


def download_pdb_structure(pdb_id: str, output_file: Path, logger) -> bool:
    """
    从 RCSB PDB 下载蛋白结构文件
    
    参数:
        pdb_id: PDB ID (如 1CRN)
        output_file: 输出文件路径
        logger: 日志记录器
    
    返回:
        True 表示成功，False 表示失败
    """
    try:
        import requests
    except ImportError:
        logger.error("需要安装 requests 库: pip install requests")
        return False
    
    pdb_id_upper = pdb_id.upper()
    url = f"https://files.rcsb.org/download/{pdb_id_upper}.pdb"
    logger.info(f"正在从 RCSB PDB 下载: {pdb_id_upper}")
    logger.info(f"URL: {url}")
    
    def fetch():
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.text
    
    try:
        content = retry_request(fetch, max_retries=3, delay=2.0)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"成功下载到: {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"下载失败: {e}")
        return False


def generate_synthetic_sequences(output_file: Path, logger) -> bool:
    """
    生成合成测试序列（用于测试，非真实生物序列）
    
    参数:
        output_file: 输出 FASTA 文件路径
        logger: 日志记录器
    
    返回:
        True 表示成功
    """
    random.seed(SYNTHETIC_SEED)
    
    logger.info(f"生成合成测试序列（随机种子: {SYNTHETIC_SEED}）")
    
    # 标准氨基酸
    amino_acids = list('ACDEFGHIKLMNPQRSTVWY')
    
    # 1. 短序列（约25个残基）
    short_seq = ''.join(random.choices(amino_acids, k=25))
    
    # 2. 中等长度序列（约150个残基）
    medium_seq = ''.join(random.choices(amino_acids, k=150))
    
    # 3. 长序列（约1200个残基，包含重复片段）
    repeat_motif = 'PGPGPG'
    long_seq_parts = []
    for _ in range(200):
        if random.random() < 0.1:  # 10%的位置插入重复片段
            long_seq_parts.append(repeat_motif)
        else:
            long_seq_parts.append(random.choice(amino_acids))
    long_seq = ''.join(long_seq_parts)
    
    # 4. 低复杂度序列（包含 poly-A 区域）
    low_complexity_parts = ['A' * 30]  # poly-A
    low_complexity_parts.append(''.join(random.choices(amino_acids, k=50)))
    low_complexity_parts.append('PGPGPGPGPGPGPGPG')  # proline-glycine repeats
    low_complexity_parts.append(''.join(random.choices(amino_acids, k=30)))
    low_complexity_seq = ''.join(low_complexity_parts)
    
    # 构建记录
    records = [
        {
            'id': 'SYNTH_SHORT_001',
            'description': 'Synthetic short peptide for testing (25 aa)',
            'sequence': short_seq
        },
        {
            'id': 'SYNTH_MEDIUM_002',
            'description': 'Synthetic medium-length protein for testing (150 aa)',
            'sequence': medium_seq
        },
        {
            'id': 'SYNTH_LONG_003',
            'description': 'Synthetic long protein with repeats for testing (approx 1200 aa)',
            'sequence': long_seq
        },
        {
            'id': 'SYNTH_LOWCOMP_004',
            'description': 'Synthetic low-complexity sequence with poly-A and PG-repeats',
            'sequence': low_complexity_seq
        }
    ]
    
    # 写入文件
    write_fasta(records, output_file)
    
    logger.info(f"已生成 {len(records)} 条合成序列:")
    for rec in records:
        logger.info(f"  - {rec['id']}: {len(rec['sequence'])} aa - {rec['description']}")
    
    return True


def write_download_manifest(downloads: list, manifest_file: Path):
    """
    写入下载清单
    
    参数:
        downloads: 下载记录列表，每项包含 (type, accession, url, file, timestamp, success)
        manifest_file: 清单文件路径
    """
    with open(manifest_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("公开数据下载清单\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("作者: Kuroneko\n")
        f.write("日期: 2025.10.04\n")
        f.write("="*70 + "\n\n")
        
        for download in downloads:
            dtype, accession, url, file, timestamp, success = download
            status = "成功" if success else "失败"
            f.write(f"数据类型: {dtype}\n")
            f.write(f"Accession/ID: {accession}\n")
            f.write(f"来源URL: {url}\n")
            f.write(f"本地文件: {file}\n")
            f.write(f"下载时间: {timestamp}\n")
            f.write(f"状态: {status}\n")
            f.write("-"*70 + "\n\n")


def write_synthetic_manifest(manifest_file: Path):
    """
    写入合成数据清单
    
    参数:
        manifest_file: 清单文件路径
    """
    with open(manifest_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("合成测试数据清单\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("作者: Kuroneko\n")
        f.write("日期: 2025.10.04\n")
        f.write("="*70 + "\n\n")
        
        f.write("注意: 这些是计算机生成的合成序列，非真实生物序列\n")
        f.write("用途: 仅用于软件功能测试与演示\n\n")
        
        f.write(f"随机种子: {SYNTHETIC_SEED}\n")
        f.write("生成方法: 使用 Python random.choices 从标准20种氨基酸中随机选择\n\n")
        
        f.write("生成序列:\n")
        f.write("  1. SYNTH_SHORT_001: 短肽链（25 aa）\n")
        f.write("  2. SYNTH_MEDIUM_002: 中等长度蛋白（150 aa）\n")
        f.write("  3. SYNTH_LONG_003: 长蛋白含重复片段（约1200 aa）\n")
        f.write("  4. SYNTH_LOWCOMP_004: 低复杂度序列（poly-A 和 PG-repeats）\n\n")
        
        f.write("复现说明:\n")
        f.write("  使用相同的随机种子可以生成完全相同的序列\n")
        f.write("  代码位于: scripts/step1_setup_and_download.py 中的 generate_synthetic_sequences 函数\n")


def main():
    """
    主函数
    """
    print_section_header("步骤 1: 环境设置与示例数据准备")
    print("作者: Kuroneko")
    print("日期: 2025.10.04\n")
    
    # 设置日志
    logger = setup_logger('step1_setup')
    
    # 确保目录存在
    logger.info("创建项目目录结构...")
    ensure_directories()
    logger.info("目录结构创建完成")
    
    # 询问用户数据获取方式
    print_section_header("数据获取方式选择")
    print("\n本项目可以使用两种方式准备示例数据：")
    print("\n选项 A: 下载公开真实数据（推荐）")
    print("  - 从公共数据库下载真实生物序列和结构")
    print("  - 包括 UniProt 蛋白序列、NCBI GenBank 记录、RCSB PDB 结构")
    print("  - 需要网络连接，数据来源可追溯")
    print("  - 更适合学习真实生物信息学分析")
    print(f"  - 默认示例: UniProt {DEFAULT_UNIPROT_ACC}, GenBank {DEFAULT_GENBANK_ACC}, PDB {DEFAULT_PDB_ID}")
    
    print("\n选项 B: 生成合成测试数据")
    print("  - 计算机生成的随机序列（非真实生物序列）")
    print("  - 不需要网络连接，立即可用")
    print("  - 仅用于软件功能测试")
    print("  - 包含不同长度和复杂度的测试序列")
    
    use_public_data = confirm_action("\n您希望下载公开真实数据吗？（推荐）", default_yes=True)
    
    downloads = []
    
    if use_public_data:
        print_section_header("下载公开数据")
        logger.info("用户选择下载公开真实数据")
        
        # UniProt
        print("\n1. UniProt 蛋白序列")
        uniprot_acc = get_user_input(
            f"请输入 UniProt accession",
            default=DEFAULT_UNIPROT_ACC
        )
        uniprot_file = DATA_DIR / "example_public_uniprot.fasta"
        url = f"https://rest.uniprot.org/uniprotkb/{uniprot_acc}.fasta"
        success = download_uniprot_fasta(uniprot_acc, uniprot_file, logger)
        downloads.append((
            "UniProt Protein",
            uniprot_acc,
            url,
            str(uniprot_file),
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            success
        ))
        
        # GenBank
        print("\n2. NCBI GenBank 记录")
        genbank_acc = get_user_input(
            f"请输入 GenBank accession",
            default=DEFAULT_GENBANK_ACC
        )
        genbank_file = DATA_DIR / "example_public_genbank.gb"
        url = f"https://www.ncbi.nlm.nih.gov/nuccore/{genbank_acc}"
        success = download_genbank_record(genbank_acc, genbank_file, logger)
        downloads.append((
            "NCBI GenBank",
            genbank_acc,
            url,
            str(genbank_file),
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            success
        ))
        
        # PDB
        print("\n3. RCSB PDB 蛋白结构")
        pdb_id = get_user_input(
            f"请输入 PDB ID",
            default=DEFAULT_PDB_ID
        )
        pdb_file = DATA_DIR / "example_public_structure.pdb"
        url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
        success = download_pdb_structure(pdb_id, pdb_file, logger)
        downloads.append((
            "RCSB PDB",
            pdb_id.upper(),
            url,
            str(pdb_file),
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            success
        ))
        
        # 写入下载清单
        manifest_file = DATA_DIR / "download_manifest.txt"
        write_download_manifest(downloads, manifest_file)
        logger.info(f"下载清单已保存到: {manifest_file}")
        
        # 检查是否有失败的下载
        failed_count = sum(1 for d in downloads if not d[5])
        if failed_count > 0:
            logger.warning(f"有 {failed_count} 项下载失败")
            print(f"\n警告: {failed_count} 项下载失败，您可以:")
            print("  1. 检查网络连接后重新运行本脚本")
            print("  2. 手动下载并放置到 data/ 目录")
            print("  3. 选择生成合成测试数据")
            
            if confirm_action("是否要生成合成数据作为备选？", default_yes=True):
                use_public_data = False
    
    if not use_public_data:
        print_section_header("生成合成测试数据")
        logger.info("生成合成测试序列")
        
        synthetic_file = DATA_DIR / "example_synthetic.fasta"
        generate_synthetic_sequences(synthetic_file, logger)
        
        manifest_file = DATA_DIR / "synthetic_manifest.txt"
        write_synthetic_manifest(manifest_file)
        logger.info(f"合成数据清单已保存到: {manifest_file}")
    
    # 创建默认输入文件（如果不存在）
    input_file = DATA_DIR / "input_sequences.fasta"
    if not input_file.exists():
        logger.info(f"创建默认输入文件: {input_file}")
        
        if use_public_data and (DATA_DIR / "example_public_uniprot.fasta").exists():
            # 复制公开数据作为默认输入
            import shutil
            shutil.copy(DATA_DIR / "example_public_uniprot.fasta", input_file)
            logger.info("已将公开 UniProt 数据复制为默认输入")
        elif (DATA_DIR / "example_synthetic.fasta").exists():
            # 复制合成数据作为默认输入
            import shutil
            shutil.copy(DATA_DIR / "example_synthetic.fasta", input_file)
            logger.info("已将合成测试数据复制为默认输入")
        else:
            # 创建空文件并提示用户
            with open(input_file, 'w', encoding='utf-8') as f:
                f.write("# 请将您的 FASTA 格式序列放置在此文件中\n")
                f.write("# 或运行本脚本重新下载/生成示例数据\n")
            logger.warning(f"已创建空的输入文件，请添加序列数据")
    
    # 完成
    print_section_header("设置完成")
    logger.info("环境设置与数据准备完成")
    
    print("\n项目目录结构:")
    print(f"  项目根目录: {PROJECT_ROOT}")
    print(f"  数据目录: {DATA_DIR}")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"  日志目录: {LOG_DIR}")
    
    print("\n可用数据文件:")
    for file in DATA_DIR.glob("*"):
        if file.is_file():
            size_kb = file.stat().st_size / 1024
            print(f"  - {file.name} ({size_kb:.1f} KB)")
    
    print("\n下一步:")
    print("  运行 step2_seq_check_and_seg.py 进行序列检查与 SEG 掩蔽")
    print("  命令: python scripts/step2_seq_check_and_seg.py")


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


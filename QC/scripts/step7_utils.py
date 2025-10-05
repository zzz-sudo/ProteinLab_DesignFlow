#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step7_utils.py - 工具模块
作者: Kuroneko
日期: 2025.10.04

功能说明:
    提供项目中所有脚本共享的工具函数，包括：
    - 日志管理（统一日志格式、自动创建日志文件）
    - 文件操作（目录创建、文件存在性检查、路径解析）
    - 序列验证（氨基酸/核酸检查、长度验证）
    - CSV/FASTA读写辅助函数
    - 网络请求重试机制
    - 隐私提示与用户确认

输入输出:
    本模块被其他脚本导入使用，不直接运行
    
运行示例:
    from step7_utils import setup_logger, check_sequence_validity
    logger = setup_logger('my_script')
    logger.info('Processing started')
"""

import os
import sys
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple


# 项目根目录与标准目录路径
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DATA_DIR = PROJECT_ROOT / 'data'
OUTPUT_DIR = PROJECT_ROOT / 'outputs'
LOG_DIR = PROJECT_ROOT / 'logs'


def ensure_directories():
    """
    确保项目所需的所有目录都存在
    """
    for dir_path in [DATA_DIR, OUTPUT_DIR, LOG_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)


def setup_logger(script_name: str, log_to_file: bool = True) -> logging.Logger:
    """
    创建并配置日志记录器
    
    参数:
        script_name: 脚本名称（用于日志文件名）
        log_to_file: 是否同时输出到文件
    
    返回:
        配置好的 Logger 对象
    """
    ensure_directories()
    
    logger = logging.getLogger(script_name)
    logger.setLevel(logging.INFO)
    
    # 清除已有的 handlers（避免重复）
    if logger.handlers:
        logger.handlers.clear()
    
    # 格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件输出
    if log_to_file:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = LOG_DIR / f'{script_name}_{timestamp}.log'
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f'日志文件: {log_file}')
    
    return logger


def get_user_input(prompt: str, default: str = '', required: bool = False, 
                   valid_options: Optional[List[str]] = None) -> str:
    """
    获取用户输入并验证
    
    参数:
        prompt: 提示信息
        default: 默认值
        required: 是否必须输入
        valid_options: 有效选项列表（忽略大小写）
    
    返回:
        用户输入或默认值
    """
    if default:
        prompt_text = f"{prompt} [默认: {default}]: "
    else:
        prompt_text = f"{prompt}: "
    
    while True:
        user_input = input(prompt_text).strip()
        
        if not user_input:
            if default:
                return default
            elif not required:
                return ''
            else:
                print("此项为必填项，请输入有效内容")
                continue
        
        if valid_options:
            if user_input.lower() in [opt.lower() for opt in valid_options]:
                return user_input
            else:
                print(f"输入无效，请选择: {', '.join(valid_options)}")
                continue
        
        return user_input


def confirm_action(prompt: str, default_yes: bool = False) -> bool:
    """
    要求用户确认操作
    
    参数:
        prompt: 确认提示信息
        default_yes: 默认是否为"是"
    
    返回:
        True表示确认，False表示取消
    """
    suffix = " [Y/n]: " if default_yes else " [y/N]: "
    user_input = input(prompt + suffix).strip().lower()
    
    if not user_input:
        return default_yes
    
    return user_input in ['y', 'yes', 'ye', '是']


def check_sequence_validity(sequence: str, seq_type: str = 'auto') -> Dict[str, any]:
    """
    检查序列有效性，支持特殊字符处理和多序列文件
    
    参数:
        sequence: 序列字符串
        seq_type: 'protein', 'dna', 'rna', 'auto'
    
    返回:
        字典包含：valid (bool), type (str), issues (list), length (int), special_chars (set)
    """
    # 清理序列：移除空格、换行符、数字等
    seq_cleaned = sequence.upper().replace(' ', '').replace('\n', '').replace('\r', '').replace('\t', '')
    # 移除数字（序列位置编号等）
    seq_cleaned = ''.join([c for c in seq_cleaned if not c.isdigit()])
    
    length = len(seq_cleaned)
    issues = []
    special_chars = set()
    
    # 标准字符集定义
    PROTEIN_CHARS = set('ACDEFGHIKLMNPQRSTVWY')
    EXTENDED_PROTEIN_CHARS = PROTEIN_CHARS | set('XBZUJO*-')  # X=未知, B=D/N, Z=E/Q, U=硒半胱氨酸, O=吡咯赖氨酸
    DNA_CHARS = set('ACGT')
    EXTENDED_DNA_CHARS = DNA_CHARS | set('NRYWSMKHBVD-')  # IUPAC 编码
    RNA_CHARS = set('ACGU')
    EXTENDED_RNA_CHARS = RNA_CHARS | set('NRYWSMKHBVD-')
    
    # 检测特殊字符（非生物序列字符）
    seq_chars = set(seq_cleaned)
    all_bio_chars = EXTENDED_PROTEIN_CHARS | EXTENDED_DNA_CHARS | EXTENDED_RNA_CHARS
    special_chars = seq_chars - all_bio_chars
    
    # 检测序列类型
    detected_type = 'unknown'
    
    if seq_type == 'auto':
        if seq_chars <= EXTENDED_DNA_CHARS:
            if 'U' not in seq_chars:
                detected_type = 'dna'
            else:
                detected_type = 'rna'
        elif seq_chars <= EXTENDED_PROTEIN_CHARS:
            detected_type = 'protein'
        else:
            detected_type = 'unknown'
            if special_chars:
                issues.append(f"包含特殊字符: {special_chars}")
    else:
        detected_type = seq_type
    
    # 检查非标准字符
    if detected_type == 'protein':
        non_standard = seq_chars - EXTENDED_PROTEIN_CHARS
        if non_standard:
            issues.append(f"非标准蛋白字符: {non_standard}")
    elif detected_type in ['dna', 'rna']:
        extended_chars = EXTENDED_DNA_CHARS if detected_type == 'dna' else EXTENDED_RNA_CHARS
        non_standard = seq_chars - extended_chars
        if non_standard:
            issues.append(f"非标准核酸字符: {non_standard}")
    
    # 长度检查
    if length == 0:
        issues.append("序列长度为0")
    elif length < 5:
        issues.append(f"序列过短 (length={length})")
    elif length > 10000:
        issues.append(f"序列过长 (length={length})，可能影响处理速度")
    
    # 低复杂度检查（简单版：单字符重复超过30%）
    if length > 0:
        max_single_char_count = max([seq_cleaned.count(char) for char in seq_chars])
        if max_single_char_count / length > 0.3:
            issues.append(f"可能的低复杂度区域（单字符重复率 > 30%）")
    
    # 检查是否包含过多未知字符
    if detected_type == 'protein' and 'X' in seq_chars:
        x_ratio = seq_cleaned.count('X') / length
        if x_ratio > 0.1:
            issues.append(f"包含过多未知氨基酸X ({x_ratio*100:.1f}%)")
    
    return {
        'valid': len(issues) == 0,
        'type': detected_type,
        'issues': issues,
        'length': length,
        'sequence': seq_cleaned,
        'special_chars': special_chars,
        'original_length': len(sequence)
    }


def read_fasta(fasta_file: Path) -> List[Dict[str, str]]:
    """
    读取 FASTA 文件
    
    参数:
        fasta_file: FASTA 文件路径
    
    返回:
        列表，每个元素为字典 {'id': str, 'description': str, 'sequence': str}
    """
    records = []
    current_id = None
    current_desc = ''
    current_seq = []
    
    with open(fasta_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith('>'):
                # 保存前一条记录
                if current_id is not None:
                    records.append({
                        'id': current_id,
                        'description': current_desc,
                        'sequence': ''.join(current_seq)
                    })
                
                # 解析新记录头
                header = line[1:].strip()
                parts = header.split(None, 1)
                current_id = parts[0]
                current_desc = parts[1] if len(parts) > 1 else ''
                current_seq = []
            else:
                current_seq.append(line)
        
        # 保存最后一条记录
        if current_id is not None:
            records.append({
                'id': current_id,
                'description': current_desc,
                'sequence': ''.join(current_seq)
            })
    
    return records


def write_fasta(records: List[Dict[str, str]], output_file: Path, line_width: int = 80):
    """
    写入 FASTA 文件
    
    参数:
        records: 记录列表，格式同 read_fasta 返回值
        output_file: 输出文件路径
        line_width: 序列每行字符数
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in records:
            # 写入header
            header = f">{record['id']}"
            if record.get('description'):
                header += f" {record['description']}"
            f.write(header + '\n')
            
            # 写入序列（分行）
            seq = record['sequence']
            for i in range(0, len(seq), line_width):
                f.write(seq[i:i+line_width] + '\n')


def retry_request(func, max_retries: int = 3, delay: float = 2.0, backoff: float = 2.0):
    """
    网络请求重试装饰器
    
    参数:
        func: 要重试的函数
        max_retries: 最大重试次数
        delay: 初始延迟（秒）
        backoff: 延迟倍增因子
    
    返回:
        包装后的函数结果
    """
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait_time = delay * (backoff ** attempt)
            print(f"请求失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            print(f"等待 {wait_time:.1f} 秒后重试...")
            time.sleep(wait_time)


def format_timestamp() -> str:
    """
    生成格式化的时间戳字符串
    
    返回:
        格式: YYYYMMDD_HHMMSS
    """
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def privacy_warning_blast():
    """
    显示 BLAST 隐私警告并要求确认
    
    返回:
        True 表示用户同意，False 表示拒绝
    """
    print("\n" + "="*70)
    print("隐私与数据安全警告")
    print("="*70)
    print("您即将使用 NCBI BLAST 远程服务。请注意:")
    print("1. 您的序列数据将上传到 NCBI 服务器进行比对")
    print("2. NCBI 可能会保留您的查询序列用于统计或研究目的")
    print("3. 如果您的序列包含敏感、专有或未发表的数据，请勿继续")
    print("4. 建议使用公开序列进行测试，或在本地部署 BLAST")
    print("5. 请遵守您所在机构的数据安全政策")
    print("="*70)
    
    return confirm_action("您确认已理解上述风险并希望继续 BLAST 查询吗？", default_yes=False)


def print_section_header(title: str):
    """
    打印格式化的章节标题
    
    参数:
        title: 标题文本
    """
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def print_progress(current: int, total: int, prefix: str = '进度'):
    """
    打印进度信息
    
    参数:
        current: 当前进度
        total: 总数
        prefix: 前缀文本
    """
    percentage = (current / total * 100) if total > 0 else 0
    print(f"\r{prefix}: {current}/{total} ({percentage:.1f}%)", end='', flush=True)
    if current == total:
        print()  # 完成后换行


if __name__ == '__main__':
    print("step7_utils.py - 工具模块")
    print("作者: Kuroneko")
    print("日期: 2025.10.04")
    print("\n本模块提供通用工具函数，不直接运行。")
    print("请在其他脚本中导入使用，例如:")
    print("    from step7_utils import setup_logger, check_sequence_validity")
    print("\n可用函数列表:")
    print("  - setup_logger: 日志管理")
    print("  - get_user_input: 用户输入获取")
    print("  - confirm_action: 操作确认")
    print("  - check_sequence_validity: 序列验证")
    print("  - read_fasta / write_fasta: FASTA 文件读写")
    print("  - retry_request: 网络请求重试")
    print("  - privacy_warning_blast: BLAST 隐私警告")
    
    # 简单测试
    print("\n运行简单测试...")
    test_seq = "ACDEFGHIKLMNPQRSTVWY"
    result = check_sequence_validity(test_seq, 'auto')
    print(f"\n测试序列: {test_seq}")
    print(f"验证结果: {result}")


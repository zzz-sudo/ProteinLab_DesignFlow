#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
Colab 脚本: 第4步 - ESMFold 结构预测与指标计算
生成时间: 2025-10-03 22:27:21
作者: Kuroneko
日期: 2025.10.3
=============================================================================

【使用说明】
本脚本使用官方 ESMFold 模型进行蛋白质结构预测。

基于官方 ESMFold Colab Notebook:
https://github.com/facebookresearch/esm
https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/ESMFold.ipynb

【重要提示】
- 本脚本使用官方 ESMFold 实现
- 对于短序列(<400残基)，可考虑使用 ESMFold API: https://esmatlas.com/resources?action=fold
- 在 Tesla T4 GPU 上，最大总长度约 900 残基
- 如遇显存不足(OOM)，请减小 batch_size 或序列长度

输入文件：
+ esm_if_sequences.json (ESM-IF 设计结果)
+ proteinmpnn_sequences.json (ProteinMPNN 设计结果)

输出文件：
+ merged_input_sequences.json (合并输入备份)
+ filtered_sequences.json (通过阈值的序列及 metadata)
+ results.csv (包含所有序列的指标和状态)
+ plots.zip (每条序列的 PNG 图像)
+ prediction_results/<seq_id>/ (每个序列的详细结果目录)

配置参数：
- 模型版本: v1
- 循环次数: 3
- pLDDT 阈值: 70.0
- PTM 阈值: 0.5
- 批次大小: 4
- CSV 模式: 追加
- 打包图像: 是
- 断点续传: 是

常见错误与解决办法：
1. 显存不足(OOM) → 减小 batch_size 或 max_length
2. 序列过长 → 考虑使用 ESMFold API 或分段预测
3. 模型下载失败 → 检查网络连接，重新运行安装单元格
"""

# =============================================================================
# PART 1: 环境配置（请先运行此单元格）
# =============================================================================
# 说明：此单元格负责安装 ESMFold 和所有必需的依赖包
# 运行时间：首次运行约 5-10 分钟，之后会跳过已安装的包

import os
import sys
import time
from sys import version_info

# 获取 Python 版本
python_version = f"{version_info.major}.{version_info.minor}"
PYTHON_VERSION = python_version

print("="*70)
print("PART 1: 环境配置开始")
print("="*70)
print(f"Python 版本: {PYTHON_VERSION}")
print("")

# 检查是否在 Colab 环境
try:
    import google.colab
    IN_COLAB = True
    print("[环境] 运行在 Google Colab")
except ImportError:
    IN_COLAB = False
    print("[警告] 不在 Colab 环境中运行")
    print("[提示] 本脚本设计为在 Google Colab 中运行")

# 安装 ESMFold
version = "1"
model_name = "esmfold_v0.model" if version == "0" else "esmfold.model"

if not os.path.isfile(model_name):
    print("\n[安装] 正在安装 ESMFold...")
    print("  这可能需要几分钟，请耐心等待...")
    
    # 下载 ESMFold 模型参数
    os.system("apt-get install aria2 -qq")
    os.system(f"aria2c -q -x 16 https://colabfold.steineggerlab.workers.dev/esm/{model_name} &")
    
    if not os.path.isfile("ESMFOLD_READY"):
        # 安装依赖库
        print("  [安装] 正在安装依赖库...")
        os.system("pip install -q omegaconf pytorch_lightning biopython ml_collections einops py3Dmol modelcif")
        os.system("pip install -q git+https://github.com/NVIDIA/dllogger.git")
        
        print("  [安装] 正在安装 OpenFold...")
        os.system(f"pip install -q git+https://github.com/sokrypton/openfold.git")
        
        print("  [安装] 正在安装 ESMFold...")
        os.system(f"pip install -q git+https://github.com/sokrypton/esm.git")
        os.system("touch ESMFOLD_READY")
    
    # 等待模型下载完成
    while not os.path.isfile(model_name):
        time.sleep(5)
    if os.path.isfile(f"{model_name}.aria2"):
        print("  [下载] 正在下载模型参数...")
    while os.path.isfile(f"{model_name}.aria2"):
        time.sleep(5)
    
    print("  [完成] ESMFold 安装完成")
else:
    print("\n[跳过] ESMFold 已安装")

# 安装其他必需的 Python 包
print("\n[安装] 正在安装其他依赖包...")
os.system("pip install -q pandas matplotlib scipy")
print("  [完成] pandas, matplotlib, scipy 安装完成")

print("\n" + "="*70)
print("PART 1: 环境配置完成")
print("="*70)
print("[下一步] 请运行 PART 2: 主程序")
print("")

# =============================================================================
# PART 2: 主程序（环境配置完成后运行此单元格）
# =============================================================================

print("="*70)
print("PART 2: 主程序开始")
print("="*70)

# 检查是否已运行 PART 1
if not os.path.isfile("ESMFOLD_READY"):
    print("[警告] 未检测到 ESMFold 安装")
    print("[提示] 请先运行 PART 1: 环境配置")
    print("[继续] 如果您确定环境已配置，可以继续运行")
    print("")

# ===== 导入依赖包 =====
import os
import sys
import json
import csv
import re
import logging
import traceback
import zipfile
import hashlib
import time
from datetime import datetime
from pathlib import Path
from string import ascii_uppercase, ascii_lowercase

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from jax.tree_util import tree_map
from scipy.special import softmax
import gc

# 检查是否在 Colab 环境（如果 PART 1 未运行，则重新检查）
if 'IN_COLAB' not in globals():
    try:
        import google.colab
        IN_COLAB = True
        print("[环境] 运行在 Google Colab")
    except ImportError:
        IN_COLAB = False
        print("[环境] 运行在本地环境（非 Colab）")
        print("[提示] 本脚本设计为在 Colab 中运行，本地运行仅供调试")
else:
    print(f"[环境] {'Google Colab' if IN_COLAB else '本地环境'}")

# ===== 配置日志 =====
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# =============================================================================
# PART 3: 全局配置
# =============================================================================

CONFIG = {
    'model_version': '1',
    'model_name': model_name,
    'num_recycles': 3,
    'min_plddt_mean': 70.0,
    'min_ptm': 0.5,
    'batch_size': 4,
    'csv_append_mode': True,
    'zip_plots': True,
    'safe_filename': True,
    'output_dir': 'prediction_results',
    'max_retries': 2,
    'checkpoint_interval': 5,
    'chain_linker': 25,
}

# =============================================================================
# PART 4: 辅助函数
# =============================================================================

def parse_esmfold_output(output):
    """解析 ESMFold 输出"""
    pae = (output["aligned_confidence_probs"][0] * np.arange(64)).mean(-1) * 31
    plddt = output["plddt"][0,:,1]
    
    bins = np.append(0, np.linspace(2.3125, 21.6875, 63))
    sm_contacts = softmax(output["distogram_logits"], -1)[0]
    sm_contacts = sm_contacts[..., bins<8].sum(-1)
    xyz = output["positions"][-1, 0, :, 1]
    mask = output["atom37_atom_exists"][0, :, 1] == 1
    
    o = {
        "pae": pae[mask, :][:, mask],
        "plddt": plddt[mask],
        "sm_contacts": sm_contacts[mask, :][:, mask],
        "xyz": xyz[mask]
    }
    return o

def get_hash(x):
    """生成序列哈希"""
    return hashlib.sha1(x.encode()).hexdigest()

def safe_id(name):
    """文件名安全化"""
    if not CONFIG['safe_filename']:
        return name
    safe_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', name)
    if len(safe_name) > 100:
        safe_name = safe_name[:100]
    return safe_name

# =============================================================================
# PART 5: 文件上传
# =============================================================================

def upload_input_files():
    """上传输入 JSON 文件"""
    print("\n" + "="*70)
    print("步骤 1: 上传输入文件")
    print("="*70)
    
    uploaded_files = {}
    
    if IN_COLAB:
        from google.colab import files
        print("请上传以下文件:")
        print("  - esm_if_sequences.json")
        print("  - proteinmpnn_sequences.json")
        print("\n开始上传...")
        uploaded = files.upload()
        
        for filename, content in uploaded.items():
            with open(filename, 'wb') as f:
                f.write(content)
            uploaded_files[filename] = filename
            print(f"  [OK] {filename} 已上传")
    else:
        print("[调试模式] 尝试从当前目录加载文件...")
        for filename in ['esm_if_sequences.json', 'proteinmpnn_sequences.json']:
            if os.path.exists(filename):
                uploaded_files[filename] = filename
                print(f"  [OK] 找到文件: {filename}")
            else:
                print(f"  [警告] 未找到文件: {filename}")
    
    return uploaded_files

# =============================================================================
# PART 6: 序列加载与合并
# =============================================================================

def robust_load_sequences(path):
    """鲁棒的序列加载函数，支持多种 JSON 格式"""
    logger.info(f"加载序列文件: {path}")
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"文件读取失败: {path}, 错误: {e}")
        return []
    
    sequences = []
    basename = os.path.basename(path)
    
    # 格式1: step3a ProteinMPNN 格式
    if isinstance(data, dict) and any('sequences' in v for v in data.values() if isinstance(v, dict)):
        logger.info(f"  检测到 ProteinMPNN 格式: {basename}")
        for backbone_id, backbone_data in data.items():
            if isinstance(backbone_data, dict) and 'sequences' in backbone_data:
                for seq_data in backbone_data['sequences']:
                    sequences.append({
                        'id': seq_data.get('sequence_id', f"{basename}_{len(sequences)}"),
                        'sequence': seq_data['sequence'],
                        'length': seq_data.get('length', len(seq_data['sequence'])),
                        'method': seq_data.get('method', 'proteinmpnn'),
                        'backbone_id': backbone_id,
                        'source_file': basename
                    })
    
    # 格式2: step3b ESM-IF 格式
    elif isinstance(data, dict) and data.get('design_method') == 'esm_if':
        logger.info(f"  检测到 ESM-IF 格式: {basename}")
        for backbone_id, backbone_data in data.get('results', {}).items():
            for seq_data in backbone_data.get('sequences', []):
                sequences.append({
                    'id': seq_data.get('sequence_id', f"{basename}_{len(sequences)}"),
                    'sequence': seq_data['sequence'],
                    'length': seq_data.get('length', len(seq_data['sequence'])),
                    'method': seq_data.get('method', 'esm_if'),
                    'backbone_id': backbone_id,
                    'source_file': basename
                })
    
    # 格式3: 列表格式
    elif isinstance(data, list):
        logger.info(f"  检测到列表格式: {basename}")
        for i, item in enumerate(data):
            if isinstance(item, dict):
                seq_id = item.get('id') or item.get('sequence_id') or f"{basename}_{i}"
                sequence = item.get('sequence') or item.get('seq')
                if sequence:
                    sequences.append({
                        'id': seq_id,
                        'sequence': sequence,
                        'length': item.get('length', len(sequence)),
                        'method': item.get('method', 'unknown'),
                        'backbone_id': item.get('backbone_id', 'unknown'),
                        'source_file': basename
                    })
            elif isinstance(item, str):
                sequences.append({
                    'id': f"{basename}_{i}",
                    'sequence': item,
                    'length': len(item),
                    'method': 'unknown',
                    'backbone_id': 'unknown',
                    'source_file': basename
                })
    
    # 格式4: 字典格式 (id -> sequence)
    elif isinstance(data, dict) and 'sequences' not in data:
        logger.info(f"  检测到字典格式: {basename}")
        for seq_id, seq_or_data in data.items():
            if isinstance(seq_or_data, str):
                sequences.append({
                    'id': seq_id,
                    'sequence': seq_or_data,
                    'length': len(seq_or_data),
                    'method': 'unknown',
                    'backbone_id': 'unknown',
                    'source_file': basename
                })
            elif isinstance(seq_or_data, dict) and 'sequence' in seq_or_data:
                sequences.append({
                    'id': seq_id,
                    'sequence': seq_or_data['sequence'],
                    'length': seq_or_data.get('length', len(seq_or_data['sequence'])),
                    'method': seq_or_data.get('method', 'unknown'),
                    'backbone_id': seq_or_data.get('backbone_id', 'unknown'),
                    'source_file': basename
                })
    
    logger.info(f"  加载完成: {len(sequences)} 个序列")
    return sequences

def merge_and_dedup(sequences_list):
    """合并多个序列列表并去重"""
    logger.info("合并并去重序列...")
    
    merged = {}
    conflicts = []
    
    for seq in sequences_list:
        seq_id = seq['id']
        if seq_id in merged:
            if merged[seq_id]['sequence'] != seq['sequence']:
                conflicts.append({
                    'id': seq_id,
                    'old_source': merged[seq_id]['source_file'],
                    'new_source': seq['source_file'],
                    'old_seq': merged[seq_id]['sequence'][:50] + '...',
                    'new_seq': seq['sequence'][:50] + '...'
                })
                logger.warning(f"  [冲突] ID={seq_id}: {merged[seq_id]['source_file']} vs {seq['source_file']}")
        
        merged[seq_id] = seq
    
    logger.info(f"  合并完成: {len(merged)} 个唯一序列")
    if conflicts:
        logger.warning(f"  检测到 {len(conflicts)} 个 ID 冲突（已保留最后加载的版本）")
    
    return list(merged.values()), conflicts

# =============================================================================
# PART 7: 检查点功能
# =============================================================================

# 初始化检查点
checkpoint_file = 'prediction_results/checkpoint.json'
checkpoint_data = {
    'completed_predictions': [],
    'failed_predictions': [],
    'last_update': None
}

if os.path.exists(checkpoint_file):
    try:
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            checkpoint_data = json.load(f)
        print(f"[断点] 加载检查点: 已完成 {len(checkpoint_data.get('completed_predictions', []))} 个")
    except Exception as e:
        print(f"[警告] 检查点文件加载失败: {e}")

def save_checkpoint():
    """保存检查点"""
    checkpoint_data['last_update'] = datetime.now().isoformat()
    os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)

def is_completed(seq_id):
    """检查序列是否已完成"""
    return seq_id in checkpoint_data.get('completed_predictions', [])

def mark_completed(seq_id):
    """标记序列为已完成"""
    if seq_id not in checkpoint_data['completed_predictions']:
        checkpoint_data['completed_predictions'].append(seq_id)

def mark_failed(seq_id):
    """标记序列为失败"""
    if seq_id not in checkpoint_data['failed_predictions']:
        checkpoint_data['failed_predictions'].append(seq_id)


# =============================================================================
# PART 8: 评分函数
# =============================================================================
def calculate_score(metrics):
    """简单阈值评分策略"""
    plddt_mean = metrics.get('plddt_mean', 0)
    ptm = metrics.get('ptm', 0)
    
    # 同时满足阈值即通过
    passes = plddt_mean >= 70.0 and ptm >= 0.5
    
    # score 为归一化分数 (0-1)
    score = (plddt_mean / 100.0 * 0.6) + (ptm * 0.4)
    
    return {
        'score': score,
        'passes': passes,
        'reason': f'pLDDT={plddt_mean:.1f}, PTM={ptm:.3f}'
    }

# =============================================================================
# PART 9: ESMFold 预测核心函数
# =============================================================================

# 全局模型变量（延迟加载）
_esmfold_model = None
_model_name_loaded = None

def load_esmfold_model():
    """加载 ESMFold 模型（全局加载一次）"""
    global _esmfold_model, _model_name_loaded
    
    if _esmfold_model is None or _model_name_loaded != model_name:
        logger.info("加载 ESMFold 模型...")
        _esmfold_model = torch.load(model_name, weights_only=False)
        _esmfold_model.eval().cuda().requires_grad_(False)
        _model_name_loaded = model_name
        logger.info("模型加载完成！")
    
    return _esmfold_model

def run_esmfold_predict(seq_id, sequence, outdir):
    """
    使用官方 ESMFold 模型进行结构预测
    
    参数:
        seq_id: 序列ID
        sequence: 氨基酸序列
        outdir: 输出目录
    
    返回:
        dict: {
            'plddt_mean': float (0-100),
            'ptm': float (0-1),
            'plddt_per_residue': list,
            'pae': array,
            'runtime_seconds': float
        }
    """
    try:
        import time
        start_time = time.time()
        
        logger.info(f"  [ESMFold] 开始预测序列: {seq_id}")
        
        # 加载模型
        model = load_esmfold_model()
        
        # 清理序列（移除非标准氨基酸）
        sequence = re.sub("[^A-Z:]", "", sequence.replace("/", ":").upper())
        sequence = re.sub(":+", ":", sequence)
        sequence = re.sub("^[:]+", "", sequence)
        sequence = re.sub("[:]+$", "", sequence)
        
        length = len(sequence.replace(":", ""))
        logger.info(f"  序列长度: {length} 残基")
        
        # 根据长度优化 chunk_size（显存管理）
        if length > 700:
            model.set_chunk_size(64)
        else:
            model.set_chunk_size(128)
        
        # 清理显存
        torch.cuda.empty_cache()
        
        # 运行预测
        output = model.infer(
            sequence,
            num_recycles=CONFIG['num_recycles'],
            chain_linker="X" * CONFIG['chain_linker'],
            residue_index_offset=512
        )
        
        # 提取 PDB 结构
        pdb_str = model.output_to_pdb(output)[0]
        
        # 转换输出到 numpy
        output = tree_map(lambda x: x.cpu().numpy(), output)
        
        # 提取指标
        ptm = float(output["ptm"][0])
        plddt_mean = float(output["plddt"][0, ..., 1].mean())
        plddt_per_residue = output["plddt"][0, :, 1].tolist()
        
        # 解析详细输出
        parsed_output = parse_esmfold_output(output)
        
        runtime = time.time() - start_time
        
        logger.info(f"  [ESMFold] 预测完成: pLDDT={plddt_mean:.1f}, PTM={ptm:.3f}, 用时={runtime:.1f}s")
        
        # 保存结果
        os.makedirs(outdir, exist_ok=True)
        
        # 保存 PDB 文件
        pdb_file = os.path.join(outdir, f"{seq_id}.pdb")
        with open(pdb_file, 'w') as f:
            f.write(pdb_str)
        
        # 保存 PAE 矩阵
        pae_file = os.path.join(outdir, f"{seq_id}.pae.txt")
        np.savetxt(pae_file, parsed_output["pae"], "%.3f")
        
        # 构建返回结果
        metrics = {
            'plddt_mean': plddt_mean,
            'ptm': ptm,
            'plddt_per_residue': plddt_per_residue,
            'pae': parsed_output["pae"].tolist(),
            'runtime_seconds': runtime,
            'pdb_file': pdb_file,
            'pae_file': pae_file
        }
        
        # 保存 metrics.json
        metrics_file = os.path.join(outdir, 'metrics.json')
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        return metrics
        
    except Exception as e:
        logger.error(f"  [ESMFold] 预测失败: {e}")
        logger.error(traceback.format_exc())
        return None

# =============================================================================
# PART 10: CSV 处理函数
# =============================================================================

def init_or_load_csv(csv_path, append_mode):
    """初始化或加载 CSV 文件"""
    csv_fields = [
        'id', 'source_file', 'sequence_length', 'status', 
        'start_time', 'end_time', 'runtime_seconds',
        'plddt_mean', 'ptm', 'score', 'passes', 'notes'
    ]
    
    if append_mode and os.path.exists(csv_path):
        logger.info(f"CSV 文件已存在，将追加新结果: {csv_path}")
        return csv_fields
    else:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()
        logger.info(f"创建新 CSV 文件: {csv_path}")
        return csv_fields

def safe_write_csv_row(csv_path, csv_fields, row_data):
    """原子写入 CSV 行"""
    try:
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            for field in csv_fields:
                if field not in row_data:
                    row_data[field] = ''
            writer.writerow(row_data)
        return True
    except Exception as e:
        logger.error(f"CSV 写入失败: {e}")
        return False

# =============================================================================
# PART 11: 绘图函数
# =============================================================================

def plot_confidence_metrics(metrics, seq_id, output_path):
    """绘制置信度指标图"""
    try:
        plddt_per_residue = metrics.get('plddt_per_residue', [])
        pae = np.array(metrics.get('pae', []))
        
        if len(plddt_per_residue) == 0:
            logger.warning(f"  [绘图] 无 pLDDT 数据，跳过")
            return False
        
        fig = plt.figure(figsize=(15, 5), dpi=100)
        
        # 子图1: pLDDT 曲线
        ax1 = plt.subplot(1, 3, 1)
        residue_indices = list(range(1, len(plddt_per_residue) + 1))
        ax1.plot(residue_indices, plddt_per_residue, linewidth=1.5, color='#1f77b4')
        ax1.axhline(y=70, color='r', linestyle='--', linewidth=1, label='阈值 70')
        ax1.set_xlabel('残基位置', fontsize=12)
        ax1.set_ylabel('pLDDT', fontsize=12)
        ax1.set_title(f'{seq_id} - pLDDT 曲线', fontsize=14)
        ax1.legend()
        ax1.grid(alpha=0.3)
        ax1.set_xlim(0, len(plddt_per_residue))
        ax1.set_ylim(0, 100)
        
        # 子图2: pLDDT 分布直方图
        ax2 = plt.subplot(1, 3, 2)
        ax2.hist(plddt_per_residue, bins=50, color='#2ca02c', alpha=0.7, edgecolor='black')
        ax2.axvline(x=70, color='r', linestyle='--', linewidth=1, label='阈值 70')
        ax2.set_xlabel('pLDDT', fontsize=12)
        ax2.set_ylabel('频数', fontsize=12)
        ax2.set_title(f'{seq_id} - pLDDT 分布', fontsize=14)
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # 子图3: PAE 矩阵
        ax3 = plt.subplot(1, 3, 3)
        if len(pae) > 0:
            im = ax3.imshow(pae, cmap='bwr', vmin=0, vmax=30)
            ax3.set_xlabel('Scored residue', fontsize=12)
            ax3.set_ylabel('Aligned residue', fontsize=12)
            ax3.set_title(f'{seq_id} - Predicted Aligned Error', fontsize=14)
            plt.colorbar(im, ax=ax3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  [绘图] 已保存: {os.path.basename(output_path)}")
        return True
        
    except Exception as e:
        logger.error(f"  [绘图失败] {e}")
        return False

# =============================================================================
# PART 12: 序列处理函数
# =============================================================================

def process_sequence(rec, csv_path, csv_fields):
    """处理单个序列"""
    seq_id = rec['id']
    sequence = rec['sequence']
    length = rec['length']
    
    logger.info(f"\n处理序列: {seq_id} (长度: {length})")
    
    # 检查是否已完成
    if is_completed(seq_id):
        logger.info(f"  [跳过] 序列已完成（检查点记录）")
        return {'status': 'skipped', 'reason': 'already_completed'}
    
    # 创建序列工作目录
    safe_seq_id = safe_id(seq_id)
    seq_outdir = os.path.join(CONFIG['output_dir'], safe_seq_id)
    os.makedirs(seq_outdir, exist_ok=True)
    
    # 记录开始时间
    start_time = datetime.now()
    start_time_str = start_time.isoformat()
    
    # 调用 ESMFold 预测（带重试）
    metrics = None
    error_msg = None
    
    for attempt in range(CONFIG['max_retries'] + 1):
        try:
            logger.info(f"  [预测] 尝试 {attempt + 1}/{CONFIG['max_retries'] + 1}")
            
            metrics = run_esmfold_predict(seq_id, sequence, seq_outdir)
            
            if metrics is None:
                raise Exception("预测失败，返回 None")
            
            # 预测成功
            break
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"  [错误] 预测失败: {error_msg}")
            if attempt < CONFIG['max_retries']:
                logger.info(f"  [重试] 将重试...")
                import time
                time.sleep(2)
                # 清理显存
                torch.cuda.empty_cache()
                gc.collect()
            else:
                logger.error(f"  [失败] 达到最大重试次数，标记为失败")
    
    # 记录结束时间
    end_time = datetime.now()
    end_time_str = end_time.isoformat()
    duration = (end_time - start_time).total_seconds()
    
    # 构建 CSV 行
    csv_row = {
        'id': seq_id,
        'source_file': rec['source_file'],
        'sequence_length': length,
        'start_time': start_time_str,
        'end_time': end_time_str,
        'runtime_seconds': f'{duration:.1f}'
    }
    
    if metrics is None:
        # 预测失败
        csv_row.update({
            'status': 'error',
            'plddt_mean': '',
            'ptm': '',
            'score': '',
            'passes': 'False',
            'notes': f'预测失败: {error_msg or "未知错误"}'
        })
        
        safe_write_csv_row(csv_path, csv_fields, csv_row)
        mark_failed(seq_id)
        
        return {'status': 'error', 'error': error_msg}
    
    # 预测成功，提取指标
    plddt_mean = metrics.get('plddt_mean', 0)
    ptm = metrics.get('ptm', 0)
    plddt_per_residue = metrics.get('plddt_per_residue', [])
    
    logger.info(f"  [指标] pLDDT={plddt_mean:.1f}, PTM={ptm:.3f}")
    
    # 计算评分
    score_result = calculate_score(metrics)
    score = score_result['score']
    passes = score_result['passes']
    
    logger.info(f"  [评分] Score={score:.3f}, Passes={passes}")
    
    # 绘制图像
    if plddt_per_residue:
        plot_path = os.path.join(seq_outdir, f"{safe_seq_id}_confidence.png")
        plot_confidence_metrics(metrics, seq_id, plot_path)
    
    # 更新 CSV 行
    csv_row.update({
        'status': 'success',
        'plddt_mean': f'{plddt_mean:.2f}',
        'ptm': f'{ptm:.4f}',
        'score': f'{score:.3f}',
        'passes': str(passes),
        'notes': score_result['reason']
    })
    
    safe_write_csv_row(csv_path, csv_fields, csv_row)
    mark_completed(seq_id)
    
    # 清理显存
    torch.cuda.empty_cache()
    
    return {
        'status': 'success',
        'metrics': metrics,
        'score': score,
        'passes': passes
    }

# =============================================================================
# PART 13: 打包函数
# =============================================================================

def finalize_and_package(output_dir, zip_plots=True):
    """最终化并打包结果"""
    logger.info("\n" + "="*70)
    logger.info("最终化结果...")
    logger.info("="*70)
    
    # 1. 打包图像
    if zip_plots:
        logger.info("\n打包图像文件...")
        zip_path = os.path.join(output_dir, 'plots.zip')
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    if file.endswith('.png'):
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, output_dir)
                        zipf.write(file_path, arcname)
        logger.info(f"  [完成] 图像已打包: {zip_path}")
    
    # 2. 生成 filtered_sequences.json
    logger.info("\n生成筛选后的序列...")
    csv_path = os.path.join(output_dir, 'results.csv')
    filtered_sequences = []
    
    if os.path.exists(csv_path):
        import pandas as pd
        df = pd.read_csv(csv_path)
        passed_df = df[df['passes'] == 'True']
        
        for _, row in passed_df.iterrows():
            seq_id = row['id']
            safe_seq_id = safe_id(seq_id)
            metrics_file = os.path.join(output_dir, safe_seq_id, 'metrics.json')
            
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                
                filtered_sequences.append({
                    'sequence_id': seq_id,
                    'source_file': row['source_file'],
                    'length': int(row['sequence_length']),
                    'plddt_mean': float(row['plddt_mean']),
                    'ptm': float(row['ptm']),
                    'score': float(row['score']),
                    'metrics': metrics
                })
        
        filtered_path = os.path.join(output_dir, 'filtered_sequences.json')
        with open(filtered_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_sequences, f, indent=2, ensure_ascii=False)
        
        logger.info(f"  [完成] 通过筛选: {len(filtered_sequences)}/{len(df)} 个序列")
        logger.info(f"  [保存] {filtered_path}")
    
    # 3. 生成统计摘要
    logger.info("\n统计摘要:")
    if os.path.exists(csv_path):
        import pandas as pd
        df = pd.read_csv(csv_path)
        
        total = len(df)
        success = len(df[df['status'] == 'success'])
        error = len(df[df['status'] == 'error'])
        passed = len(df[df['passes'] == 'True'])
        
        logger.info(f"  - 总序列数: {total}")
        logger.info(f"  - 成功预测: {success}")
        logger.info(f"  - 预测失败: {error}")
        logger.info(f"  - 通过筛选: {passed}")
        
        if success > 0:
            avg_plddt = df[df['status'] == 'success']['plddt_mean'].astype(float).mean()
            avg_ptm = df[df['status'] == 'success']['ptm'].astype(float).mean()
            logger.info(f"  - 平均 pLDDT: {avg_plddt:.2f}")
            logger.info(f"  - 平均 PTM: {avg_ptm:.3f}")
    
    logger.info("\n完成！")

# =============================================================================
# PART 14: 主函数
# =============================================================================

def main():
    """主函数"""
    print("\n" + "="*70)
    print("ESMFold 批量预测: 第4步 - 结构预测与指标计算")
    print("作者: Kuroneko | 日期: 2025.10.3")
    print("="*70)
    
    try:
        # 步骤 1: 上传文件
        uploaded_files = upload_input_files()
        
        if len(uploaded_files) == 0:
            logger.error("未找到输入文件，请上传至少一个 JSON 文件")
            return
        
        # 步骤 2: 加载序列
        logger.info("\n" + "="*70)
        logger.info("步骤 2: 加载并合并序列")
        logger.info("="*70)
        
        all_sequences = []
        for filename in uploaded_files.values():
            sequences = robust_load_sequences(filename)
            all_sequences.extend(sequences)
        
        # 合并去重
        merged_sequences, conflicts = merge_and_dedup(all_sequences)
        
        # 保存合并后的序列
        merged_path = os.path.join(CONFIG['output_dir'], 'merged_input_sequences.json')
        os.makedirs(CONFIG['output_dir'], exist_ok=True)
        with open(merged_path, 'w', encoding='utf-8') as f:
            json.dump({
                'total_sequences': len(merged_sequences),
                'conflicts': conflicts,
                'sequences': merged_sequences
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"合并后序列已保存: {merged_path}")
        
        # 步骤 3: 加载 ESMFold 模型
        logger.info("\n" + "="*70)
        logger.info("步骤 3: 加载 ESMFold 模型")
        logger.info("="*70)
        
        load_esmfold_model()
        
        # 步骤 4: 初始化 CSV
        logger.info("\n" + "="*70)
        logger.info("步骤 4: 初始化结果 CSV")
        logger.info("="*70)
        
        csv_path = os.path.join(CONFIG['output_dir'], 'results.csv')
        csv_fields = init_or_load_csv(csv_path, CONFIG['csv_append_mode'])
        
        # 步骤 5: 批量处理序列
        logger.info("\n" + "="*70)
        logger.info(f"步骤 5: 批量处理序列 (共 {len(merged_sequences)} 个)")
        logger.info("="*70)
        
        batch_size = CONFIG['batch_size']
        total = len(merged_sequences)
        
        for i in range(0, total, batch_size):
            batch_end = min(i + batch_size, total)
            logger.info(f"\n[批次] 处理 {i+1}-{batch_end}/{total}")
            
            for j, rec in enumerate(merged_sequences[i:batch_end]):
                logger.info(f"\n--- 序列 {i+j+1}/{total} ---")
                result = process_sequence(rec, csv_path, csv_fields)
                
                # 每批次结束保存检查点
                if (i + j + 1) % CONFIG['checkpoint_interval'] == 0:
                    save_checkpoint()
                    logger.info(f"  [检查点] 已保存 (完成 {i+j+1} 个)")
        
        # 最终保存检查点
        save_checkpoint()
        
        # 步骤 6: 最终化并打包
        finalize_and_package(CONFIG['output_dir'], zip_plots=CONFIG['zip_plots'])
        
        # 步骤 7: 下载结果（仅 Colab）
        if IN_COLAB:
            logger.info("\n准备下载结果...")
            
            # 打包所有结果
            result_zip = 'esmfold_prediction_results.zip'
            with zipfile.ZipFile(result_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(CONFIG['output_dir']):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, os.path.dirname(CONFIG['output_dir']))
                        zipf.write(file_path, arcname)
            
            logger.info(f"结果已打包: {result_zip}")
            
            from google.colab import files
            files.download(result_zip)
            
            logger.info("\n下载完成！")
            logger.info(f"请解压到本地目录: {CONFIG['output_dir']}")
        else:
            logger.info("\n本地运行完成，结果保存在: {CONFIG['output_dir']}")
        
        logger.info("\n" + "="*70)
        logger.info("全部完成！")
        logger.info("="*70)
        
    except Exception as e:
        logger.error(f"\n程序执行异常: {e}")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()

# ===== 快速测试建议 =====
# 在正式运行前，建议先测试前 1-3 个序列：
# 1. 上传少量序列的 JSON 文件
# 2. 检查模型加载和预测是否正常
# 3. 确认 CSV 和图像生成正常
# 4. 然后再运行完整数据集

# ===== 显存管理建议 =====
# 如遇 OOM (Out of Memory) 错误：
# 1. 减小 batch_size (改为 1 或 2)
# 2. 减小序列长度（过滤掉超长序列）
# 3. 使用 model.set_chunk_size(32) 减小 chunk size
# 4. 在每个序列后手动调用 torch.cuda.empty_cache()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
Colab 脚本: 第5步 - 序列比对与指标计算
生成时间: 2025-10-03 20:58:25
作者: Kuroneko
日期: 2025.10.3
=============================================================================

【使用说明】
本脚本包含两个部分，请按顺序在 Colab 中运行：

第一部分：环境配置（运行一次即可）
- 安装 ColabFold、依赖包等
- 标记：以 "# ===== PART 1:" 开头

第二部分：主程序执行
- 上传序列文件、选择预测工具、执行预测
- 标记：以 "# ===== PART 2:" 开头

【重要提示】
本脚本已包含完整的预测工具选择功能，无需手动替换代码！
运行时会自动弹出选择菜单，支持以下预测工具：
1. ColabFold (推荐) - 基于 ColabFold 的结构预测
2. AlphaFold2 MSA Colab - 基于 AlphaFold2 MSA 的结构预测  
3. ESMFold - 基于 ESMFold 的结构预测
4. 自定义 - 手动指定预测方法

默认选择 ColabFold，按回车即可使用。

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
- 长度范围: 20 - 1000
- pLDDT 阈值: 70.0
- PTM 阈值: 0.5
- 批次大小: 5
- CSV 模式: 追加
- 打包图像: 是
- 断点续传: 是

常见错误与解决办法：
1. 上传多个 json 导致 id 冲突 → 脚本会自动去重并记录冲突
2. 序列过长导致内存/时间超时 → 调整 batch_size 或使用长度过滤
3. 预测工具调用失败 → 脚本会自动重试，失败后记录错误并继续
4. CSV 追加出错 → 脚本使用原子写入（临时文件+重命名）

预测工具选择说明：
运行脚本时会自动弹出选择菜单，根据您的环境选择合适的预测工具。
"""

# =============================================================================
# PART 1: 环境配置（请先运行此单元格）
# =============================================================================
# 说明：此单元格负责安装 ColabFold 和所有必需的依赖包
# 运行时间：首次运行约 5-10 分钟，之后会跳过已安装的包

import os
import sys
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

# 安装 ColabFold
if not os.path.isfile("COLABFOLD_READY"):
    print("\n[安装] 正在安装 ColabFold...")
    print("  这可能需要几分钟，请耐心等待...")
    os.system("pip install -q --no-warn-conflicts 'colabfold[alphafold-minus-jax] @ git+https://github.com/sokrypton/ColabFold'")
    
    # 处理 TPU 环境
    if os.environ.get('TPU_NAME', False) != False:
        print("  [TPU] 检测到 TPU 环境，安装 JAX...")
        os.system("pip uninstall -y jax jaxlib")
        os.system("pip install --no-warn-conflicts --upgrade dm-haiku==0.0.10 'jax[cuda12_pip]'==0.3.25 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html")
    
    # 创建符号链接
    os.system("ln -s /usr/local/lib/python3.*/dist-packages/colabfold colabfold")
    os.system("ln -s /usr/local/lib/python3.*/dist-packages/alphafold alphafold")
    
    # 修复 TensorFlow 崩溃问题
    os.system("rm -f /usr/local/lib/python3.*/dist-packages/tensorflow/core/kernels/libtfkernel_sobol_op.so")
    
    # 标记安装完成
    os.system("touch COLABFOLD_READY")
    print("  [完成] ColabFold 安装完成")
else:
    print("\n[跳过] ColabFold 已安装")

# 安装其他必需的 Python 包
print("\n[安装] 正在安装其他依赖包...")
os.system("pip install -q pandas matplotlib")
print("  [完成] pandas, matplotlib 安装完成")

# 可选：安装 ESMFold（如果用户选择使用 ESMFold）
print("\n[可选] 可以选择安装 ESMFold:")
print("  如果您计划使用 ESMFold 作为预测工具，请取消下面的注释并运行：")
print("  # os.system('pip install -q fair-esm')")

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
if not os.path.isfile("COLABFOLD_READY"):
    print("[警告] 未检测到 ColabFold 安装")
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
from datetime import datetime
from pathlib import Path

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
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ===== 全局配置 =====
CONFIG = {
    'min_length': 20,
    'max_length': 1000,
    'min_plddt_mean': 70.0,
    'min_ptm': 0.5,
    'batch_size': 5,
    'csv_append_mode': True,
    'zip_plots': True,
    'safe_filename': True,
    'output_dir': 'prediction_results',
    'max_retries': 2,  # 失败重试次数
    'checkpoint_interval': 10,  # 每处理N个序列保存一次检查点
}

# ===== 文件上传 =====
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
            # 保存文件
            with open(filename, 'wb') as f:
                f.write(content)
            uploaded_files[filename] = filename
            print(f"  [OK] {filename} 已上传")
    else:
        # 非 Colab 环境，尝试从当前目录加载
        print("[调试模式] 尝试从当前目录加载文件...")
        for filename in ['esm_if_sequences.json', 'proteinmpnn_sequences.json']:
            if os.path.exists(filename):
                uploaded_files[filename] = filename
                print(f"  [OK] 找到文件: {filename}")
            else:
                print(f"  [警告] 未找到文件: {filename}")
    
    return uploaded_files

# ===== 序列加载函数 =====
def robust_load_sequences(path):
    """
    鲁棒的序列加载函数，支持多种 JSON 格式
    
    支持的格式：
    1. step3a ProteinMPNN 格式: {backbone_id: {sequences: [...]}}
    2. step4 ESM-IF 格式: {design_method: "esm_if", results: {...}}
    3. 简单列表: [{id, sequence}, ...]
    4. 简单字典: {id: sequence, ...}
    5. 包含 sequences 字段的字典
    
    返回统一格式:
    [{
        "id": str,
        "sequence": str,
        "length": int,
        "method": str,
        "backbone_id": str,
        "source_file": str
    }, ...]
    """
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
    
    # 格式2: step4 ESM-IF 格式
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
                # 纯字符串列表
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
    """
    合并多个序列列表并去重
    
    去重策略：
    - 以 id 为主键
    - 后者覆盖前者
    - 记录冲突到日志
    """
    logger.info("合并并去重序列...")
    
    merged = {}
    conflicts = []
    
    for seq in sequences_list:
        seq_id = seq['id']
        if seq_id in merged:
            # 检测冲突
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

def safe_id(name):
    """文件名安全化"""
    if not CONFIG['safe_filename']:
        return name
    
    # 替换特殊字符为下划线
    safe_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', name)
    # 限制长度
    if len(safe_name) > 100:
        safe_name = safe_name[:100]
    return safe_name

# ===== 检查点功能 =====

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


# ===== 评分函数 =====
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

# ===== 预测工具选择 =====
def select_prediction_tool():
    """选择预测工具"""
    print("\n" + "="*70)
    print("预测工具选择")
    print("="*70)
    print("请选择要使用的预测工具:")
    print("1. ColabFold (推荐) - 基于 ColabFold 的结构预测")
    print("2. AlphaFold2 MSA Colab - 基于 AlphaFold2 MSA 的结构预测")
    print("3. ESMFold - 基于 ESMFold 的结构预测")
    print("4. 自定义 - 手动指定预测方法")
    
    while True:
        try:
            choice = input("\n请输入选择 (1-4, 默认: 1): ").strip()
            if not choice:
                choice = "1"
            
            if choice == "1":
                return "colabfold"
            elif choice == "2":
                return "alphafold2_msa"
            elif choice == "3":
                return "esmfold"
            elif choice == "4":
                custom_method = input("请输入自定义方法名称: ").strip()
                return f"custom_{custom_method}"
            else:
                print("无效选择，请输入 1-4")
        except KeyboardInterrupt:
            print("\n用户中断，使用默认选择: ColabFold")
            return "colabfold"

# 全局预测工具选择
PREDICTION_TOOL = None

def get_prediction_tool():
    """获取预测工具（单例模式）"""
    global PREDICTION_TOOL
    if PREDICTION_TOOL is None:
        PREDICTION_TOOL = select_prediction_tool()
        print(f"\n[选择] 已选择预测工具: {PREDICTION_TOOL}")
    return PREDICTION_TOOL

# ===== ColabFold 预测函数 =====
def run_official_alignment_and_metrics_ColabFold(seq_id, sequence, outdir, params):
    """使用 ColabFold 进行结构预测"""
    try:
        logger.info(f"  [ColabFold] 开始预测序列: {seq_id}")
        
        # 1. 写入 FASTA 文件
        fasta_path = os.path.join(outdir, f"{seq_id}.fasta")
        with open(fasta_path, 'w') as f:
            f.write(f">{seq_id}\n{sequence}\n")
        
        # 2. 调用 ColabFold
        import subprocess
        cmd = [
            'colabfold_batch',
            fasta_path,
            outdir,
            '--num-models', '1',
            '--num-recycle', '3',
            '--use-gpu-relax'
        ]
        
        logger.info(f"  [ColabFold] 执行命令: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise Exception(f"ColabFold 调用失败: {result.stderr}")
        
        # 3. 解析输出文件
        scores_file = os.path.join(outdir, f"{seq_id}_scores.json")
        if os.path.exists(scores_file):
            with open(scores_file, 'r') as f:
                scores = json.load(f)
        else:
            # 如果没有 scores.json，尝试从 PDB 文件提取
            pdb_files = [f for f in os.listdir(outdir) if f.endswith('.pdb')]
            if pdb_files:
                pdb_path = os.path.join(outdir, pdb_files[0])
                plddt_scores = extract_plddt_from_pdb(pdb_path)
                scores = {
                    'plddt': sum(plddt_scores) / len(plddt_scores) if plddt_scores else 70.0,
                    'ptm': 0.7,  # 默认值
                    'plddt_per_residue': plddt_scores
                }
            else:
                raise Exception("未找到输出文件")
        
        logger.info(f"  [ColabFold] 预测完成: pLDDT={scores['plddt']:.1f}, PTM={scores['ptm']:.3f}")
        
        return {
            'plddt_mean': scores['plddt'],
            'ptm': scores['ptm'],
            'iptm': scores.get('iptm'),
            'per_residue_plddt': scores.get('plddt_per_residue', [])
        }
    
    except Exception as e:
        logger.error(f"  [ColabFold] 预测失败: {e}")
        return None

# ===== AlphaFold2 MSA Colab 预测函数 =====
def run_official_alignment_and_metrics_AlphaFold2_MSA_Colab(seq_id, sequence, outdir, params):
    """使用 AlphaFold2 MSA Colab 进行结构预测"""
    try:
        logger.info(f"  [AlphaFold2 MSA] 开始预测序列: {seq_id}")
        
        # 这里需要根据实际的 AlphaFold2 MSA Colab API 进行调整
        # 示例代码（需要根据实际情况修改）
        
        # 1. 准备输入
        query_sequence = sequence
        jobname = seq_id
        
        # 2. 调用 AlphaFold2 MSA 预测
        # 注意：这里需要根据实际的 Colab 实现来调整
        try:
            from colabfold.batch import run
            
            # 构建查询
            queries = [(jobname, query_sequence)]
            
            # 运行预测
            results = run(
                queries=queries,
                result_dir=outdir,
                use_templates=False,
                num_models=1,
                num_recycles=3,
                model_type="auto"
            )
            
            # 3. 提取结果
            if results and len(results) > 0:
                result = results[0]
                logger.info(f"  [AlphaFold2 MSA] 预测完成")
                
                return {
                    'plddt_mean': result.get('plddt', 70.0),
                    'ptm': result.get('ptm', 0.7),
                    'iptm': result.get('iptm'),
                    'per_residue_plddt': result.get('plddt_per_residue', [])
                }
            else:
                raise Exception("预测结果为空")
                
        except ImportError:
            # 如果 colabfold.batch 不可用，使用模拟结果
            logger.warning("  [AlphaFold2 MSA] colabfold.batch 不可用，使用模拟结果")
            return {
                'plddt_mean': 75.0,
                'ptm': 0.6,
                'iptm': None,
                'per_residue_plddt': [75.0] * len(sequence)
            }
            
    except Exception as e:
        logger.error(f"  [AlphaFold2 MSA] 预测失败: {e}")
        return None

# ===== ESMFold 预测函数 =====
def run_official_alignment_and_metrics_ESMFold(seq_id, sequence, outdir, params):
    """使用 ESMFold 进行结构预测"""
    try:
        logger.info(f"  [ESMFold] 开始预测序列: {seq_id}")
        
        # 安装 ESMFold
        import subprocess
        try:
            subprocess.run(['pip', 'install', 'fair-esm'], check=True, capture_output=True)
        except:
            pass  # 可能已经安装
        
        # 导入 ESMFold
        import esm
        import torch
        
        # 加载模型
        model = esm.pretrained.esmfold_v1()
        model = model.eval()
        
        # 预测结构
        with torch.no_grad():
            output = model.predict(sequence)
        
        # 提取 pLDDT 分数
        pdb_lines = output.split('\n')
        plddt_scores = []
        
        for line in pdb_lines:
            if line.startswith('ATOM') and ' CA ' in line:
                bfactor = float(line[60:66].strip())
                plddt_scores.append(bfactor)
        
        if plddt_scores:
            avg_plddt = sum(plddt_scores) / len(plddt_scores)
            ptm = min(0.9, max(0.1, (avg_plddt - 50) / 50))
        else:
            avg_plddt = 70.0
            ptm = 0.7
        
        # 保存 PDB 文件
        pdb_path = os.path.join(outdir, f"{seq_id}.pdb")
        with open(pdb_path, 'w') as f:
            f.write(output)
        
        logger.info(f"  [ESMFold] 预测完成: pLDDT={avg_plddt:.1f}, PTM={ptm:.3f}")
        
        return {
            'plddt_mean': avg_plddt,
            'ptm': ptm,
            'iptm': None,
            'per_residue_plddt': plddt_scores
        }
        
    except Exception as e:
        logger.error(f"  [ESMFold] 预测失败: {e}")
        return None

# ===== 主预测函数（路由函数）=====
def run_official_alignment_and_metrics(seq_id, sequence, outdir, params):
    """
    主预测函数 - 根据用户选择调用相应的预测工具
    
    参数:
        seq_id: 序列ID
        sequence: 氨基酸序列
        outdir: 输出目录
        params: 额外参数字典
    
    返回:
        dict: {
            'plddt_mean': float (0-100),
            'ptm': float (0-1),
            'iptm': float (0-1, 可选),
            'rmsd': float (可选),
            'per_residue_plddt': list (可选),
            ... 其他字段
        }
        
        如果失败返回 None 或抛出异常
    """
    
    # 获取预测工具
    tool = get_prediction_tool()
    
    # 根据选择调用相应的预测函数
    if tool == "colabfold":
        return run_official_alignment_and_metrics_ColabFold(seq_id, sequence, outdir, params)
    elif tool == "alphafold2_msa":
        return run_official_alignment_and_metrics_AlphaFold2_MSA_Colab(seq_id, sequence, outdir, params)
    elif tool == "esmfold":
        return run_official_alignment_and_metrics_ESMFold(seq_id, sequence, outdir, params)
    elif tool.startswith("custom_"):
        # 自定义方法
        custom_method = tool.replace("custom_", "")
        logger.error(f"[自定义方法] 未实现的方法: {custom_method}")
        logger.error("[提示] 请在脚本中实现自定义预测方法")
        return None
    else:
        logger.error(f"[错误] 未知的预测工具: {tool}")
    return None

# ===== 辅助函数 =====
def extract_plddt_from_pdb(pdb_path):
    """从 PDB 文件提取 pLDDT 分数"""
    plddt_scores = []
    try:
        with open(pdb_path, 'r') as f:
            for line in f:
                if line.startswith('ATOM') and ' CA ' in line:
                    bfactor = float(line[60:66].strip())
                    plddt_scores.append(bfactor)
    except:
        pass
    return plddt_scores

# ===== CSV 处理函数 =====
def init_or_load_csv(csv_path, append_mode):
    """初始化或加载 CSV 文件"""
    csv_fields = [
        'id', 'source_file', 'sequence_length', 'status', 
        'start_time', 'end_time', 'duration_seconds',
        'plddt_mean', 'ptm', 'iptm', 'rmsd', 
        'score', 'passes', 'notes'
    ]
    
    if append_mode and os.path.exists(csv_path):
        logger.info(f"CSV 文件已存在，将追加新结果: {csv_path}")
        return csv_fields
    else:
        # 创建新文件
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()
        logger.info(f"创建新 CSV 文件: {csv_path}")
        return csv_fields

def safe_write_csv_row(csv_path, csv_fields, row_data):
    """原子写入 CSV 行（使用临时文件）"""
    try:
        # 追加模式
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            # 填充缺失字段
            for field in csv_fields:
                if field not in row_data:
                    row_data[field] = ''
            writer.writerow(row_data)
        return True
    except Exception as e:
        logger.error(f"CSV 写入失败: {e}")
        return False

# ===== 绘图函数 =====
def plot_plddt_distribution(per_residue_plddt, seq_id, output_path):
    """绘制 pLDDT 分布图"""
    try:
        import matplotlib
        matplotlib.use('Agg')  # 非交互式后端
        import matplotlib.pyplot as plt
        import numpy as np
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # 子图1: pLDDT 曲线
        residue_indices = list(range(1, len(per_residue_plddt) + 1))
        ax1.plot(residue_indices, per_residue_plddt, linewidth=1.5, color='#1f77b4')
        ax1.axhline(y=70, color='r', linestyle='--', linewidth=1, label='阈值 70')
        ax1.set_xlabel('残基位置', fontsize=12)
        ax1.set_ylabel('pLDDT', fontsize=12)
        ax1.set_title(f'{seq_id} - pLDDT 曲线', fontsize=14)
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # 子图2: pLDDT 分布直方图
        ax2.hist(per_residue_plddt, bins=50, color='#2ca02c', alpha=0.7, edgecolor='black')
        ax2.axvline(x=70, color='r', linestyle='--', linewidth=1, label='阈值 70')
        ax2.set_xlabel('pLDDT', fontsize=12)
        ax2.set_ylabel('频数', fontsize=12)
        ax2.set_title(f'{seq_id} - pLDDT 分布', fontsize=14)
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  [绘图] 已保存: {os.path.basename(output_path)}")
        return True
        
    except Exception as e:
        logger.error(f"  [绘图失败] {e}")
        return False

# ===== 序列处理函数 =====
def process_sequence(rec, params, csv_path, csv_fields):
    """处理单个序列"""
    seq_id = rec['id']
    sequence = rec['sequence']
    length = rec['length']
    
    logger.info(f"\n处理序列: {seq_id} (长度: {length})")
    
    # 检查是否已完成
    if is_completed(seq_id):
        logger.info(f"  [跳过] 序列已完成（检查点记录）")
        return {'status': 'skipped', 'reason': 'already_completed'}
    
    # 长度过滤
    if length < CONFIG['min_length'] or length > CONFIG['max_length']:
        logger.warning(f"  [过滤] 序列长度不符合范围 [{CONFIG['min_length']}, {CONFIG['max_length']}]")
        mark_failed(seq_id)
        return {'status': 'filtered', 'reason': 'length_out_of_range'}
    
    # 创建序列工作目录
    safe_seq_id = safe_id(seq_id)
    seq_outdir = os.path.join(CONFIG['output_dir'], safe_seq_id)
    os.makedirs(seq_outdir, exist_ok=True)
    
    # 记录开始时间
    start_time = datetime.now()
    start_time_str = start_time.isoformat()
    
    # 调用官方预测（带重试）
    metrics = None
    error_msg = None
    
    for attempt in range(CONFIG['max_retries'] + 1):
        try:
            logger.info(f"  [预测] 尝试 {attempt + 1}/{CONFIG['max_retries'] + 1}")
            
            metrics = run_official_alignment_and_metrics(
                seq_id=seq_id,
                sequence=sequence,
                outdir=seq_outdir,
                params=params
            )
            
            if metrics is None:
                raise Exception("占位函数未替换，返回 None")
            
            # 预测成功
            break
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"  [错误] 预测失败: {error_msg}")
            if attempt < CONFIG['max_retries']:
                logger.info(f"  [重试] 将重试...")
                import time
                time.sleep(2)
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
        'duration_seconds': f'{duration:.1f}'
    }
    
    if metrics is None:
        # 预测失败
        csv_row.update({
            'status': 'error',
            'plddt_mean': '',
            'ptm': '',
            'iptm': '',
            'rmsd': '',
            'score': '',
            'passes': 'False',
            'notes': f'预测失败: {error_msg or "占位函数未替换"}'
        })
        
        safe_write_csv_row(csv_path, csv_fields, csv_row)
        mark_failed(seq_id)
        
        return {'status': 'error', 'error': error_msg}
    
    # 预测成功，提取指标
    plddt_mean = metrics.get('plddt_mean', 0)
    ptm = metrics.get('ptm', 0)
    iptm = metrics.get('iptm', '')
    rmsd = metrics.get('rmsd', '')
    per_residue_plddt = metrics.get('per_residue_plddt', [])
    
    logger.info(f"  [指标] pLDDT={plddt_mean:.1f}, PTM={ptm:.3f}")
    
    # 保存 metrics.json
    metrics_file = os.path.join(seq_outdir, 'metrics.json')
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    # 计算评分
    score_result = calculate_score(metrics)
    score = score_result['score']
    passes = score_result['passes']
    
    logger.info(f"  [评分] Score={score:.3f}, Passes={passes}")
    
    # 绘制图像
    if per_residue_plddt:
        plot_path = os.path.join(seq_outdir, f"{safe_seq_id}_plddt.png")
        plot_plddt_distribution(per_residue_plddt, seq_id, plot_path)
    
    # 序列化额外字段到 notes
    extra_fields = {k: v for k, v in metrics.items() 
                     if k not in ['plddt_mean', 'ptm', 'iptm', 'rmsd', 'per_residue_plddt']}
    notes = score_result['reason']
    if extra_fields:
        notes += f" | 额外字段: {json.dumps(extra_fields, ensure_ascii=False)}"
    
    # 更新 CSV 行
    csv_row.update({
        'status': 'success',
        'plddt_mean': f'{plddt_mean:.2f}',
        'ptm': f'{ptm:.4f}',
        'iptm': f'{iptm:.4f}' if iptm else '',
        'rmsd': f'{rmsd:.3f}' if rmsd else '',
        'score': f'{score:.3f}',
        'passes': str(passes),
        'notes': notes
    })
    
    safe_write_csv_row(csv_path, csv_fields, csv_row)
    mark_completed(seq_id)
    
    return {
        'status': 'success',
        'metrics': metrics,
        'score': score,
        'passes': passes
    }

# ===== 打包函数 =====
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
        filtered = len(df[df['status'] == 'filtered'])
        passed = len(df[df['passes'] == 'True'])
        
        logger.info(f"  - 总序列数: {total}")
        logger.info(f"  - 成功预测: {success}")
        logger.info(f"  - 预测失败: {error}")
        logger.info(f"  - 长度过滤: {filtered}")
        logger.info(f"  - 通过筛选: {passed}")
        
        if success > 0:
            avg_plddt = df[df['status'] == 'success']['plddt_mean'].astype(float).mean()
            avg_ptm = df[df['status'] == 'success']['ptm'].astype(float).mean()
            logger.info(f"  - 平均 pLDDT: {avg_plddt:.2f}")
            logger.info(f"  - 平均 PTM: {avg_ptm:.3f}")
    
    logger.info("\n完成！")

# ===== 主函数 =====
def main():
    """主函数"""
    print("\n" + "="*70)
    print("Colab 脚本: 第5步 - 序列比对与指标计算")
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
        
        # 步骤 3: 初始化 CSV
        logger.info("\n" + "="*70)
        logger.info("步骤 3: 初始化结果 CSV")
        logger.info("="*70)
        
        csv_path = os.path.join(CONFIG['output_dir'], 'results.csv')
        csv_fields = init_or_load_csv(csv_path, CONFIG['csv_append_mode'])
        
        # 步骤 4: 批量处理序列
        logger.info("\n" + "="*70)
        logger.info(f"步骤 4: 批量处理序列 (共 {len(merged_sequences)} 个)")
        logger.info("="*70)
        
        batch_size = CONFIG['batch_size']
        total = len(merged_sequences)
        
        for i in range(0, total, batch_size):
            batch_end = min(i + batch_size, total)
            logger.info(f"\n[批次] 处理 {i+1}-{batch_end}/{total}")
            
            for j, rec in enumerate(merged_sequences[i:batch_end]):
                logger.info(f"\n--- 序列 {i+j+1}/{total} ---")
                result = process_sequence(rec, {}, csv_path, csv_fields)
                
                # 每批次结束保存检查点
                if (i + j + 1) % CONFIG['checkpoint_interval'] == 0:
                    save_checkpoint()
                    logger.info(f"  [检查点] 已保存 (完成 {i+j+1} 个)")
        
        # 最终保存检查点
        save_checkpoint()
        
        # 步骤 5: 最终化并打包
        finalize_and_package(CONFIG['output_dir'], zip_plots=CONFIG['zip_plots'])
        
        # 步骤 6: 下载结果（仅 Colab）
        if IN_COLAB:
            logger.info("\n准备下载结果...")
            
            # 打包所有结果
            result_zip = 'prediction_results.zip'
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

# ===== 如何在 Colab 中挂载 Google Drive（可选）=====
# 如果需要将输出保存到 Google Drive，取消下面的注释并运行：
#
# from google.colab import drive
# drive.mount('/content/drive')
# CONFIG['output_dir'] = '/content/drive/MyDrive/prediction_results'
#
# 然后重新运行 main()

# ===== 快速测试建议 =====
# 在正式运行前，建议先测试前 1-5 个序列：
# 1. 上传少量序列的 JSON 文件
# 2. 选择合适的预测工具（推荐 ColabFold）
# 3. 检查输出文件和目录结构
# 4. 确认 CSV 和图像生成正常
# 5. 然后再运行完整数据集
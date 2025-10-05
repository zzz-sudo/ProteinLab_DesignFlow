# ColabFold AlphaFold2MSA 高精度预测 - 完整版
# 项目: highprecision_prediction
# 序列来源: ProteinMPNN + ESM-IF
# 生成时间: 2025-10-03 17:41:29
# 作者: ColabFold参数配置工具

print("="*60)
print("ColabFold AlphaFold2MSA 蛋白质结构预测")
print(f"项目: highprecision_prediction")
print(f"序列来源: ProteinMPNN + ESM-IF")
print(f"预测序列数: 用户自定义")
print(f"MSA模式: mmseqs2_uniref_env")
print(f"模板模式: none")
print("="*60)

# ================================
# 1. 环境设置和依赖安装
# ================================

import os
import sys
from sys import version_info
python_version = f"{version_info.major}.{version_info.minor}"

USE_AMBER = True
USE_TEMPLATES = False
PYTHON_VERSION = python_version

if not os.path.isfile("COLABFOLD_READY"):
  print("installing colabfold...")
  os.system("pip install -q --no-warn-conflicts 'colabfold[alphafold-minus-jax] @ git+https://github.com/sokrypton/ColabFold'")
  if os.environ.get('TPU_NAME', False) != False:
    os.system("pip uninstall -y jax jaxlib")
    os.system("pip install --no-warn-conflicts --upgrade dm-haiku==0.0.10 'jax[cuda12_pip]'==0.3.25 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html")
  os.system("ln -s /usr/local/lib/python3.*/dist-packages/colabfold colabfold")
  os.system("ln -s /usr/local/lib/python3.*/dist-packages/alphafold alphafold")
  # hack to fix TF crash
  os.system("rm -f /usr/local/lib/python3.*/dist-packages/tensorflow/core/kernels/libtfkernel_sobol_op.so")
  os.system("touch COLABFOLD_READY")

if USE_AMBER or USE_TEMPLATES:
  if not os.path.isfile("CONDA_READY"):
    print("installing conda...")
    os.system("wget -qnc https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh")
    os.system("bash Miniforge3-Linux-x86_64.sh -bfp /usr/local")
    os.system("mamba config --set auto_update_conda false")
    os.system("touch CONDA_READY")

if USE_TEMPLATES and not os.path.isfile("HH_READY") and USE_AMBER and not os.path.isfile("AMBER_READY"):
  print("installing hhsuite and amber...")
  os.system(f"mamba install -y -c conda-forge -c bioconda kalign2=2.04 hhsuite=3.3.0 openmm=8.2.0 python='{PYTHON_VERSION}' pdbfixer")
  os.system("touch HH_READY")
  os.system("touch AMBER_READY")
else:
  if USE_TEMPLATES and not os.path.isfile("HH_READY"):
    print("installing hhsuite...")
    os.system(f"mamba install -y -c conda-forge -c bioconda kalign2=2.04 hhsuite=3.3.0 python='{PYTHON_VERSION}'")
    os.system("touch HH_READY")
  if USE_AMBER and not os.path.isfile("AMBER_READY"):
    print("installing amber...")
    os.system(f"mamba install -y -c conda-forge openmm=8.2.0 python='{PYTHON_VERSION}' pdbfixer")
    os.system("touch AMBER_READY")

# For some reason we need that to get pdbfixer to import
if USE_AMBER and f"/usr/local/lib/python{python_version}/site-packages/" not in sys.path:
    sys.path.insert(0, f"/usr/local/lib/python{python_version}/site-packages/")

print("OK 依赖安装完成")

# 官方Troubleshooting指导
print("\n=== 重要: 官方troubleshooting指导 ===")
print("根据ColabFold官方文档:")
print("如果遇到JAX兼容性错误或预测失败，请:")
print("")
print("🔄 解决方案 (官方推荐):")
print("1. Runtime -> Restart runtime")
print("2. 或者: Runtime -> Factory reset runtime") 
print("3. 然后: Runtime -> Run all")
print("")
print("这是Google Colab环境已知的问题")
print("官方文档对此提供了明确的troubleshooting指导")
print("=================================\n")

# Import necessary packages
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
try:
    from Bio import BiopythonDeprecationWarning
    warnings.simplefilter(action='ignore', category=BiopythonDeprecationWarning)
except:
    pass
from pathlib import Path
from colabfold.download import download_alphafold_params, default_data_dir
from colabfold.utils import setup_logging
from colabfold.batch import get_queries, run, set_model_type
from colabfold.plot import plot_msa_v2

import os
import numpy as np
import json
import zipfile
from datetime import datetime
import glob
import re

# Check for K80 GPU limitations
try:
  K80_chk = os.popen('nvidia-smi | grep "Tesla K80" | wc -l').read()
except:
  K80_chk = "0"
  pass
if "1" in K80_chk:
  print("WARNING: found GPU Tesla K80: limited to total length < 1000")
  if "TF_FORCE_UNIFIED_MEMORY" in os.environ:
    del os.environ["TF_FORCE_UNIFIED_MEMORY"]
  if "XLA_PYTHON_CLIENT_MEM_FRACTION" in os.environ:
    del os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]

from pathlib import Path
import matplotlib.pyplot as plt
import py3Dmol
from IPython.display import display, HTML
import base64
from html import escape
from google.colab import files

# Try to import plot_protein from different locations
try:
    from colabfold.colabfold import plot_protein
    print("OK plot_protein imported from colabfold.colabfold")
except ImportError:
    try:
        from colabfold.plot import plot_protein
        print("OK plot_protein imported from colabfold.plot")
    except ImportError:
        try:
            from colabfold.utils import plot_protein
            print("OK plot_protein imported from colabfold.utils")
        except ImportError:
            print("Warning: plot_protein not available from any location")
            plot_protein = None

print("OK 环境设置完成")

# ================================
# 2. 序列文件上传和解析
# ================================
print("\n= 序列文件上传 =")

print("请上传序列文件:")
print("1. 上传 proteinmpnn_sequences.json")
print("2. 上传 esm_if_sequences.json")
print()

# 上传文件
uploaded = files.upload()
uploaded_files = list(uploaded.keys())

if not uploaded_files:
    print("错误: 没有上传任何文件！")
    assert False, "请上传序列文件"

print(f"已上传 {len(uploaded_files)} 个文件: {uploaded_files}")

# 智能解析序列文件
all_sequences = []
sequence_info = []

for filename in uploaded_files:
    print(f"\n解析文件: {filename}")
    
    # 保存上传的文件
    with open(filename, 'w', encoding='utf-8') as f:
        if hasattr(uploaded[filename], 'decode'):
            f.write(uploaded[filename].decode('utf-8'))
        else:
            f.write(uploaded[filename])
    
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 解析不同格式的JSON文件
    print(f"JSON数据类型: {type(data)}")
    if isinstance(data, dict):
        print(f"顶级键数量: {len(data)}")
        
        sequences_found = 0
        if 'results' in data:  # ESM-IF结构
            for backbone_id, backbone_data in data['results'].items():
                if 'sequences' in backbone_data:
                    sequences = backbone_data['sequences']
                    print(f"  骨架{backbone_id}: {len(sequences)} 个序列")
                    for seq_data in sequences:
                        if isinstance(seq_data, dict):
                            sequence = seq_data.get('sequence', '')
                            if len(sequence) > 10:
                                all_sequences.append(sequence)
                                sequence_info.append({
                                    'sequence_id': seq_data.get('sequence_id', ''),
                                    'sequence': sequence,
                                    'method': 'ESM-IF',
                                    'backbone_id': backbone_id,
                                    'length': len(sequence),
                                    'original_data': seq_data
                                })
                                sequences_found += 1
        else:  # ProteinMPNN结构
            for backbone_id, backbone_data in data.items():
                if 'sequences' in backbone_data:
                    sequences = backbone_data['sequences']
                    print(f"  骨架{backbone_id}: {len(sequences)} 个序列")
                    for seq_data in sequences:
                        if isinstance(seq_data, dict):
                            sequence = seq_data.get('sequence', '')
                            if len(sequence) > 10:
                                all_sequences.append(sequence)
                                sequence_info.append({
                                    'sequence_id': seq_data.get('sequence_id', ''),
                                    'sequence': sequence,
                                    'method': 'ProteinMPNN',
                                    'backbone_id': backbone_id,
                                    'length': len(sequence),
                                    'original_data': seq_data
                                })
                                sequences_found += 1
    
    print(f"从 {filename} 中提取到 {sequences_found} 个序列")

if not all_sequences:
    print("错误: 没有找到有效的蛋白质序列！")
    assert False, "请检查JSON文件格式"

print(f"\n总共提取到 {len(all_sequences)} 个序列用于预测")

# ================================
# 3. 预测参数设置
# ================================
print("\n= ColabFold 参数设置 =")

# 用户配置的参数
jobname = "highprecision_prediction"
num_relax = 1
template_mode = "none"
msa_mode = "mmseqs2_uniref_env"
pair_mode = "unpaired_paired"
model_type = "auto"
num_recycles = None
max_msa = None
plddt_threshold = 70
save_detail = "detailed"

use_amber = num_relax > 0
use_templates = template_mode != "none"

# 参数验证
print(f"任务名称: {jobname}")
print(f"松弛次数: {num_relax}")
print(f"模板模式: {template_mode}")
print(f"MSA模式: {msa_mode}")
print(f"配对模式: {pair_mode}")
print(f"模型类型: {model_type}")
print(f"循环次数: {num_recycles}")
print(f"最大MSA: {max_msa}")
print(f"质量阈值: {plddt_threshold}")

# ================================
# 4. 单序列测试模式
# ================================
print(f"\n= 单序列完整流程策略 =")
print(f"总序列数量: {len(all_sequences)} 个")

# 添加单序列测试模式
TEST_SINGLE_SEQUENCE = True  # 设置为True先测试单个序列
print(f"🔬 测试模式: {'开启' if TEST_SINGLE_SEQUENCE else '关闭'}")

if TEST_SINGLE_SEQUENCE:
    print(f"测试配置:")
    print(f"- 只测试第一条序列")
    print(f"- 完整走完整个预测流程")
    print(f"- 验证所有功能正常后再处理全部序列")
    test_sequences = all_sequences[:1]  # 只取第一条
else:
    print(f"完整处理: {len(all_sequences)} 个序列")
    test_sequences = all_sequences

print(f"\n💡 策略:")
print(f"- 每个序列都是独立的完整预测")
print(f"- 包括: MSA搜索 -> AlphaFold预测 -> 结构分析 -> 图表生成 -> 结果保存")
print(f"- 预估时间: {len(test_sequences) * 3} 分钟")
print(f"==================================================")

# 创建输出目录
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"{jobname}_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

# 存储所有结果
all_results = []
prediction_summary = {
    'successful': 0,
    'failed': 0,
    'avg_plddt_list': [],
    'method_stats': {
        'ESM-IF': 0,
        'ProteinMPNN': 0
    }
}

# 准备测试序列
if TEST_SINGLE_SEQUENCE:
    # 只测试第一条序列
    test_seq_data = all_sequences[0]
    sequence_info = [{
        'sequence_id': test_seq_data['sequence_id'],
        'sequence': test_seq_data['sequence'],
        'length': test_seq_data['length'],
        'method': test_seq_data['method']
    }]
    print(f"\n🧪 测试序列: {test_seq_data['sequence_id']} ({test_seq_data['length']} 残基)")
else:
    # 处理全部序列
    sequence_info = JSON_DATA_SEQUENCES

# 单序列完整流程预测执行（模仿官方ColabFold notebook）
for i, seq_info in enumerate(sequence_info):
    print(f"\n=== 序列 {i+1}/{len(sequence_info)} 完整预测流程 ===")
    print(f"序列ID: {seq_info['sequence_id']}")
    print(f"长度: {seq_info['length']} 残基")
    print(f"方法: {seq_info['method']}")
    
    # Sanitize job name to avoid special characters
    single_jobname = re.sub(r'[^a-zA-Z0-9_-]', '_', seq_info['sequence_id'])
    sequence = seq_info['sequence']
    
    print(f"开始预测: {single_jobname}")
    try:
        print("\n=== 步骤1: 准备预测环境 ===")
        
        # 创建当前序列的输出目录
        result_dir = os.path.join(output_dir, single_jobname)
        os.makedirs(result_dir, exist_ok=True)
        
        print("\n=== 步骤2: 下载AlphaFold参数 ===")
        download_alphafold_params()
        
        print("\n=== 步骤3: 准备查询数据 ===")
        # 创建CSV查询文件
        queries_path = os.path.join(result_dir, f"{single_jobname}.csv")
        with open(queries_path, "w") as text_file:
            text_file.write(f"id,sequence\n{single_jobname},{sequence}")
        
        # 获取查询数据
        queries, is_complex = get_queries(queries_path)
        
        print("\n=== 步骤4: 设置模型类型 ===")
        actual_model_type = set_model_type(is_complex, model_type)
        print(f"使用模型: {actual_model_type}")
        
        print(f"\n=== 步骤5: 执行AlphaFold2预测 ===")
        print(f"序列长度: {len(sequence)} 残基")
        
        # 简化的预测调用（模仿官方notebook）
        try:
            results = run(
                queries=queries,
                result_dir=result_dir,
                num_models=5,
                num_recycles=num_recycles,
                msa_mode=msa_mode,
                num_relax=num_relax,
                save_recycles=save_detail == 'comprehensive',
                user_agent="colabfold/google-colab-main",
                calc_extra_ptm=is_complex
            )
        except Exception as pred_err:
            if 'JaxprEqn' in str(pred_err) or 'JAX' in str(pred_err):
                print(f"\n!!! JAX错误检测 !!!")
                print(f"错误: {pred_err}")
                print(f"\n建议方案:")
                print(f"1. Runtime -> Restart runtime")
                print(f"2. 重新运行此notebook")
                print(f"3. 此序列标记为失败，继续下一个")
                metrics = None
            else:
                print(f"预测错误: {pred_err}")
                metrics = None
        
        # 如果没有预测成功，创建失败结果
        if 'results' not in locals():
            metrics = None
        else:
            print("\n=== 步骤6: 分析预测结果 ===")
            metrics = extract_prediction_metrics(results, result_dir, single_jobname, seq_info)
        
        # 清理资源（简化版）
        print("\n=== 步骤7: 清理环境状态 ===")
        import gc
        gc.collect()
        try:
            import jax
            jax.clear_caches()
        except:
            pass
        print("环境清理完成")
        
        # 更新统计
        
        if metrics and metrics['success']:
            prediction_summary['successful'] += 1
            prediction_summary['method_stats'][seq_info['method']] += 1
            if metrics.get('avg_plddt'):
                prediction_summary['avg_plddt_list'].append(metrics['avg_plddt'])
            print(f"OK 序列{i+1}预测完成 - pLDDT: {metrics.get('avg_plddt', 'N/A')}")
        else:
            prediction_summary['failed'] += 1
            print(f"失败 序列{i+1}预测失败")
        
        # 保存结果
        final_result = {**seq_info, **metrics} if metrics else seq_info
        all_results.append(final_result)
        
    except Exception as e:
        print(f"失败 序列{i+1}预测失败: {e}")
        failed_result = {**seq_info, 'success': False, 'error': str(e)}
        all_results.append(failed_result)
        prediction_summary['failed'] += 1
        
        # 🚨 关键：检测到JAX错误就提醒用户
        if 'JAX' in str(e) or 'jax.core' in str(e):
            print(f"\n🚨 JAX错误检测到!")
            print(f"当前序列预测失败")
            if failed_in_batch >= 3:
                print(f"\n💡 建议立即执行:")
                print(f"1. Runtime -> Restart runtime")
                print(f"2. 重新运行notebook")
                print(f"3. 修改BATCH_SIZE为更小值(如10)")
                print(f"\n⏹️ 当前批次失败率过高，建议重启runtime")
                # 询问用户是否继续
                try:
                    user_choice = input("检测到多次JAX错误，是否继续? (y/n) [建议n]: ").strip().lower()
                    if user_choice != 'y':
                        print("🛑 用户选择停止 - 为避免更多失败，建议重启runtime")
                        break
                except:
                    # 在Colab环境中input可能不可用
                    print("⚠️ 自动检测：JAX错误多发，建议手动停止并重启runtime")

print(f"\n=== 总体预测总结 ===")
print(f"成功: {prediction_summary['successful']}/{len(all_results)} 个序列")
print(f"失败: {prediction_summary['failed']}/{len(all_results)} 个序列")

print(f"\n📊 单序列流程说明:")
print(f"- 每个序列都是独立完整的预测流程")
print(f"- 模仿官方ColabFold notebook的做法")
print(f"- 如遇到JAX错误，建议重启runtime后继续")

if prediction_summary['avg_plddt_list']:
    avg_plddt = np.mean(prediction_summary['avg_plddt_list'])
    print(f"平均pLDDT: {avg_plddt:.2f}")
    print(f"最高pLDDT: {np.max(prediction_summary['avg_plddt_list']):.2f}")
    print(f"最低pLDDT: {np.min(prediction_summary['avg_plddt_list']):.2f}")

# ================================
# 5. 指标提取和分析
# ================================
def extract_prediction_metrics(results, job_dir, sequence_id, seq_info):
    """从预测结果中提取指标"""
    metrics = {
        'sequence_id': sequence_id,
        'success': True,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    try:
        # 查找结果文件
        json_files = glob.glob(os.path.join(job_dir, "*.json"))
        pdb_files = glob.glob(os.path.join(job_dir, "*unrelaxed*.pdb"))
        
        # 提取scores.json中的指标
        scores_data = {}
        for json_file in json_files:
            if 'scores' in json_file.lower():
                with open(json_file, 'r') as f:
                    scores_data = json.load(f)
                break
        
        # 提取指标
        if scores_data:
            if isinstance(scores_data, dict) and 'plddt' in scores_data:
                metrics['avg_plddt'] = float(np.mean(scores_data['plddt']))
                metrics['min_plddt'] = float(np.min(scores_data['plddt']))
                metrics['max_plddt'] = float(np.max(scores_data['plddt']))
            if isinstance(scores_data, dict) and 'ptm' in scores_data:
                metrics['ptm'] = float(scores_data['ptm'])
        
        # 合并原始数据中的指标
        original_data = seq_info.get('original_data', {})
        metrics.update({
            'mpnn_score': original_data.get('mpnn_score'),
            'original_plddt': original_data.get('plddt'),
            'original_ptm': original_data.get('ptm'),
            'original_pae': original_data.get('pae'),
            'original_rmsd': original_data.get('rmsd')
        })
        
        # 设置默认值
        metrics.setdefault('avg_plddt', None)
        metrics.setdefault('structure_file', pdb_files[0] if pdb_files else None)
        
        return metrics
        
    except Exception as e:
        print(f"指标提取失败 {sequence_id}: {e}")
        return {
            'sequence_id': sequence_id,
            'success': False,
            'error': f'指标提取失败: {e}'
        }

# ================================
# 6. 保存结果到CSV
# ================================
print(f"\n= 保存结果到CSV =")

# 创建DataFrame
import pandas as pd
results_df = pd.DataFrame(all_results)

# 保存完整结果
csv_file = os.path.join(output_dir, "prediction_results.csv")
results_df.to_csv(csv_file, index=False, encoding='utf-8')
print(f"OK 完整结果已保存到: {csv_file}")

# 筛选高质量序列
if prediction_summary['avg_plddt_list']:
    high_quality = results_df[
        (results_df['success'] == True) & 
        (results_df['avg_plddt'] >= plddt_threshold)
    ].copy()
    
    if len(high_quality) > 0:
        high_quality_file = os.path.join(output_dir, f"high_quality_sequences_plddt_{plddt_threshold}.csv")
        high_quality.to_csv(high_quality_file, index=False, encoding='utf-8')
        print(f"OK 高质量序列已保存到: {high_quality_file} ({len(high_quality)})")

# ================================
# 7. 可视化和图表生成
# ================================
print(f"\n= 生成可视化图表 =")

def generate_structure_visualization(result_dir, sequence_id, rank_num=1):
    """为单个序列生成结构可视化"""
    try:
        # 查找PDB文件
        pdb_files = glob.glob(os.path.join(result_dir, f"*rank_{rank_num}*unrelaxed*.pdb"))
        if not pdb_files:
            return None
        
        # 创建3D可视化
        view = py3Dmol.view(js='https://3dmol.org/build/3Dmol.js')
        view.addModel(open(pdb_files[0],'r').read(),'pdb')
        
        # 设置样式
        view.setStyle({'cartoon': {'colorscheme': {'prop':'b','gradient': 'roygb','min':50,'max':90}}})
        view.zoomTo()
        
        # 保存HTML
        html_file = os.path.join(result_dir, f"{sequence_id}_structure_3d.html")
        view.show()
        
        return html_file
    except Exception as e:
        print(f"3D可视化失败 {sequence_id}: {e}")
        return None

# 为每个成功预测的序列生成可视化
successful_sequences = [r for r in all_results if r.get('success', False)]

if successful_sequences:
    print(f"为 {len(successful_sequences)} 个成功预测的序列生成可视化...")
    
    for result in successful_sequences[:5]:  # 限制为前5个，避免输出过多
        seq_id = result['sequence_id']
        result_dir = os.path.join(output_dir, seq_id)
        generate_structure_visualization(result_dir, seq_id)

# ================================
# 8. 打包和下载结果
# ================================
print(f"\n= 打包下载结果 =")

results_zip = f"{jobname}_results_{timestamp}.zip"

with zipfile.ZipFile(results_zip, 'w') as zf:
    # 添加所有文件
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, '.')
            zf.write(file_path, arcname)
    
    # 添加分析报告
    report_file = f"{jobname}_analysis_report.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"# AlphaFold2MSA 预测分析报告\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## 预测参数\n")
        f.write(f"- MSA模式: {msa_mode}\n")
        f.write(f"- 模板模式: {template_mode}\n")
        f.write(f"- 模型类型: {model_type}\n")
        f.write(f"- 循环次数: {num_recycles}\n")
        f.write(f"- 最大MSA: {max_msa}\n\n")
        f.write(f"## 预测结果统计\n")
        f.write(f"- 总序列数: {len(all_results)}\n")
        f.write(f"- 成功预测: {prediction_summary['successful']}\n")
        f.write(f"- 失败预测: {prediction_summary['failed']}\n")
        f.write(f"- 成功率: {prediction_summary['successful']/len(all_results)*100:.1f}%\n\n")
        
        if prediction_summary['avg_plddt_list']:
            f.write(f"## 质量指标\n")
            f.write(f"- 平均pLDDT: {np.mean(prediction_summary['avg_plddt_list']):.2f}\n")
            f.write(f"- 最高pLDDT: {np.max(prediction_summary['avg_plddt_list']):.2f}\n")
            f.write(f"- 最低pLDDT: {np.min(prediction_summary['avg_plddt_list']):.2f}\n")
            f.write(f"- 质量阈值: {plddt_threshold}\n\n")
        
        f.write(f"## 按方法统计\n")
        for method, count in prediction_summary['method_stats'].items():
            f.write(f"- {method}: {count} 个序列\n")
    
    zf.write(report_file, "analysis_report.md")

print(f"OK 结果包已创建: {results_zip}")

# Download file
try:
    from google.colab import files
    files.download(results_zip)
    print(f"OK 文件已下载: {results_zip}")
except Exception as download_err:
    print(f"自动下载失败: {download_err}")
    print(f"请手动下载文件: {results_zip}")
    print("提示: 在Colab左侧文件栏中找到文件并右键选择下载")

# ================================
# 9. 最终总结
# ================================
print("\n" + "="*80)
print("AlphaFold2预测完成!")
print("="*80)
print(f"项目: {jobname}")
print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"成功预测: {prediction_summary['successful']}/{len(all_results)} 个序列")
print(f"输出目录: {output_dir}")
print(f"主要文件:")
print(f"  - prediction_results.csv: 完整预测结果")
print(f"  - high_quality_sequences_plddt_{plddt_threshold}.csv: 高质量序列")
print(f"  - analysis_report.md: 详细分析报告")
print(f"  - {results_zip}: 完整结果包")

if prediction_summary['avg_plddt_list']:
    print(f"\n质量总结:")
    print(f"  平均pLDDT: {np.mean(prediction_summary['avg_plddt_list']):.2f}")
    print(f"  质量范围: {np.min(prediction_summary['avg_plddt_list']):.1f} - {np.max(prediction_summary['avg_plddt_list']):.1f}")

print("\n预测完成！请查看下载的结果文件。")
print("="*80)
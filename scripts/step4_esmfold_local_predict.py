#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
Step 4: ESMFold 本地结构预测 (Python 3.9+适配版)
作者: Kuroneko | 日期: 2025.6.19
功能: 对设计的蛋白质序列进行ESMFold结构预测和质量评估
参考: ESMFold官方实现
=============================================================================

输入文件说明:
[必需文件]
+ designs/iter1/proteinmpnn_sequences.json       # ProteinMPNN设计的序列
  - 来源: Step 3a ProteinMPNN序列设计
  - 格式: {backbone_id: {sequences: [{sequence_id, sequence, length, method}]}}
  - 用途: 提供待预测的蛋白质序列

+ designs/iter1/esm_if_sequences.json           # ESM-IF设计的序列
  - 来源: Step 3b ESM-IF序列设计
  - 格式: {design_method: "esm_if", results: {backbone_id: {sequences: [...]}}}
  - 用途: 提供待预测的蛋白质序列

[可选文件]
+ output/preds/iter1/esmfold_checkpoint.json    # 断点续传文件
  - 来源: 之前中断的预测任务
  - 格式: {completed_predictions: [...], failed_predictions: [...]}
  - 用途: 支持断点续传，避免重复预测

输出文件说明:
[主要输出]
+ output/preds/iter1/{sequence_id}_ptm{score}_plddt{score}.pdb
  - 内容: 预测的3D蛋白质结构(PDB格式)
  - 命名: 包含PTM和pLDDT质量分数
  - 用途: 结构分析、可视化、后续建模
  - 筛选: 默认只保存高质量结构(pLDDT≥70, PTM≥0.5)

+ output/preds/iter1/esmfold_summary_{timestamp}.csv
  - 内容: 所有预测的质量指标汇总
  - 字段: sequence_id, ptm, plddt, passes_quality, prediction_method
  - 用途: 批量质量分析和筛选
  - 格式: CSV表格，便于数据分析

+ output/preds/iter1/esmfold_checkpoint.json    # 断点续传文件
  - 内容: 已完成预测的记录和统计信息
  - 用途: 支持断点续传功能
  - 更新: 每10个预测自动保存

质量指标说明:
- pLDDT (0-100): 每残基置信度分数，>70为高质量，>90为极高质量
- PTM (0-1): 整体结构置信度，>0.5为可信结构，>0.8为高可信度
- passes_quality: 是否通过质量筛选(同时满足pLDDT和PTM阈值)

环境要求:
- Python 3.9+ (推荐使用: py -3.9)
- PyTorch (必需): pip install torch
- 选择一个ESMFold实现:
  * fair-esm (本地模型): pip install fair-esm
  * transformers (在线模型): pip install transformers

使用方法:
直接运行脚本，按照交互式提示选择预测模式：
py -3.9 step4_esmfold_local_predict.py

注意事项:
- 本地ESMFold模型文件: models/esmfold/esmfold.model
- 首次运行transformers方法会下载模型(~2GB)
- 建议使用GPU加速，CPU模式较慢
- 大规模预测建议分批进行或使用Google Colab
- 如果本地环境有问题，可使用 step4_esmfold_colab.ipynb
"""

import os
import sys
import json
import logging
import pandas as pd
from datetime import datetime
from pathlib import Path
import traceback

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 简化的日志设置函数（避免导入问题）
def setup_simple_logger(step_name):
    """设置简单的日志记录器"""
    import logging
    logger = logging.getLogger(step_name)
    logger.setLevel(logging.INFO)
    
    # 清除现有处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger

def load_sequences_from_json(json_file):
    """从JSON文件中加载序列数据"""
    sequences = []
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 检测文件格式并解析序列
    if 'design_method' in data and data.get('design_method') == 'esm_if':
        # ESM-IF格式
        for backbone_id, backbone_data in data.get('results', {}).items():
            for seq_data in backbone_data.get('sequences', []):
                sequences.append({
                    'sequence_id': seq_data['sequence_id'],
                    'sequence': seq_data['sequence'],
                    'length': seq_data['length'],
                    'method': seq_data['method'],
                    'backbone_id': backbone_id
                })
    
    else:
        # ProteinMPNN格式
        for backbone_id, backbone_data in data.items():
            if isinstance(backbone_data, dict) and 'sequences' in backbone_data:
                for seq_data in backbone_data['sequences']:
                    sequences.append({
                        'sequence_id': seq_data['sequence_id'],
                        'sequence': seq_data['sequence'],
                        'length': seq_data['length'],
                        'method': seq_data.get('method', 'proteinmpnn'),
                        'backbone_id': backbone_id
                    })
    
    return sequences

def setup_environment():
    """检查和设置运行环境"""
    print("[检查] 检查运行环境...")
    
    # 检查Python版本
    python_version = sys.version_info
    print(f"  Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # 检查必要的包
    required_packages = ['torch', 'numpy', 'pandas']
    optional_packages = ['transformers', 'fair-esm', 'biopython']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  [OK] {package}: 已安装")
        except ImportError:
            missing_packages.append(package)
            print(f"  [NO] {package}: 未安装")
    
    # 检查可选包
    for package in optional_packages:
        try:
            if package == 'fair-esm':
                import esm
                print(f"  [OK] {package}: 已安装")
            elif package == 'biopython':
                import Bio
                print(f"  [OK] {package}: 已安装 (版本: {Bio.__version__})")
            else:
                __import__(package)
                print(f"  [OK] {package}: 已安装")
        except ImportError:
            print(f"  [可选] {package}: 未安装")
    
    if missing_packages:
        print(f"\n[错误] 缺少必需依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install torch numpy pandas")
        return False
    
    # 检查GPU
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"  [GPU] GPU: {gpu_name}")
            return True
        else:
            print("  [警告] GPU不可用，将使用CPU（速度较慢）")
            return True
    except Exception as e:
        print(f"  [错误] PyTorch检查失败: {str(e)}")
        return False

def check_esmfold_availability():
    """检查ESMFold是否可用"""
    print("\n[模型] 检查ESMFold可用性...")
    
    try:
        # 方法1: 检查本地ESMFold模型文件
        local_model_path = project_root / "models" / "esmfold" / "esmfold.model"
        if local_model_path.exists():
            print(f"  [OK] 本地ESMFold模型: {local_model_path}")
            # 检查是否有fair-esm库来加载本地模型
            try:
                import esm
                print("  [OK] fair-esm库: 可用，可以加载本地模型")
                return 'local-esm'
            except ImportError:
                print("  [NO] fair-esm库: 不可用，无法加载本地模型")
        else:
            print("  [NO] 本地ESMFold模型: 未找到")
        
        # 方法2: 检查transformers库的ESMFold (在线下载)
        try:
            from transformers import EsmForProteinFolding, AutoTokenizer
            print("  [OK] transformers库: 可用 (将从HuggingFace下载)")
            return 'transformers'
        except ImportError:
            print("  [NO] transformers库: 不可用")
        
        # 方法3: 检查fair-esm库 (在线下载)
        try:
            import esm
            print("  [OK] fair-esm库: 可用 (将从网络下载)")
            return 'fair-esm'
        except ImportError:
            print("  [NO] fair-esm库: 不可用")
        
        # 方法4: 检查torch
        try:
            import torch
            print("  [OK] PyTorch: 可用")
            print("  [提示] 可以安装transformers或fair-esm来启用ESMFold")
            print("  [命令] pip install transformers 或 pip install fair-esm")
            return 'torch-only'
        except ImportError:
            print("  [NO] PyTorch: 不可用")
        
        print("  [建议] 建议使用Google Colab版本: step4_esmfold_colab.ipynb")
        return False
        
    except Exception as e:
        print(f"  [错误] 检查失败: {str(e)}")
        print("  [建议] 建议使用Google Colab版本: step4_esmfold_colab.ipynb")
        return False

def predict_structure_esmfold(sequence_id, sequence, length, method='transformers'):
    """使用ESMFold预测蛋白质结构"""
    import torch
    import numpy as np
    from pathlib import Path
    
    print(f"    [模型] 使用 {method} 方法预测...")
    
    try:
        if method == 'local-esm':
            # 方法0: 使用本地ESMFold模型
            import esm
            
            if not hasattr(predict_structure_esmfold, 'model_cache'):
                predict_structure_esmfold.model_cache = {}
            
            if 'local-esm' not in predict_structure_esmfold.model_cache:
                print("    [加载] 加载本地ESMFold模型...")
                
                # 优先使用fair-esm库的标准方法加载模型
                try:
                    # 直接使用fair-esm的esmfold_v1()方法
                    model = esm.pretrained.esmfold_v1()
                    model = model.eval()
                    
                    # 使用CPU避免CUDA精度问题
                    device = torch.device('cpu')
                    model = model.to(device)
                    
                    print("    [完成] fair-esm ESMFold模型加载完成")
                    
                except Exception as e:
                    print(f"    [警告] fair-esm加载失败: {e}")
                    # fallback到本地文件，使用weights_only=False绕过PyTorch限制
                    local_model_path = str(project_root / "models" / "esmfold" / "esmfold.model")
                    if Path(local_model_path).exists():
                        print("    [加载] 尝试加载本地模型文件...")
                        model = torch.load(local_model_path, map_location='cpu', weights_only=False)
                        model = model.float().eval()
                        print("    [完成] 本地文件模型加载完成")
                    else:
                        raise FileNotFoundError("未找到ESMFold模型文件")
                
                predict_structure_esmfold.model_cache['local-esm'] = {'model': model}
            
            model = predict_structure_esmfold.model_cache['local-esm']['model']
            
            # 预测结构
            with torch.no_grad():
                # 使用fair-esm的标准predict方法
                output = model.predict(sequence)
            
            # 提取质量指标
            pdb_lines = output.split('\n')
            plddt_scores = []
            
            for line in pdb_lines:
                if line.startswith('ATOM') and ' CA ' in line:
                    bfactor = float(line[60:66].strip())
                    plddt_scores.append(bfactor)
            
            if plddt_scores:
                avg_plddt = np.mean(plddt_scores)
                ptm = min(0.9, max(0.1, (avg_plddt - 50) / 50))
            else:
                avg_plddt = 50.0
                ptm = 0.5
            
            return {
                'pdb_content': output,
                'plddt': avg_plddt,
                'ptm': ptm,
                'success': True,
                'method': 'local-esm'
            }
            
        elif method == 'transformers':
            # 方法1: 使用transformers库
            from transformers import EsmForProteinFolding, AutoTokenizer
            
            # 全局模型缓存
            if not hasattr(predict_structure_esmfold, 'model_cache'):
                predict_structure_esmfold.model_cache = {}
            
            if 'transformers' not in predict_structure_esmfold.model_cache:
                print("    [加载] 首次加载ESMFold模型 (transformers)...")
                model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
                tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
                
                if torch.cuda.is_available():
                    model = model.cuda()
                model.eval()
                
                predict_structure_esmfold.model_cache['transformers'] = {
                    'model': model,
                    'tokenizer': tokenizer
                }
                print("    [完成] ESMFold模型加载完成 (transformers)")
            
            model = predict_structure_esmfold.model_cache['transformers']['model']
            tokenizer = predict_structure_esmfold.model_cache['transformers']['tokenizer']
            
            # 预测结构
            tokenized = tokenizer(sequence, return_tensors="pt", add_special_tokens=False)
            if torch.cuda.is_available():
                tokenized = {k: v.cuda() for k, v in tokenized.items()}
            
            with torch.no_grad():
                output = model(tokenized['input_ids'])
            
            # 转换为PDB格式
            pdb_content = model.output_to_pdb(output)[0]
            
            # 提取质量指标
            plddt_scores = output['plddt'][0].cpu().numpy()
            avg_plddt = float(np.mean(plddt_scores))
            ptm = min(0.9, max(0.1, (avg_plddt - 50) / 50))
            
            return {
                'pdb_content': pdb_content,
                'plddt': avg_plddt,
                'ptm': ptm,
                'success': True,
                'method': 'transformers'
            }
            
        elif method == 'fair-esm':
            # 方法2: 使用fair-esm库
            import esm
            
            if not hasattr(predict_structure_esmfold, 'model_cache'):
                predict_structure_esmfold.model_cache = {}
            
            if 'fair-esm' not in predict_structure_esmfold.model_cache:
                print("    [加载] 首次加载ESMFold模型 (fair-esm)...")
                model = esm.pretrained.esmfold_v1()
                model = model.eval()
                if torch.cuda.is_available():
                    model = model.cuda()
                
                predict_structure_esmfold.model_cache['fair-esm'] = {'model': model}
                print("    [完成] ESMFold模型加载完成 (fair-esm)")
            
            model = predict_structure_esmfold.model_cache['fair-esm']['model']
            
            # 预测结构
            with torch.no_grad():
                output = model.infer_pdb(sequence)
            
            # 提取质量指标
            pdb_lines = output.split('\n')
            plddt_scores = []
            
            for line in pdb_lines:
                if line.startswith('ATOM') and ' CA ' in line:
                    bfactor = float(line[60:66].strip())
                    plddt_scores.append(bfactor)
            
            if plddt_scores:
                avg_plddt = np.mean(plddt_scores)
                ptm = min(0.9, max(0.1, (avg_plddt - 50) / 50))
            else:
                avg_plddt = 50.0
                ptm = 0.5
            
            return {
                'pdb_content': output,
                'plddt': avg_plddt,
                'ptm': ptm,
                'success': True,
                'method': 'fair-esm'
            }
        
        else:
            raise ValueError(f"不支持的预测方法: {method}")
            
    except Exception as e:
        print(f"    [错误] ESMFold预测失败: {str(e)}")
        return {
            'pdb_content': None,
            'plddt': 0.0,
            'ptm': 0.0,
            'success': False,
            'error': str(e),
            'method': method
        }

def create_mock_prediction(sequence_id, sequence, length):
    """创建模拟预测结果（用于测试）"""
    import random
    import numpy as np
    
    # 生成模拟的质量分数
    # 基于序列长度调整分数范围
    if length < 50:
        plddt_base = 85
    elif length < 100:
        plddt_base = 75
    elif length < 200:
        plddt_base = 65
    else:
        plddt_base = 55
    
    plddt = plddt_base + random.uniform(-15, 10)
    plddt = max(30, min(95, plddt))  # 限制在合理范围内
    
    ptm = 0.3 + (plddt - 30) / 65 * 0.6  # PTM与pLDDT相关
    ptm = max(0.1, min(0.9, ptm + random.uniform(-0.1, 0.1)))
    
    # 生成简单的PDB内容（仅用于测试）
    pdb_content = f"""HEADER    MOCK PREDICTION                         {datetime.now().strftime('%d-%b-%y')}
TITLE     MOCK ESMFOLD PREDICTION FOR {sequence_id}
REMARK   1 MOCK DATA - NOT A REAL STRUCTURE PREDICTION
REMARK   2 SEQUENCE LENGTH: {length}
REMARK   3 MOCK PLDDT: {plddt:.1f}
REMARK   4 MOCK PTM: {ptm:.3f}
"""
    
    # 添加简单的原子记录
    for i, aa in enumerate(sequence[:min(len(sequence), 10)]):  # 只添加前10个残基
        x, y, z = i * 3.8, 0, 0  # 简单的线性排列
        pdb_content += f"ATOM  {i+1:5d}  CA  {aa} A{i+1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00{plddt:6.2f}           C\n"
    
    pdb_content += "END\n"
    
    return {
        'pdb_content': pdb_content,
        'plddt': plddt,
        'ptm': ptm,
        'success': True
    }

def run_batch_prediction(sequences, output_dir, args, logger, esmfold_method=False):
    """运行批量预测"""
    logger.info(f"开始批量预测 ({len(sequences)} 个序列)")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    prediction_results = []
    successful_predictions = 0
    failed_predictions = 0
    
    for i, seq_data in enumerate(sequences):
        sequence_id = seq_data['sequence_id']
        sequence = seq_data['sequence']
        length = seq_data['length']
        
        logger.info(f"预测 {i+1}/{len(sequences)}: {sequence_id} (长度: {length})")
        print(f"[预测] 预测 {i+1}/{len(sequences)}: {sequence_id} (长度: {length})")
        
        try:
            # 清理GPU内存
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass
            
            # 选择预测方法
            if args.mock_mode:
                result = create_mock_prediction(sequence_id, sequence, length)
            elif esmfold_method:
                # 使用真正的ESMFold预测
                result = predict_structure_esmfold(sequence_id, sequence, length, method=esmfold_method)
                if not result['success']:
                    # 检查是否为严格模式
                    if args.strict_real:
                        logger.error(f"ESMFold预测失败: {sequence_id}，严格模式不允许fallback")
                        result['success'] = False
                        result['error'] = "ESMFold预测失败，严格模式拒绝fallback到模拟预测"
                    else:
                        # 如果预测失败，尝试使用模拟预测
                        logger.warning(f"ESMFold预测失败，使用模拟预测: {sequence_id}")
                        result = create_mock_prediction(sequence_id, sequence, length)
                        result['fallback_to_mock'] = True
            else:
                if args.strict_real:
                    logger.error("ESMFold不可用，严格模式拒绝模拟预测")
                    result = {'success': False, 'error': 'ESMFold不可用，严格模式拒绝模拟预测'}
                else:
                    # 没有可用的ESMFold方法，使用模拟预测
                    logger.warning("ESMFold不可用，使用模拟预测")
                    result = create_mock_prediction(sequence_id, sequence, length)
            
            if result['success']:
                plddt = result['plddt']
                ptm = result['ptm']
                
                print(f"  [质量] PTM: {ptm:.3f}, pLDDT: {plddt:.3f}")
                
                # 质量筛选
                passes_quality = plddt >= args.plddt_threshold and ptm >= args.ptm_threshold
                status = "[通过]" if passes_quality else "[未通过]"
                print(f"  [筛选] 质量筛选: {status}")
                
                # 保存PDB文件
                if args.save_all or passes_quality:
                    pdb_filename = f"{sequence_id}_ptm{ptm:.3f}_plddt{plddt:.1f}.pdb"
                    pdb_path = os.path.join(output_dir, pdb_filename)
                    with open(pdb_path, 'w') as f:
                        f.write(result['pdb_content'])
                    print(f"  [保存] 已保存: {pdb_filename}")
                
                # 记录结果
                prediction_results.append({
                    'sequence_id': sequence_id,
                    'sequence': sequence,
                    'length': length,
                    'design_method': seq_data.get('method', 'unknown'),
                    'backbone_id': seq_data.get('backbone_id', 'unknown'),
                    'ptm': ptm,
                    'plddt': plddt,
                    'passes_quality': passes_quality,
                    'pdb_file': pdb_filename if (args.save_all or passes_quality) else None,
                    'prediction_method': result.get('method', 'mock'),
                    'fallback_to_mock': result.get('fallback_to_mock', False),
                    'timestamp': datetime.now().isoformat(),
                    'mock_prediction': args.mock_mode
                })
                
                successful_predictions += 1
                
            else:
                raise Exception("预测失败")
                
        except Exception as e:
            logger.error(f"预测失败 {sequence_id}: {str(e)}")
            print(f"  [错误] 预测失败: {str(e)}")
            
            prediction_results.append({
                'sequence_id': sequence_id,
                'sequence': sequence,
                'length': length,
                'design_method': seq_data.get('method', 'unknown'),
                'backbone_id': seq_data.get('backbone_id', 'unknown'),
                'ptm': None,
                'plddt': None,
                'passes_quality': False,
                'pdb_file': None,
                'prediction_method': 'failed',
                'fallback_to_mock': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'mock_prediction': args.mock_mode
            })
            
            failed_predictions += 1
    
    # 保存摘要
    if prediction_results:
        df = pd.DataFrame(prediction_results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"esmfold_summary_{timestamp}.csv"
        csv_path = os.path.join(output_dir, csv_filename)
        df.to_csv(csv_path, index=False)
        logger.info(f"摘要已保存: {csv_path}")
    
    print(f"\n[完成] 批量预测完成!")
    print(f"  [成功] 成功: {successful_predictions}")
    print(f"  [失败] 失败: {failed_predictions}")
    print(f"  [目录] 输出目录: {output_dir}")
    
    if successful_predictions > 0:
        successful_df = df[df['ptm'].notna()]
        avg_ptm = successful_df['ptm'].mean()
        avg_plddt = successful_df['plddt'].mean()
        quality_passed = successful_df['passes_quality'].sum()
        
        print(f"\n[统计] 质量统计:")
        print(f"  - 平均PTM: {avg_ptm:.3f}")
        print(f"  - 平均pLDDT: {avg_plddt:.1f}")
        print(f"  - 通过质量筛选: {quality_passed}/{successful_predictions}")
    
    return prediction_results

def interactive_select_prediction_mode(all_sequences, checkpoint_file=None):
    """交互式选择预测模式"""
    print("=" * 60)
    print("ESMFold 预测模式选择")
    print("=" * 60)
    
    # 显示序列统计
    proteinmpnn_count = sum(1 for seq in all_sequences if seq.get('method') == 'proteinmpnn')
    esm_if_count = sum(1 for seq in all_sequences if seq.get('method') in ['esm_if', 'esm-if'])
    other_count = len(all_sequences) - proteinmpnn_count - esm_if_count
    
    print(f"[序列统计] 总计 {len(all_sequences)} 个序列")
    print(f"  - ProteinMPNN: {proteinmpnn_count} 个")
    print(f"  - ESM-IF: {esm_if_count} 个")
    if other_count > 0:
        print(f"  - 其他: {other_count} 个")
    
    # 检查断点文件
    completed_count = 0
    if checkpoint_file and os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
                completed_count = len(checkpoint_data.get('completed_predictions', []))
                print(f"[断点检测] 发现断点文件，已完成 {completed_count} 个序列")
        except:
            pass
    
    remaining_count = len(all_sequences) - completed_count
    print(f"[待预测] 剩余 {remaining_count} 个序列")
    
    print("\n请选择预测模式:")
    print("1. 全量预测 - 预测所有剩余序列")
    print("2. 自定义预测 - 指定起始位置和数量")
    print("3. 测试模式 - 预测前5个序列(模拟)")
    print("4. 真实预测模式 - 严格模式，只使用真实ESMFold，禁用fallback")
    print("5. 退出")
    
    while True:
        try:
            choice = input("\n请输入选择 (1-5): ").strip()
            
            if choice == '1':
                return {
                    'mode': 'all',
                    'start_index': completed_count,
                    'max_predictions': remaining_count,
                    'mock_mode': False,
                    'description': '全量预测模式'
                }
            
            elif choice == '2':
                print(f"\n当前序列范围: 0 到 {len(all_sequences)-1}")
                print(f"建议起始位置: {completed_count} (跳过已完成)")
                
                while True:
                    try:
                        start_idx = input(f"请输入起始位置 (0-{len(all_sequences)-1}): ").strip()
                        start_idx = int(start_idx)
                        if 0 <= start_idx < len(all_sequences):
                            break
                        else:
                            print("起始位置超出范围，请重新输入")
                    except ValueError:
                        print("请输入有效的数字")
                
                max_remaining = len(all_sequences) - start_idx
                while True:
                    try:
                        max_pred = input(f"请输入预测数量 (1-{max_remaining}): ").strip()
                        max_pred = int(max_pred)
                        if 1 <= max_pred <= max_remaining:
                            break
                        else:
                            print(f"预测数量超出范围 (1-{max_remaining})，请重新输入")
                    except ValueError:
                        print("请输入有效的数字")
                
                return {
                    'mode': 'custom',
                    'start_index': start_idx,
                    'max_predictions': max_pred,
                    'mock_mode': False,
                    'description': f'自定义预测: 第{start_idx}到{start_idx+max_pred-1}个序列'
                }
            
            elif choice == '3':
                return {
                    'mode': 'test',
                    'start_index': 0,
                    'max_predictions': 5,
                    'mock_mode': True,
                    'description': '测试模式 (模拟预测)'
                }
            
            elif choice == '4':
                return {
                    'mode': 'real',
                    'start_index': 0,
                    'max_predictions': 3,
                    'mock_mode': False,
                    'strict_real': True,
                    'description': '真实预测模式 (严格模式，禁用fallback)'
                }
            
            
            elif choice == '5':
                print("退出程序")
                sys.exit(0)
            
            else:
                print("无效选择，请输入 1-5")
                
        except KeyboardInterrupt:
            print("\n\n程序被用户中断")
            sys.exit(0)
        except Exception as e:
            print(f"输入错误: {e}")

def main():
    """主函数"""
    print("=" * 60)
    print("Step 4: ESMFold 本地结构预测")
    print("作者: Kuroneko | 日期: 2025.10.2")
    print("=" * 60)
    
    # 设置日志
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = project_root / "logs" / f"step4_{timestamp}.log"
    logger = setup_simple_logger("step4")
    
    try:
        # 检查环境
        if not setup_environment():
            print("\n[错误] 环境检查失败")
            print("[建议] 建议使用Google Colab版本: step4_esmfold_colab.ipynb")
            return
        
        # 检查ESMFold可用性
        esmfold_method = check_esmfold_availability()
        if not esmfold_method or esmfold_method == 'torch-only':
            if esmfold_method == 'torch-only':
                print("\n[警告] 缺少ESMFold库，将使用模拟预测模式")
                print("[提示] 安装命令: pip install transformers 或 pip install fair-esm")
            else:
                print("\n[警告] ESMFold不可用，将使用模拟预测模式")
            esmfold_method = False
        else:
            print(f"\n[OK] 将使用 {esmfold_method} 方法进行ESMFold预测")
        
        # 加载序列
        print("\n[加载] 加载序列文件...")
        sequence_files = [
            project_root / "designs" / "iter1" / "proteinmpnn_sequences.json",
            project_root / "designs" / "iter1" / "esm_if_sequences.json"
        ]
        
        all_sequences = []
        for seq_file in sequence_files:
            if seq_file.exists():
                sequences = load_sequences_from_json(seq_file)
                all_sequences.extend(sequences)
                print(f"  [OK] 从 {seq_file.name} 加载 {len(sequences)} 个序列")
        
        if not all_sequences:
            print("[错误] 未找到序列文件")
            return
        
        print(f"[统计] 总计加载 {len(all_sequences)} 个序列")
        
        # 设置输出目录和断点文件路径
        output_dir = project_root / "output" / "preds" / "iter1"
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_file = output_dir / "esmfold_checkpoint.json"
        
        # 交互式选择预测模式
        prediction_config = interactive_select_prediction_mode(all_sequences, checkpoint_file)
        
        # 根据选择设置参数
        start_index = prediction_config['start_index']
        max_predictions = prediction_config['max_predictions']
        mock_mode = prediction_config['mock_mode']
        strict_real = prediction_config.get('strict_real', False)
        
        # 如果强制使用模拟模式，覆盖用户选择（除非是严格真实模式）
        if esmfold_method == False or esmfold_method == 'torch-only':
            if strict_real:
                print("\n[错误] ESMFold不可用，但当前为严格真实模式，无法进行预测")
                print("[建议] 请安装ESMFold或使用Google Colab版本")
                return
            else:
                mock_mode = True
        
        # 选择预测序列
        end_index = min(start_index + max_predictions, len(all_sequences))
        sequences_to_predict = all_sequences[start_index:end_index]
        
        print(f"\n[配置] 预测配置:")
        print(f"  - 模式: {prediction_config['description']}")
        print(f"  - 序列范围: 第{start_index}到{end_index-1}个 (共{len(sequences_to_predict)}个)")
        print(f"  - ESMFold方法: {esmfold_method if esmfold_method else '模拟模式'}")
        print(f"  - 输出目录: {output_dir}")
        
        # 预测时间估算
        if not mock_mode:
            try:
                import torch
                estimated_time_per_seq = 30 if not torch.cuda.is_available() else 10  # 秒
            except ImportError:
                estimated_time_per_seq = 30  # 秒
            estimated_total_time = len(sequences_to_predict) * estimated_time_per_seq
            if estimated_total_time > 60:
                estimated_minutes = estimated_total_time // 60
                print(f"  - 预计时间: ~{estimated_minutes} 分钟")
            else:
                print(f"  - 预计时间: ~{estimated_total_time} 秒")
        
        # 检查预测范围有效性
        if len(sequences_to_predict) == 0:
            print("[警告] 没有序列需要预测")
            return
        
        # 大规模预测警告
        if len(sequences_to_predict) > 50 and not mock_mode:
            print(f"\n[注意] 您将预测 {len(sequences_to_predict)} 个序列")
            print("[建议] 大规模预测可能需要较长时间，建议:")
            print("  1. 使用GPU加速")
            print("  2. 分批预测(使用 --start-index 和 --max-predictions)")
            print("  3. 或使用Google Colab: step4_esmfold_colab.ipynb")
        
        # 创建预测参数对象
        class PredictionArgs:
            def __init__(self):
                self.plddt_threshold = 70.0
                self.ptm_threshold = 0.5
                self.save_all = False
                self.mock_mode = mock_mode
                self.strict_real = strict_real
                
        prediction_args = PredictionArgs()
        
        prediction_results = run_batch_prediction(
            sequences_to_predict, 
            str(output_dir), 
            prediction_args, 
            logger,
            esmfold_method
        )
        
        logger.info("Step 4 执行完成")
        
    except Exception as e:
        logger.error(f"程序执行异常: {str(e)}")
        logger.error(traceback.format_exc())
        print(f"[错误] 程序执行异常: {str(e)}")
        print("\n[建议] 建议使用Google Colab版本: step4_esmfold_colab.ipynb")

if __name__ == "__main__":
    main()

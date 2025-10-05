#!/usr/bin/env python3
"""
单序列AlphaFold2预测测试工具
基于官方ColabFold notebook的简化版本
用户输入单条序列，完整下载和处理所有必要文件
"""

import os
import re
import hashlib
import random
from datetime import datetime

# 添加辅助函数到生成代码中

def generate_single_sequence_test_code(config_data):
    """生成单序列测试代码"""
    
    code = f'''# ============================================================================
# ColabFold单序列测试工具 v1.0
# 基于官方notebook的简化版本
# 生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
# ============================================================================

print("="*80)
print("ColabFold单序列AlphaFold2预测测试")
print("="*80)

# ============================================================================
# 辅助函数定义
# ============================================================================

def add_hash(jobname, sequence):
    """添加序列哈希到jobname"""
    import hashlib
    return jobname + "_" + hashlib.sha1(sequence.encode()).hexdigest()[:5]

def sanitize_jobname(name):
    """清理job名称"""
    import re
    name = "".join(name.split())
    name = re.sub(r'[^a-zA-Z0-9_-]', '', name)
    return name

# ============================================================================
# 1. 导入必要的库和依赖安装
# ============================================================================

# 安装ColabFold（如果还没有）
print("\\n=== 步骤1: 安装ColabFold ===")
import os
import subprocess

try:
    import colabfold
    print("✅ ColabFold已安装")
except ImportError:
    print("📦 安装ColabFold...")
    subprocess.run(["pip", "install", "colabfold[alphafold-minus-jax]", "-q"], check=True)
    print("✅ 安装完成")

# 导入关键库
try:
    from colabfold.batch import get_queries, run, set_model_type
    from colabfold.download import download_alphafold_params
    import concurrent.futures
    
    # TensorFlow/JAX设置
    import os
    os.environ["TF_FORCE_UNIFIED_MEMORY"] = "1"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.0"
    
    print("✅ 依赖库导入成功")
except Exception as e:
    print(f"❌ 导入失败: {{e}}")
    raise

# ============================================================================
# 2. 用户输入和序列处理
# ============================================================================

print("\\n=== 步骤2: 序列输入和处理 ===")

# 用户输入序列（Colab友好版本）
print("请输入要预测的蛋白质序列:")
print("示例: PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK")
print("\\n支持:")
print("- 单体序列: 直接输入序列")
print("- 复合物: 用冒号分隔链 (如: SEQ1:SEQ2:SEQ3)")
print("- 如果不输入，将使用默认测试序列")
print("\\n请在下面修改 query_sequence 变量:")

# 用户需要在这里修改序列
query_sequence = """MKHQFGCLTVKLMLWGFHVLKRLQGGNFIYQKQSPQYVQHLDLQKNKLKALVLWQDKQGQVIGTEFDDSLKKEQMQSGAHGMDLISRLKNQIQVVKEGSTDNLLQYKQDLFQVKKQLKLEKDDGLQSQDTKLKKILNAMAEKILNLLKELNQDQTQQKLIELNKEKQDLQLQDKQAQQEKQQLKYLKQLIDELNKNNKQLKELNKQILKEQKKNLQLQKKQILEQKKKQDLKEQKKNQQQLKLLNEQADKLEQLQQQEKQKDLQLEQKQKQ"""

# 检查是否为空或默认值
if query_sequence.strip() == "" or "MKHQFGCLTVKLMLWGFHVLKRLQGG" in query_sequence:
    print("\\n⚠️  检测到可能未修改的默认序列")
    print("📝 请在上方修改 query_sequence 变量为您要预测的序列")
    print("💡 或者直接使用默认序列进行测试")
    
    # 直接使用简短测试序列（避免Colab input问题）
    print("\\n🔧 自动使用简洁测试序列")
    query_sequence = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIKDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITLGMDELYK"
    print("✅ 使用绿色荧光蛋白(GFP)测试序列 - 239残基")

# 清理序列（移除空格）
query_sequence = "".join(query_sequence.split())

# 创建jobname
jobname = f"test_{{datetime.now().strftime('%Y%m%d_%H%M%S')}}"
jobname = add_hash(jobname, query_sequence)

print(f"\\n序列信息:")
print(f"- 长度: {{len(query_sequence)}} 残基")
print(f"- Job名称: {{jobname}}")
print(f"- 序列预览: {{query_sequence[:50]}}...")

if ':' in query_sequence:
    print("\\n检测到多链复合物!")
    chains = query_sequence.split(':')
    print(f"- 链数: {{len(chains)}}")
    for i, chain in enumerate(chains, 1):
        print(f"  链{{i}}: {{len(chain)}} 残基")

# ============================================================================
# 3. AlphaFold参数配置
# ============================================================================

print("\\n=== 步骤3: AlphaFold参数配置 ===")

# 使用配置文件中的参数
PARAMS = {config_data}
num_relax = PARAMS['num_relax']
num_recycles = PARAMS['num_recycles'] 
msa_mode = PARAMS['msa_mode']
template_mode = PARAMS['template_mode']
model_type = PARAMS['model_type']
max_msa = PARAMS['max_msa']
pair_mode = PARAMS['pair_mode']
save_detail = PARAMS['save_all']
plddt_threshold = PARAMS['plddt_threshold']

print(f"预测参数:")
print(f"- 结构松弛: {{num_relax}} 个模型")
print(f"- 循环次数: {{num_recycles}}")
print(f"- MSA模式: {{msa_mode}}")
print(f"- 模板模式: {{template_mode}}")
print(f"- 模型类型: {{model_type}}")
print(f"- 最大MSA: {{max_msa}}")
print(f"- 质量阈值: {{plddt_threshold}}")

# Amber松弛开关
use_amber = num_relax > 0

# ============================================================================
# 4. 创建作业目录
# ============================================================================

print("\\n=== 步骤4: 准备作业目录 ===")

# 创建输出目录
output_dir = f"./{{jobname}}"
os.makedirs(output_dir, exist_ok=True)

print(f"输出目录: {{output_dir}}")

# 创建CSV序列文件
csv_path = os.path.join(output_dir, f"{{jobname}}.csv")
with open(csv_path, "w") as f:
    f.write(f"id,sequence\\n{{jobname}},{{query_sequence}}")

print(f"序列文件: {{csv_path}}")

# ============================================================================
# 5. 下载AlphaFold参数
# ============================================================================

print("\\n=== 步骤5: 下载AlphaFold参数 ===")
print("这可能需要几分钟时间...")

# 先检测用户参数设置，再下载
model_type_settings = PARAMS.get('model_type', 'auto')
if model_type_settings == 'auto':
    # 根据序列长度自动判断模型类型
    param_model_type = 'alphafold2_ptm' if len(query_sequence) < 1000 else 'alphafold2_multimer_v3'
    print(f"自动选择模型类型: {{param_model_type}} (序列长度: {{len(query_sequence)}})")
else:
    param_model_type = model_type_settings
    print(f"使用指定模型类型: {{param_model_type}}")

# 下载参数到当前目录（参照官方实现）
data_dir = Path(".")
print(f"下载路径: {{data_dir}}")

try:
    download_alphafold_params(param_model_type, data_dir)
    print("✅ AlphaFold参数下载完成")
    
    # 检查参数文件是否存在
    param_files = list(data_dir.glob("params/*.npz"))
    if param_files:
        print(f"✅ 找到参数文件: {{len(param_files)}} 个")
        print(f"示例文件名: {{param_files[0].name}}")
    else:
        print("⚠️  未找到参数文件，但继续尝试预测...")
        
except Exception as e:
    print(f"❌ 参数下载失败: {{e}}")
    print("请检查网络连接后重试")
    print("常见解决方案:")
    print("1. 检查Colab运行时的网络连接")
    print("2. Runtime -> Restart runtime 后重试")
    print("3. 尝试使用不同的AlphaFold模型")
    raise

# ============================================================================
# 6. 获取查询和设置模型
# ============================================================================

print("\\n=== 步骤6: 设置预测模型 ===")

# 获取查询数据
queries, is_complex = get_queries(csv_path)

print(f"查询信息:")
print(f"- 查询数量: {{len(queries)}}")
print(f"- 复合物模式: {{is_complex}}")

# 设置模型类型
actual_model_type = set_model_type(is_complex, model_type)
print(f"- 实际使用模型: {{actual_model_type}}")

# ============================================================================
# 7. 执行AlphaFold2预测
# ============================================================================

print("\\n=== 步骤7: 执行AlphaFold2预测 ===")
print(f"预测参数: {{len(query_sequence)}} 残基序列")

start_time = datetime.now()
print(f"开始时间: {{start_time.strftime('%H:%M:%S')}}")

# 设置cluster_profile逻辑（参照官方代码）
if "multimer" in actual_model_type and max_msa is not None and max_msa != 'auto':
    use_cluster_profile = False
else:
    use_cluster_profile = True

# 处理max_msa参数，将'auto'转换为None
processed_max_msa = None
if max_msa and max_msa != 'auto':
    processed_max_msa = max_msa
else:
    processed_max_msa = None

# 处理num_recycles参数，将'auto'转换为数值
if num_recycles == 'auto' or num_recycles is None:
    if "multimer" in actual_model_type:
        processed_num_recycles = 20  # 复合物默认20次
    else:
        processed_num_recycles = 3   # 单体默认3次
else:
    processed_num_recycles = int(num_recycles)

try:
    # 执行预测（完全按照官方notebook的参数）
    results = run(
        queries=queries,
        result_dir=output_dir,
        use_templates=False,
        custom_template_path=None,
        num_relax=num_relax if use_amber else 0,
        msa_mode=msa_mode,
        model_type=actual_model_type,
        num_models=5,
        num_recycles=processed_num_recycles,
        relax_max_iterations=200,
        recycle_early_stop_tolerance=0.0,
        num_seeds=1,
        use_dropout=False,
        model_order=[1,2,3,4,5],
        is_complex=is_complex,
        data_dir=Path("."),
        keep_existing_results=False,
        rank_by="auto",
        pair_mode=pair_mode if is_complex else "unpaired",
        pairing_strategy="greedy",
        stop_at_score=float(100),
        dpi=200,
        zip_results=False,
        save_all=save_detail in ['detailed', 'comprehension'],
        max_msa=processed_max_msa,
        use_cluster_profile=use_cluster_profile,
        input_features_callback=lambda x: None,
        prediction_callback=lambda *args: None,
        use_folding_cache=False
    )
    
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"\\n✅ 预测完成!")
    print(f"耗时: {{duration.seconds}} 秒")

except Exception as e:
    print(f"\\n❌ 预测失败: {{e}}")
    if 'jax.core' in str(e).lower() or 'jax' in str(e).lower():
        print("\\n🔄 检测到JAX错误，建议:")
        print("1. Runtime -> Restart runtime")
        print("2. 重新运行此notebook")
        raise RuntimeError("JAX环境错误，需要重启runtime")
    else:
        print(f"\\n📋 完整错误信息:")
        print(f"{{e}}")
        raise

# ============================================================================
# 8. 分析预测结果
# ============================================================================

print("\\n=== 步骤8: 分析预测结果 ===")

# 查找结果文件
import glob

# 查找JSON结果文件
json_files = glob.glob(os.path.join(output_dir, "*.json"))
pdb_files = glob.glob(os.path.join(output_dir, "*.pdb"))

print(f"生成的文件:")
print(f"- JSON结果: {{len(json_files)}} 个")
print(f"- PDB结构: {{len(pdb_files)}} 个")

if json_files:
    # 读取scores.json
    scores_file = os.path.join(output_dir, f"{{jobname}}_scores.json")
    if os.path.exists(scores_file):
        import json
        with open(scores_file, 'r') as f:
            scores_data = json.load(f)
        
        print(f"\\n预测分数:")
        for i, model in enumerate(scores_data.get('plddt', []), 1):
            avg_plddt = sum(model) / len(model)
            print(f"- 模型{{i}}: pLDDT = {{avg_plddt:.2f}}")
        
        # 找到最佳模型
        best_model_idx = 0
        best_score = max([sum(score) / len(score) for score in scores_data.get('plddt', [])])
        print(f"\\n🏆 最佳模型: 模型{{best_model_idx + 1}} (pLDDT: {{best_score:.2f}})")

# ============================================================================
# 9. 可视化结果
# ============================================================================

print("\\n=== 步骤9: 生成可视化 ===")

try:
    import matplotlib.pyplot as plt
    import numpy as np
    
    # 如果有scores数据，绘制pLDDT图
    if 'scores_data' in locals() and 'plddt' in scores_data:
        fig, axes = plt.subplots(1, min(5, len(scores_data['plddt'])), figsize=(12, 4))
        if len(scores_data['plddt']) == 1:
            axes = [axes]
        
        for i, plddt_scores in enumerate(scores_data['plddt'][:5]):
            axes[i].plot(plddt_scores, alpha=0.8)
            axes[i].set_title(f'模型{{i+1}}')
            axes[i].set_xlabel('残基位置')
            axes[i].set_ylabel('pLDDT分数')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'plddt_plots.png'), dpi=150, bbox_inches='tight')
        print("✅ pLDDT图表已保存")
        
except Exception as e:
    print(f"⚠️ 可视化生成失败: {{e}}")

# ============================================================================
# 10. 保存结果和下载
# ============================================================================

print("\\n=== 步骤10: 打包和下载结果 ===")

# 创建结果包
import zipfile

zip_name = f"{{jobname}}_results.zip"
with zipfile.ZipFile(zip_name, 'w') as zipf:
    # 添加所有结果文件
    for file_path in glob.glob(os.path.join(output_dir, "*")):
        if os.path.isfile(file_path):
            zipf.write(file_path, os.path.basename(file_path))

print(f"✅ 结果包已创建: {{zip_name}}")

# 尝试下载
try:
    from google.colab import files
    files.download(zip_name)
    print(f"✅ 文件已下载: {{zip_name}}")
except ImportError:
    print("⚠️ 不在Colab环境中，请手动下载文件")

# ============================================================================
# 完成总结
# ============================================================================

print("\\n" + "="*80)
print("🎉 单序列AlphaFold2预测完成!")
print("="*80)
print(f"Job名称: {{jobname}}")
print(f"序列长度: {{len(query_sequence)}} 残基")
print(f"输出目录: {{output_dir}}")
print(f"结果包: {{zip_name}}")
print("\\n📋 生成的文件:")
print("- PDB结构文件 (*.pdb)")
print("- 预测分数 (*.json)")
if 'scores_data' in locals():
    best_score = max([sum(score) / len(score) for score in scores_data.get('plddt', [])])
    print(f"- 最佳预测质量: {{best_score:.2f}} pLDDT")
print("- 可视化图表 (*.png)")
print("- 压缩结果包 (*.zip)")
print("\\n✅ 完成后请检查预测质量，合理的pLDDT分数应 > 70")
print("="*80)
'''

    return code

def main():
    """主函数"""
    print("="*80)
    print("单序列AlphaFold2预测代码生成器")
    print("="*80)
    
    # 配置参数（简化版）
    config_data = {
        "num_relax": 1,
        "num_recycles": "auto",
        "msa_mode": "mmseqs2_uniref_env", 
        "template_mode": "none",
        "model_type": "auto",
        "max_msa": "auto",
        "pair_mode": "unpaired_paired",
        "save_all": "detailed",
        "plddt_threshold": 70
    }
    
    print("\n📝 配置参数:")
    print("- 结构松弛: 1个模型")
    print("- MSA模式: MMseqs2 + UniRef + 环境序列")
    print("- 模板模式: 无模板(从头预测)")
    print("- 模型类型: 自动选择")
    print("- 质量阈值: 70")
    
    print(f"\n🔄 生成代码...")
    
    # 生成代码
    generated_code = generate_single_sequence_test_code(config_data)
    
    # 保存文件
    output_path = "F:\\Project\\蛋白质设计\\designs\\iter1\\single_sequence_test.py"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(generated_code)
    
    print(f"✅ 代码已保存到: {output_path}")
    print(f"\n📋 使用方法:")
    print(f"1. 上传 {output_path} 到Google Colab")
    print(f"2. 运行notebook")
    print(f"3. 输入要预测的蛋白质序列")
    print(f"4. 等待预测完成后下载结果")
    
    # 也保存配置
    config_path = "F:\\Project\\蛋白质设计\\designs\\iter1\\single_sequence_config.json"
    import json
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 配置已保存到: {config_path}")
    print(f"\n🎉 单序列测试工具生成完成!")
    print("="*80)

if __name__ == "__main__":
    main()

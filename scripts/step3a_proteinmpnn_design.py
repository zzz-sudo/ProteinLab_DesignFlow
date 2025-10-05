"""
脚本名: step3a_proteinmpnn_design.py
作者: Kuroneko
日期: 2025.9.30

功能: 使用 ProteinMPNN 对骨架进行序列设计（基于ColabDesign封装版本）

输入文件:
- backbones/iterN/backbone_*.pdb (Step2 生成的骨架)
- config.json (全局配置)

输出文件:
- designs/iterN/proteinmpnn_sequences.json (设计的序列)
- designs/iterN/proteinmpnn_colab.py (Colab运行代码)
- logs/step3a_YYYYMMDD_HHMMSS.log (执行日志)

运行示例:
python scripts/step3a_proteinmpnn_design.py

依赖: 
- 需要 step2 完成的骨架文件
- 在 Google Colab 中运行 ColabDesign版本的ProteinMPNN

参数示例:
- num_sequences: 8 (1-64)
- sampling_temperature: 0.1 (0.0001-1.0)
- remove_amino_acids: "C" (禁用的氨基酸)
"""

import os
import sys
import json
import datetime
from pathlib import Path
from typing import Dict

# 导入工具模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
try:
    from utils_io import (
        get_project_root, ensure_dir, get_abs_path, setup_logger, 
        validate_input, save_config, load_config, get_iteration_dir
    )
except ImportError:
    print("错误: 无法导入 utils_io.py，请确保文件存在")
    sys.exit(1)

def generate_proteinmpnn_colab_code(params: Dict, output_dir: Path) -> str:
    """基于designs/proteinMPNN.txt生成ColabDesign版本的ProteinMPNN代码"""
    
    colab_code = f'''# ProteinMPNN 序列设计 - 基于ColabDesign封装版本
# 参数: 每个骨架生成{params['num_sequences']}个序列，采样温度{params['temperature']}

# ===== 单元格1: 安装ColabDesign环境 =====
import os, time, signal
import sys, random, string, re

# 下载参数文件（后台下载）
if not os.path.isdir("params"):
    !apt-get install aria2
    !mkdir params
    !aria2c -q -x 16 https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar &
    !tar -xf alphafold_params_2022-12-06.tar -C params &
    !touch params/done.txt &

# 安装ColabDesign
if not os.path.isdir("colabdesign"):
    print("installing ColabDesign...")
    !pip -q install git+https://github.com/sokrypton/ColabDesign.git@v1.1.1
    !ln -s /usr/local/lib/python3.*/dist-packages/colabdesign colabdesign

print("✓ ColabDesign环境准备完成")

# ===== 单元格2: 上传骨架文件 =====
from google.colab import files
print("请上传您的骨架PDB文件（可多选）:")
uploaded = files.upload()

# 创建输出目录
!mkdir -p outputs
pdb_files = list(uploaded.keys())

# 保存上传的文件到outputs目录，重命名为标准格式
for i, filename in enumerate(pdb_files):
    output_name = f"outputs/backbone_{{i:04d}}.pdb"
    with open(output_name, "wb") as f:
        f.write(uploaded[filename])

print(f"已处理 {{len(pdb_files)}} 个骨架文件")
!ls -la outputs/backbone_*.pdb

# ===== 单元格3: 批量运行ProteinMPNN设计 =====
# 等待参数下载完成
if not os.path.isfile("params/done.txt"):
    print("等待AlphaFold参数下载...")
    while not os.path.isfile("params/done.txt"):
        time.sleep(5)

# 设置ProteinMPNN参数
num_seqs = {params['num_sequences']}
mpnn_sampling_temp = {params['temperature']}
rm_aa = "{params.get('remove_amino_acids', '')}"
num_recycles = 1

print(f"ProteinMPNN设计参数:")
print(f"- 每个骨架序列数: {{num_seqs}}")
print(f"- 采样温度: {{mpnn_sampling_temp}}")
print(f"- 禁用氨基酸: {{rm_aa}}")
print(f"- AlphaFold验证轮数: {{num_recycles}}")

# 批量处理所有骨架
all_results = {{}}

for i in range(len(pdb_files)):
    backbone_id = f"backbone_{{i:04d}}"
    print(f"\\n处理骨架 {{i+1}}/{{len(pdb_files)}}: {{backbone_id}}")
    
    # 构建ColabDesign命令参数
    opts = [
        f"--pdb=outputs/{{backbone_id}}.pdb",
        f"--loc=outputs/{{backbone_id}}",
        f"--contig={params.get('contigs_format', '100-100')}",  # 使用用户选择的contigs格式
        f"--copies=1",
        f"--num_seqs={{num_seqs}}",
        f"--num_recycles={{num_recycles}}",
        f"--rm_aa={{rm_aa}}",
        f"--mpnn_sampling_temp={{mpnn_sampling_temp}}",
        f"--num_designs=1"
    ]
    
    opts_str = ' '.join(opts)
    
    # 运行ColabDesign的ProteinMPNN（这是关键调用）
    !python colabdesign/rf/designability_test.py {{opts_str}}
    
    print(f"  ✓ {{backbone_id}} ProteinMPNN设计完成")

print("\\n所有骨架的ProteinMPNN设计完成！")

# ===== 单元格4: 收集和整合结果 =====
import glob
import json

all_sequences = {{}}

# 收集每个骨架的设计结果
for i in range(len(pdb_files)):
    backbone_id = f"backbone_{{i:04d}}"
    result_dir = f"outputs/{{backbone_id}}"
    
    if os.path.exists(f"{{result_dir}}/best.pdb"):
        sequences = []
        
        # 查找序列文件
        seq_files = glob.glob(f"{{result_dir}}/seqs/*.fa")
        
        for seq_file in seq_files:
            try:
                with open(seq_file, 'r') as f:
                    lines = f.read().strip().split('\\n')
                    if len(lines) >= 2:
                        seq_id = lines[0].replace('>', '')
                        sequence = lines[1]
                        
                        sequences.append({{
                            "sequence_id": seq_id,
                            "sequence": sequence,
                            "length": len(sequence),
                            "method": "proteinmpnn_colabdesign"
                        }})
            except:
                continue
        
        all_sequences[backbone_id] = {{
            "backbone_file": f"{{backbone_id}}.pdb",
            "backbone_id": backbone_id,
            "design_method": "proteinmpnn",
            "parameters": {{
                "num_sequences": num_seqs,
                "sampling_temp": mpnn_sampling_temp,
                "remove_aa": rm_aa,
                "num_recycles": num_recycles
            }},
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "sequences": sequences
        }}
        
        print(f"{{backbone_id}}: {{len(sequences)}} 个序列")

# 保存最终结果
with open("proteinmpnn_sequences.json", "w") as f:
    json.dump(all_sequences, f, indent=2)

# 统计
total_sequences = sum(len(data["sequences"]) for data in all_sequences.values())
print(f"\\n最终统计:")
print(f"成功处理: {{len(all_sequences)}} 个骨架")
print(f"总序列数: {{total_sequences}} 个")

# ===== 单元格5: 下载结果 =====
!zip -r proteinmpnn_design_results.zip outputs/ proteinmpnn_sequences.json

from google.colab import files
files.download('proteinmpnn_design_results.zip')

print("下载完成！解压到: F:/Project/蛋白质设计/designs/iter1/")
'''
    
    colab_file = output_dir / "proteinmpnn_colab.py"
    with open(colab_file, 'w', encoding='utf-8') as f:
        f.write(colab_code)
    
    return str(colab_file)

def main():
    """主函数"""
    print("=" * 60)
    print("Step 3A: ProteinMPNN 序列设计（ColabDesign版本）")
    print("作者: Kuroneko | 日期: 2025.9.30")
    print("=" * 60)
    
    logger = setup_logger("step3a")
    start_time = datetime.datetime.now()
    
    try:
        # 加载配置
        config = load_config()
        current_iter = config.get("current_iteration", 1)
        
        print(f"当前迭代: iter{current_iter}")
        
        # 检查是否已存在ProteinMPNN结果
        designs_root = get_abs_path("designs")
        zip_files = list(designs_root.glob("*proteinmpnn*results*.zip"))
        
        if zip_files:
            print(f"发现已存在的ProteinMPNN结果: {zip_files[0].name}")
            response = input("是否自动解压并解析结果? (y/n) (默认: y): ").strip().lower()
            if response in ['', 'y', 'yes']:
                print("自动解压ProteinMPNN结果...")
                try:
                    import zipfile
                    with zipfile.ZipFile(zip_files[0], 'r') as zip_ref:
                        zip_ref.extractall(designs_root)
                    print(f"解压完成: {zip_files[0].name}")
                    
                    # 运行解析脚本
                    parse_script = get_abs_path("scripts/parse_proteinmpnn_results.py")
                    if parse_script.exists():
                        print("运行ProteinMPNN结果解析...")
                        import subprocess
                        result = subprocess.run([sys.executable, str(parse_script)], 
                                              capture_output=True, text=True)
                        if result.returncode == 0:
                            print("ProteinMPNN结果解析完成")
                            print("可以直接使用Step 3D进行统一预测")
                            return True
                        else:
                            print(f"解析失败: {result.stderr}")
                except Exception as e:
                    print(f"解压失败: {e}")
        
        # 查找骨架文件
        backbone_dir = get_abs_path(f"backbones/iter{current_iter}")
        if not backbone_dir.exists():
            print(f"错误: 骨架目录不存在: {backbone_dir}")
            print("请先运行 step2 生成骨架")
            return False
        
        # 寻找RFdiffusion生成的骨架文件
        # 只选择主骨架文件（排除轨迹文件）
        all_files = list(backbone_dir.glob("protein_backbone_*.pdb"))
        backbone_files = [f for f in all_files if not ("traj" in f.name or "pX0" in f.name)]
        if not backbone_files:
            print("错误: 未找到骨架文件")
            print(f"在目录: {backbone_dir}")
            print(f"查找模式: protein_backbone_*.pdb")
            print("找到的文件:")
            for f in backbone_dir.glob("*.pdb"):
                print(f"  {f.name}")
            return False
        
        print(f"找到 {len(backbone_files)} 个骨架文件")
        
        # 获取参数
        print("\n请输入ProteinMPNN设计参数:")
        
        num_sequences = validate_input(
            "每个骨架设计序列数量",
            int,
            valid_range=(1, 64),
            default_value=8
        )
        
        sampling_temp = validate_input(
            "采样温度",
            float,
            valid_range=(0.0001, 1.0),
            default_value=0.1
        )
        
        # 功能特性控制
        print("\n功能特性控制:")
        print("1. 亲水性蛋白 (富含极性氨基酸)")
        print("2. 疏水性蛋白 (富含疏水氨基酸)")
        print("3. 金属结合蛋白 (富含H,C,D,E)")
        print("4. 酶类蛋白 (富含催化残基)")
        print("5. 自定义")
        
        function_type = validate_input(
            "选择功能类型 (1-5)",
            int,
            valid_range=(1, 5),
            default_value=1
        )
        
        if function_type == 1:  # 亲水性
            remove_aa = "A,I,L,M,F,W,Y,V"  # 禁用疏水氨基酸
            print(" 设置为亲水性蛋白设计")
        elif function_type == 2:  # 疏水性
            remove_aa = "S,T,N,Q,K,R,D,E,G"  # 禁用亲水氨基酸
            print(" 设置为疏水性蛋白设计")
        elif function_type == 3:  # 金属结合
            remove_aa = "A,I,L,M,F,W,Y,V,S,T,N,Q"  # 保留H,C,D,E,K,R
            print(" 设置为金属结合蛋白设计")
        elif function_type == 4:  # 酶类
            remove_aa = "A,I,L,M,F,W,Y,V"  # 保留催化残基
            print(" 设置为酶类蛋白设计")
        else:  # 自定义
            custom_input = validate_input(
                "禁用的氨基酸类型 (如: C 表示禁用半胱氨酸, 空表示不禁用)",
                str,
                default_value="C"
            )
            remove_aa = custom_input if custom_input else ""
        
        # 新增：contigs格式选择
        print("\nContigs格式选择:")
        print("1. 固定长度 (如: 100-100)")
        print("2. 可变长度 (如: 80-120)")
        print("3. 自定义格式 (如: A1-50/60-110)")
        
        contigs_mode = validate_input(
            "选择contigs模式",
            int,
            valid_range=(1, 3),
            default_value=1
        )
        
        if contigs_mode == 1:
            # 固定长度模式
            target_length = config["parameters"].get("target_length", 100)
            contigs_format = f"{target_length}-{target_length}"
        elif contigs_mode == 2:
            # 可变长度模式
            min_length = validate_input("最小长度", int, valid_range=(30, 200), default_value=80)
            max_length = validate_input("最大长度", int, valid_range=(min_length, 300), default_value=120)
            contigs_format = f"{min_length}-{max_length}"
        else:
            # 自定义格式
            contigs_format = validate_input(
                "输入自定义contigs格式 (如: A1-50/60-110)",
                str,
                default_value="100-100"
            )
        
        max_backbones = validate_input(
            "最大处理骨架数",
            int,
            valid_range=(1, len(backbone_files)),
            default_value=min(10, len(backbone_files))
        )
        
        # 设置输出目录（自动创建iter目录）
        output_dir = get_iteration_dir("step3a", current_iter)
        print(f"输出目录: {output_dir}")
        
        # 参数
        params = {
            "num_sequences": num_sequences,
            "temperature": sampling_temp,
            "remove_amino_acids": remove_aa,
            "max_backbones": max_backbones,
            "contigs_format": contigs_format,  # 新增：用户选择的contigs格式
            "target_length": config["parameters"].get("target_length", 100),
            "iteration": current_iter,
            "seed": config["parameters"].get("rfdiffusion_seed", 42)
        }
        
        # 生成Colab代码
        colab_file = generate_proteinmpnn_colab_code(params, output_dir)
        print(f"\nProteinMPNN Colab代码已生成: {colab_file}")
        
        print("\n" + "=" * 50)
        print("使用说明:")
        print("1. 复制Colab代码到Google Colab")
        print("2. 分5个单元格运行")
        print("3. 上传您的骨架PDB文件")
        print("4. 下载proteinmpnn_design_results.zip")
        print("5. 解压到designs/iter{}/".format(current_iter))
        print("=" * 50)
        
        return True
        
    except Exception as e:
        logger.error(f"异常: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

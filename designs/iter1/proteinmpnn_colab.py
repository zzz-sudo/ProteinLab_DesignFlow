# ProteinMPNN 序列设计 - 基于ColabDesign封装版本
# 参数: 每个骨架生成8个序列，采样温度0.1

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
    output_name = f"outputs/backbone_{i:04d}.pdb"
    with open(output_name, "wb") as f:
        f.write(uploaded[filename])

print(f"已处理 {len(pdb_files)} 个骨架文件")
!ls -la outputs/backbone_*.pdb

# ===== 单元格3: 批量运行ProteinMPNN设计 =====
# 等待参数下载完成
if not os.path.isfile("params/done.txt"):
    print("等待AlphaFold参数下载...")
    while not os.path.isfile("params/done.txt"):
        time.sleep(5)

# 设置ProteinMPNN参数
num_seqs = 8
mpnn_sampling_temp = 0.1
rm_aa = "A,I,L,M,F,W,Y,V"
num_recycles = 1

print(f"ProteinMPNN设计参数:")
print(f"- 每个骨架序列数: {num_seqs}")
print(f"- 采样温度: {mpnn_sampling_temp}")
print(f"- 禁用氨基酸: {rm_aa}")
print(f"- AlphaFold验证轮数: {num_recycles}")

# 批量处理所有骨架
all_results = {}

for i in range(len(pdb_files)):
    backbone_id = f"backbone_{i:04d}"
    print(f"\n处理骨架 {i+1}/{len(pdb_files)}: {backbone_id}")
    
    # 构建ColabDesign命令参数
    opts = [
        f"--pdb=outputs/{backbone_id}.pdb",
        f"--loc=outputs/{backbone_id}",
        f"--contig=100-100",  # 使用用户选择的contigs格式
        f"--copies=1",
        f"--num_seqs={num_seqs}",
        f"--num_recycles={num_recycles}",
        f"--rm_aa={rm_aa}",
        f"--mpnn_sampling_temp={mpnn_sampling_temp}",
        f"--num_designs=1"
    ]
    
    opts_str = ' '.join(opts)
    
    # 运行ColabDesign的ProteinMPNN（这是关键调用）
    !python colabdesign/rf/designability_test.py {opts_str}
    
    print(f"  ✓ {backbone_id} ProteinMPNN设计完成")

print("\n所有骨架的ProteinMPNN设计完成！")

# ===== 单元格4: 收集和整合结果 =====
import glob
import json

all_sequences = {}

# 收集每个骨架的设计结果
for i in range(len(pdb_files)):
    backbone_id = f"backbone_{i:04d}"
    result_dir = f"outputs/{backbone_id}"
    
    if os.path.exists(f"{result_dir}/best.pdb"):
        sequences = []
        
        # 查找序列文件
        seq_files = glob.glob(f"{result_dir}/seqs/*.fa")
        
        for seq_file in seq_files:
            try:
                with open(seq_file, 'r') as f:
                    lines = f.read().strip().split('\n')
                    if len(lines) >= 2:
                        seq_id = lines[0].replace('>', '')
                        sequence = lines[1]
                        
                        sequences.append({
                            "sequence_id": seq_id,
                            "sequence": sequence,
                            "length": len(sequence),
                            "method": "proteinmpnn_colabdesign"
                        })
            except:
                continue
        
        all_sequences[backbone_id] = {
            "backbone_file": f"{backbone_id}.pdb",
            "backbone_id": backbone_id,
            "design_method": "proteinmpnn",
            "parameters": {
                "num_sequences": num_seqs,
                "sampling_temp": mpnn_sampling_temp,
                "remove_aa": rm_aa,
                "num_recycles": num_recycles
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "sequences": sequences
        }
        
        print(f"{backbone_id}: {len(sequences)} 个序列")

# 保存最终结果
with open("proteinmpnn_sequences.json", "w") as f:
    json.dump(all_sequences, f, indent=2)

# 统计
total_sequences = sum(len(data["sequences"]) for data in all_sequences.values())
print(f"\n最终统计:")
print(f"成功处理: {len(all_sequences)} 个骨架")
print(f"总序列数: {total_sequences} 个")

# ===== 单元格5: 下载结果 =====
!zip -r proteinmpnn_design_results.zip outputs/ proteinmpnn_sequences.json

from google.colab import files
files.download('proteinmpnn_design_results.zip')

print("下载完成！解压到: F:/Project/蛋白质设计/designs/iter1/")

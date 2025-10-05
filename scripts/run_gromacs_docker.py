"""
脚本名: run_gromacs_docker.py
作者: Kuroneko
日期: 2025.10.3

功能: 使用Docker运行GROMACS进行分子动力学模拟

依赖: Docker Desktop

运行示例:
python scripts/run_gromacs_docker.py

MD模拟流程说明:
===============

1. 预处理阶段:
   - pdb2gmx: 将PDB文件转换为GROMACS格式，生成拓扑文件
   - editconf: 创建模拟盒子，为蛋白质添加周期性边界条件
   - solvate: 在盒子中填充水分子
   - genion: 添加离子中和电荷

2. 平衡阶段:
   - 能量最小化: 消除原子间冲突
   - NVT平衡: 恒定体积下加热到目标温度
   - NPT平衡: 恒定压力下调整密度

3. 生产阶段:
   - 生产MD: 正式的数据收集模拟
   - 分析: 计算RMSD等稳定性指标

超时说明:
=========
- MD模拟本身就很耗时，超时是正常现象
- 脚本已优化超时设置：MD运行30分钟，预处理1分钟
- 测试模式使用减少的步数，可以更快完成
- 完整模式需要更长时间，建议先运行测试模式验证

文件说明:
=========
- 输入: scores/top_10_sequences_iter1.json (前10名序列)
- 输出: scores/gromacs_md_results_iter1.csv (MD结果)
- 中间文件: scores/*.gro, *.top, *.xtc (轨迹文件)
"""

import os
import sys
import subprocess
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict

def check_docker():
    """检查Docker是否可用"""
    try:
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ Docker 可用: {result.stdout.strip()}")
            return True
        else:
            print("✗ Docker 不可用")
            return False
    except Exception as e:
        print(f"✗ Docker 检查失败: {e}")
        return False

def start_gromacs_container():
    """检查GROMACS容器状态"""
    print("\n检查GROMACS容器状态...")
    try:
        # 检查容器是否在运行
        result = subprocess.run(['docker', 'ps', '--filter', 'name=gromacs_md', '--format', '{{.Names}}'], 
                              capture_output=True, text=True)
        if 'gromacs_md' in result.stdout:
            print("✓ GROMACS容器正在运行")
            return True
        else:
            print("✗ GROMACS容器未运行，请先启动容器")
            return False
    except Exception as e:
        print(f"✗ 检查容器状态出错: {e}")
        return False

def run_gromacs_md(pdb_file: str, output_name: str, md_time_ns: float = 1.0, test_mode: bool = False) -> Dict:
    """
    在Docker容器中运行GROMACS MD模拟
    
    参数:
    - pdb_file: PDB文件路径
    - output_name: 输出文件名前缀
    - md_time_ns: MD模拟时间（纳秒）
    - test_mode: 测试模式，使用更少的步数快速测试
    """
    print(f"\n运行GROMACS MD模拟: {Path(pdb_file).name}")
    if test_mode:
        print("  [测试模式] 使用减少的步数进行快速测试")
    
    # 准备GROMACS命令序列
    # 这是一个完整的MD模拟流程，包含8个主要步骤
    commands = [
        # 步骤1: 生成拓扑文件 - 将PDB文件转换为GROMACS格式
        # pdb2gmx: 读取PDB文件，生成.gro坐标文件和.top拓扑文件
        # -ff amber99sb-ildn: 使用AMBER力场
        # -water tip3p: 使用TIP3P水模型
        f"gmx pdb2gmx -f scores/{Path(pdb_file).name} -o {output_name}.gro -p {output_name}.top -ff amber99sb-ildn -water tip3p",
        
        # 步骤2: 定义模拟盒子 - 为蛋白质添加周期性边界条件
        # editconf: 创建模拟盒子，蛋白质居中，周围留出空间
        # -c: 将蛋白质放在盒子中心
        # -d 1.0: 蛋白质边缘到盒子边缘最小距离1.0nm
        # -bt dodecahedron: 使用十二面体盒子（最节省空间）
        f"gmx editconf -f {output_name}.gro -o {output_name}_box.gro -c -d 1.0 -bt dodecahedron",
        
        # 步骤3: 添加溶剂 - 在盒子中填充水分子
        # solvate: 在盒子中填充水分子
        # -cp: 输入坐标文件（带盒子的蛋白质）
        # -cs spc216.gro: 使用SPC水模型
        f"gmx solvate -cp {output_name}_box.gro -cs spc216.gro -o {output_name}_solv.gro -p {output_name}.top",
        
        # 步骤4: 添加离子 - 中和系统电荷并添加生理盐浓度
        # grompp: 预处理，准备离子添加的输入文件
        # -maxwarn 1: 允许1个警告（忽略电荷警告）
        f"gmx grompp -f scripts/ions.mdp -c {output_name}_solv.gro -p {output_name}.top -o {output_name}_ions.tpr -maxwarn 1",
        # genion: 添加离子
        # echo 'SOL': 选择溶剂分子进行替换
        # -pname NA -nname CL: 使用Na+和Cl-离子
        # -neutral: 中和系统电荷
        f"echo 'SOL' | gmx genion -s {output_name}_ions.tpr -o {output_name}_ions.gro -p {output_name}.top -pname NA -nname CL -neutral",
        
        # 步骤5: 能量最小化 - 消除原子间冲突
        # 使用最陡下降法优化原子位置，消除不合理的原子重叠
        f"gmx grompp -f scripts/minim.mdp -c {output_name}_ions.gro -p {output_name}.top -o {output_name}_min.tpr -maxwarn 1",
        f"gmx mdrun -v -deffnm {output_name}_min",
        
        # 步骤6: NVT平衡 - 恒定体积和温度平衡
        # 在恒定体积下将系统加热到目标温度（300K）
        f"gmx grompp -f scripts/nvt.mdp -c {output_name}_min.gro -r {output_name}_min.gro -p {output_name}.top -o {output_name}_nvt.tpr -maxwarn 1",
        f"gmx mdrun -v -deffnm {output_name}_nvt",
        
        # 步骤7: NPT平衡 - 恒定压力和温度平衡
        # 在恒定压力下调整盒子大小，达到目标密度
        f"gmx grompp -f scripts/npt.mdp -c {output_name}_nvt.gro -r {output_name}_nvt.gro -t {output_name}_nvt.cpt -p {output_name}.top -o {output_name}_npt.tpr -maxwarn 1",
        f"gmx mdrun -v -deffnm {output_name}_npt",
        
        # 步骤8: 生产MD - 正式的数据收集模拟
        # 这是真正的MD模拟，用于收集数据和分析
        f"gmx grompp -f scripts/md.mdp -c {output_name}_npt.gro -r {output_name}_npt.gro -t {output_name}_npt.cpt -p {output_name}.top -o {output_name}_md.tpr -maxwarn 1",
        f"gmx mdrun -v -deffnm {output_name}_md -nt 4"
    ]
    
    results = {
        "success": False,
        "rmsd": 0.0,
        "energy": 0.0,
        "stability_score": 0.0,
        "error": None
    }
    
    try:
        for i, cmd in enumerate(commands, 1):
            print(f"  步骤 {i}/{len(commands)}: {cmd.split()[1]}")
            
            # 在Docker容器中执行命令
            docker_cmd = ['docker', 'exec', 'gromacs_md', 'bash', '-c', f"cd /workspace && {cmd}"]
            # 根据步骤类型设置不同的超时时间
            if 'mdrun' in cmd:
                timeout = 1800  # MD运行30分钟超时
            elif 'grompp' in cmd:
                timeout = 60   # 预处理1分钟超时
            else:
                timeout = 300  # 其他步骤5分钟超时
            
            result = subprocess.run(docker_cmd, capture_output=True, text=True, timeout=timeout)
            
            if result.returncode != 0:
                print(f"    ✗ 失败: {result.stderr}")
                results["error"] = f"步骤{i}失败: {result.stderr}"
                return results
        
        # 分析结果
        print("  分析MD结果...")
        analysis_results = analyze_md_results(output_name)
        results.update(analysis_results)
        results["success"] = True
        
        print(f"  ✓ MD模拟完成: RMSD={results['rmsd']:.2f} Å")
        
    except subprocess.TimeoutExpired:
        results["error"] = "MD模拟超时"
        print("  ✗ MD模拟超时")
    except Exception as e:
        results["error"] = str(e)
        print(f"  ✗ MD模拟出错: {e}")
    
    return results

def analyze_md_results(output_name: str) -> Dict:
    """分析MD模拟结果"""
    results = {
        "rmsd": 0.0,
        "energy": 0.0,
        "stability_score": 0.0
    }
    
    try:
        # 计算RMSD
        rmsd_cmd = f"echo 'Protein' | gmx rms -s {output_name}_md.tpr -f {output_name}_md.xtc -o {output_name}_rmsd.xvg"
        docker_cmd = ['docker', 'exec', 'gromacs_md', 'bash', '-c', f"cd /workspace && {rmsd_cmd}"]
        subprocess.run(docker_cmd, capture_output=True, text=True)
        
        # 读取RMSD数据
        rmsd_file = f"scores/{output_name}_rmsd.xvg"
        if Path(rmsd_file).exists():
            with open(rmsd_file, 'r') as f:
                lines = f.readlines()
                rmsd_values = []
                for line in lines:
                    if not line.startswith('#') and not line.startswith('@'):
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            rmsd_values.append(float(parts[1]))
                
                if rmsd_values:
                    results["rmsd"] = sum(rmsd_values) / len(rmsd_values)
        
        # 计算稳定性分数 (RMSD越小越稳定)
        if results["rmsd"] > 0:
            results["stability_score"] = max(0, 100 - results["rmsd"] * 10)
        
    except Exception as e:
        print(f"  分析结果时出错: {e}")
    
    return results

def create_mdp_files():
    """创建GROMACS参数文件"""
    print("\n创建GROMACS参数文件...")
    
    # 创建scripts目录
    scripts_dir = Path("scripts")
    scripts_dir.mkdir(exist_ok=True)
    
    # 离子参数文件
    ions_mdp = """; Ions
integrator  = steep
emtol       = 1000.0
emstep      = 0.01
nsteps      = 50000
nstlist     = 10
cutoff-scheme = Verlet
coulombtype = PME
rcoulomb    = 1.0
rvdw        = 1.0
pbc         = xyz
"""
    
    # 能量最小化参数文件
    minim_mdp = """; Minimization
integrator  = steep
emtol       = 1000.0
emstep      = 0.01
nsteps      = 5000
nstlist     = 10
cutoff-scheme = Verlet
coulombtype = PME
rcoulomb    = 1.0
rvdw        = 1.0
pbc         = xyz
"""
    
    # NVT平衡参数文件
    nvt_mdp = """; NVT
integrator  = md
dt          = 0.002
nsteps      = 10000
nstlist     = 10
cutoff-scheme = Verlet
coulombtype = PME
rcoulomb    = 1.0
rvdw        = 1.0
pbc         = xyz
tcoupl      = V-rescale
tc-grps     = Protein Non-Protein
tau-t       = 0.1 0.1
ref-t       = 300 300
"""
    
    # NPT平衡参数文件
    npt_mdp = """; NPT
integrator  = md
dt          = 0.002
nsteps      = 10000
nstlist     = 10
cutoff-scheme = Verlet
coulombtype = PME
rcoulomb    = 1.0
rvdw        = 1.0
pbc         = xyz
tcoupl      = V-rescale
tc-grps     = Protein Non-Protein
tau-t       = 0.1 0.1
ref-t       = 300 300
pcoupl      = Parrinello-Rahman
pcoupltype  = isotropic
tau-p       = 2.0
ref-p       = 1.0
compressibility = 4.5e-5
"""
    
    # MD生产参数文件
    md_mdp = """; MD
integrator  = md
dt          = 0.002
nsteps      = 50000
nstlist     = 10
cutoff-scheme = Verlet
coulombtype = PME
rcoulomb    = 1.0
rvdw        = 1.0
pbc         = xyz
tcoupl      = V-rescale
tc-grps     = Protein Non-Protein
tau-t       = 0.1 0.1
ref-t       = 300 300
pcoupl      = Parrinello-Rahman
pcoupltype  = isotropic
tau-p       = 2.0
ref-p       = 1.0
compressibility = 4.5e-5
nstxout     = 5000
nstvout     = 5000
nstenergy   = 5000
nstlog      = 5000
nstxout-compressed = 5000
"""
    
    # 写入文件
    files = {
        "ions.mdp": ions_mdp,
        "minim.mdp": minim_mdp,
        "nvt.mdp": nvt_mdp,
        "npt.mdp": npt_mdp,
        "md.mdp": md_mdp
    }
    
    for filename, content in files.items():
        filepath = scripts_dir / filename
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"  ✓ 创建 {filename}")

def run_md_on_top_sequences(test_mode: bool = False):
    """
    对前10名序列运行MD模拟
    
    参数:
    - test_mode: 测试模式，只运行第一个序列进行快速测试
    """
    print("\n" + "="*60)
    if test_mode:
        print("GROMACS MD模拟 - 测试模式（只运行第一个序列）")
    else:
        print("对前10名序列运行GROMACS MD模拟")
    print("="*60)
    
    # 读取前10名序列
    top_sequences_file = Path("scores/top_10_sequences_iter1.json")
    if not top_sequences_file.exists():
        print("错误: 未找到前10名序列文件")
        return
    
    with open(top_sequences_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    sequences = data.get('top_sequences', [])
    if not sequences:
        print("错误: 序列数据为空")
        return
    
    # 测试模式只处理第一个序列
    if test_mode:
        sequences = sequences[:1]
        print(f"测试模式: 只处理第一个序列")
    else:
        print(f"找到 {len(sequences)} 个序列，开始MD模拟...")
    
    results = []
    for i, seq_info in enumerate(sequences, 1):
        sequence_id = seq_info['sequence_id']
        pdb_file = f"scores/{seq_info['pdb_file']}"
        
        if not Path(pdb_file).exists():
            print(f"  ✗ 跳过 {sequence_id}: PDB文件不存在")
            continue
        
        print(f"\n处理序列 {i}/{len(sequences)}: {sequence_id}")
        
        # 运行MD模拟
        # test_mode参数传递给MD模拟函数
        md_result = run_gromacs_md(pdb_file, sequence_id, md_time_ns=1.0, test_mode=test_mode)
        
        # 保存结果
        result = {
            "sequence_id": sequence_id,
            "pdb_file": seq_info['pdb_file'],
            "rosetta_score": seq_info.get('total_score', 0),
            "md_success": md_result["success"],
            "md_rmsd": md_result["rmsd"],
            "md_stability_score": md_result["stability_score"],
            "md_error": md_result.get("error", "")
        }
        results.append(result)
    
    # 保存MD结果
    results_df = pd.DataFrame(results)
    results_file = "scores/gromacs_md_results_iter1.csv"
    results_df.to_csv(results_file, index=False)
    
    print(f"\n✓ MD模拟完成，结果保存到: {results_file}")
    
    # 打印统计
    successful = results_df[results_df['md_success'] == True]
    if len(successful) > 0:
        print(f"成功模拟: {len(successful)}/{len(results)}")
        print(f"平均RMSD: {successful['md_rmsd'].mean():.2f} Å")
        print(f"平均稳定性: {successful['md_stability_score'].mean():.2f}")

def main():
    """主函数"""
    print("=" * 80)
    print("GROMACS Docker MD 模拟")
    print("作者: Kuroneko | 日期: 2025.10.3")
    print("=" * 80)
    
    # 询问用户是否使用测试模式
    print("\n选择运行模式:")
    print("1. 测试模式 - 只运行第一个序列，快速验证（推荐首次使用）")
    print("2. 完整模式 - 运行所有10个序列（需要较长时间）")
    
    while True:
        choice = input("请选择 (1/2): ").strip()
        if choice == "1":
            test_mode = True
            print("✓ 选择测试模式")
            break
        elif choice == "2":
            test_mode = False
            print("✓ 选择完整模式")
            break
        else:
            print("无效选择，请输入 1 或 2")
    
    # 检查Docker
    if not check_docker():
        print("请先安装并启动Docker Desktop")
        return
    
    # 创建参数文件
    create_mdp_files()
    
    # 启动容器
    if not start_gromacs_container():
        return
    
    # 运行MD模拟
    run_md_on_top_sequences(test_mode=test_mode)
    
    print("\n" + "="*60)
    if test_mode:
        print("测试模式完成! 如果成功，可以运行完整模式")
    else:
        print("MD模拟完成!")
    print("="*60)

if __name__ == "__main__":
    main()

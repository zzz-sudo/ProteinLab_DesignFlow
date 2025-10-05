# RFdiffusion 蛋白质骨架生成 - 完整版
# 项目: protein_backbone
# 生成时间: 2025-10-03 03:44:25
# 作者: Kuroneko
# 参数: contigs=100, designs=20, iterations=50

print("="*60)
print("RFdiffusion 蛋白质骨架生成")
print(f"项目: protein_backbone")
print(f"contigs: 100")
print(f"PDB模板: 无条件生成")
print(f"骨架数量: 20")
print(f"迭代次数: 50")
print(f"对称性: none")
print("="*60)

import os, time, signal, sys, random, string, re
import json, numpy as np, matplotlib.pyplot as plt
from IPython.display import display, HTML
import ipywidgets as widgets

# 安装py3Dmol用于3D可视化
print("\n安装3D可视化依赖...")
os.system("pip install py3Dmol")
import py3Dmol

from google.colab import files

%cd /content

# ================================
# RFdiffusion 环境安装
# ================================
print("\n=== 开始环境安装 ===")
print(f"开始时间: {time.ctime()}")
print("安装RFdiffusion和相关依赖...")

# 下载参数文件
if not os.path.isdir("params"):
    print("\n1. 下载参数文件...")
    os.system("apt-get install aria2")
    os.system("mkdir params")


    os.system("(aria2c -q -x 16 https://files.ipd.uw.edu/krypton/schedules.zip; aria2c -q -x 16 http://files.ipd.uw.edu/pub/RFdiffusion/6f5902ac237024bdd0c176cb93063dc4/Base_ckpt.pt; aria2c -q -x 16 http://files.ipd.uw.edu/pub/RFdiffusion/e29311f6f1bf1af907f9ef9f44b8328b/Complex_base_ckpt.pt; aria2c -q -x 16 https://storage.googleapis.com/alphafold/alphafold_params_2022-86.tar; tar -xf alphafold_params_2022-86.tar -C params; touch params/done.txt) &")

# 安装RFdiffusion
if not os.path.isdir("RFdiffusion"):
    print("\n2. 安装RFdiffusion...")
    os.system("git clone https://github.com/sokrypton/RFdiffusion.git")
    os.system("pip install jedi omegaconf hydra-core icecream pyrsistent pynvml decorator")
    os.system("pip install git+https://github.com/NVIDIA/dllogger#egg=dllogger")
    os.system("pip install --no-dependencies dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html")
    os.system("pip install --no-dependencies e3nn==0.5.5 opt_einsum_fx")
    os.system("cd RFdiffusion/env/SE3Transformer; pip install .")
    os.system("wget -qnc https://files.ipd.uw.edu/krypton/ananas")
    os.system("chmod +x ananas")
    print("RFdiffusion安装完成")

# 安装ColabDesign
if not os.path.isdir("colabdesign"):
    print("\n3. 安装ColabDesign...")
    os.system("pip -q install git+https://github.com/sokrypton/ColabDesign.git@v1.1.1")
    os.system("ln -s /usr/local/lib/python3.*/dist-packages/colabdesign colabdesign")

# 下载模型文件
if not os.path.isdir("RFdiffusion/models"):
    print("\n4. 下载模型文件...")
    os.system("mkdir RFdiffusion/models")
    models = ["Base_ckpt.pt","Complex_base_ckpt.pt"]
    for m in models:
        while os.path.isfile(f"{m}.aria2"):
            time.sleep(5)
    os.system(f"mv {' '.join(models)} RFdiffusion/models")
    os.system("unzip schedules.zip; rm schedules.zip")

# 设置环境
if 'RFdiffusion' not in sys.path:
    os.environ["DGLBACKEND"] = "pytorch"
    sys.path.append('RFdiffusion')

from inference.utils import parse_pdb
from colabdesign.rf.utils import get_ca, fix_contigs, fix_partial_contigs, fix_pdb, sym_it
from colabdesign.shared.protein import pdb_to_string
from colabdesign.shared.plot import plot_pseudo_3D

print("\n=== 环境安装完成 ===")

# ================================
# RFdiffusion 核心函数
# ================================

def get_pdb(pdb_code=None):
    """获取PDB文件"""
    print(f"\n=== PDB文件处理 ===")
    print(f"输入参数: {pdb_code if pdb_code else '无(将提示上传文件)'}")
    
    if pdb_code is None or pdb_code == "":
        print("\n 模式: 本地文件上传")
        print("请选择并上传PDB文件...")
        upload_dict = files.upload()
        pdb_string = upload_dict[list(upload_dict.keys())[0]]
        with open("tmp.pdb","wb") as out:
            out.write(pdb_string)
        print(" PDB文件上传完成: tmp.pdb")
        return "tmp.pdb"
    elif os.path.isfile(pdb_code):
        print(f" 模式: 使用已存在文件")
        print(f"文件名: {pdb_code}")
        return pdb_code
    elif len(pdb_code) == 4:
        fn = f"{pdb_code}.pdb1"
        print(f"\n 模式: RCSB PDB数据库下载")
        print(f"PDB代码: {pdb_code}")
        if not os.path.isfile(fn):
            print(f"正在从RCSB下载 {pdb_code}...")
            os.system(f"wget -qnc https://files.rcsb.org/download/{pdb_code}.pdb1.gz")
            os.system(f"gunzip {pdb_code}.pdb1.gz")
        print(f" RCSB文件下载完成: {fn}")
        return fn
    else:
        print(f"\n🧬 模式: AlphaFold数据库下载")
        print(f"UniProt代码: {pdb_code}")
        print(f"正在从AlphaFold下载 {pdb_code}...")
        os.system(f"wget -qnc https://alphafold.ebi.ac.uk/files/AF-{pdb_code}-F1-model_v3.pdb")
        afn = f"AF-{pdb_code}-F1-model_v3.pdb"
        print(f" AlphaFold文件下载完成: {afn}")
        return afn

def run_ananas(pdb_str, path, sym=None):
    """AnAnaS对称检测"""
    print("\n--- AnAnaS 对称检测 ---")
    pdb_filename = f"outputs/{path}/ananas_input.pdb"
    out_filename = f"outputs/{path}/ananas.json"
    with open(pdb_filename,"w") as handle:
        handle.write(pdb_str)

    cmd = f"./ananas {pdb_filename} -u -j {out_filename}"
    if sym is None: 
        os.system(cmd)
    else: 
        os.system(f"{cmd} {sym}")

    try:
        out = json.load(open(out_filename,"r").read())
        results,AU = out[0], out[-1]["AU"]
        group = AU["group"]
        chains = AU["chain names"]
        rmsd = results["Average_RMSD"]
        print(f"AnAnaS检测结果: {group} 对称，RMSD {rmsd:.3}, 链 {chains}")
        return results, pdb_str
    except:
        print("AnAnaS检测失败，继续使用原始结构")
        return None, pdb_str

def run(command, steps, num_designs=1, visual="none"):
    """RFdiffusion主执行函数"""
    
    def run_command_and_get_pid(command):
        pid_file = '/dev/shm/pid'
        os.system(f'nohup {command} > /dev/null & echo $! > {pid_file}')
        with open(pid_file, 'r') as f:
            pid = int(f.read().strip())
        os.remove(pid_file)
        return pid
    
    def is_process_running(pid):
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        else:
            return True

    print(f"\n=== 开始RFdiffusion生成 ===")
    print(f"生成目标: {num_designs} 个骨架，{steps} 步")
    print(f"执行命令: {command}")
    
    # 进度显示
    if visual != "none":
        run_output = widgets.Output()
        progress = widgets.FloatProgress(min=0, max=1, description='RFdiffusion运行中', bar_style='info')
        display(widgets.VBox([progress, run_output]))
    else:
        progress = None

    # 清理之前的运行
    for n in range(steps):
        sf = f"/dev/shm/{n}.pdb"
        if os.path.isfile(sf):
            os.remove(sf)

    pid = run_command_and_get_pid(command)
    print(f"RFdiffusion进程已启动，PID: {pid}")
    
    try:
        for design_num in range(num_designs):
            print(f"\n--- 生成骨架 {design_num + 1}/{num_designs} ---")
            
            step_start_time = time.time()
            for n in range(steps):
                step_file = f"/dev/shm/{n}.pdb"
                
                wait = True
                while wait:
                    time.sleep(0.1)
                    if os.path.isfile(step_file):
                        with open(step_file) as f:
                            pdb_str = f.read()
                        if pdb_str[-3:] == "TER":
                            wait = False
                            elapsed = time.time() - step_start_time
                            print(f"  步骤 {n+1}/{steps} 完成 ({elapsed:.1f}秒)")
                        elif not is_process_running(pid):
                            print(f"  进程意外终止!")
                            if progress:
                                progress.bar_style = 'danger'
                                progress.description = '失败'
                            return False
                    elif not is_process_running(pid):
                        print(f"  进程意外终止!")
                        if progress:
                            progress.bar_style = 'danger'
                            progress.description = '失败'
                        return False

                if progress:
                    progress.value = (n+1) / steps
                
                # 可视化输出
                if visual == "image" and progress and n % 5 == 0:  # 每5步显示一次
                    with run_output:
                        run_output.clear_output(wait=True)
                        try:
                            xyz, bfactor = get_ca(step_file, get_bfact=True)
                            fig = plt.figure(dpi=100, figsize=(6,6))
                            ax = fig.add_subplot(111)
                            ax.set_xticks([])
                            ax.set_yticks([])
                            plot_pseudo_3D(xyz, c=bfactor, cmin=0.5, cmax=0.9, ax=ax)
                            plt.show()
                        except:
                            pass
        
        # 等待进程完全结束
        while is_process_running(pid):
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n用户中断生成")
        os.kill(pid, signal.SIGTERM)
        if progress:
            progress.bar_style = 'danger'
            progress.description = '用户中断'
        return False
    
    print(f"\n=== RFdiffusion生成完成 ===")
    if progress:
        progress.bar_style = 'success'
        progress.description = '完成'
    return True

def run_diffusion(contigs, path, pdb=None, iterations=50,
                  symmetry="none", order=1, hotspot=None,
                  chains=None, add_potential=False,
                  num_designs=1, visual="none"):
    """RFdiffusion核心运行函数"""
    
    print(f"\n=== RFdiffusion参数设置 ===")
    print(f"contigs: {contigs}")
    print(f"iterations: {iterations}")
    print(f"symmetry: {symmetry}")
    print(f"hotspot: {hotspot}")
    print(f"pdb: {pdb}")
    
    full_path = f"outputs/{path}"
    os.makedirs(full_path, exist_ok=True)
    print(f"输出目录: {full_path}")
    
    opts = [
        f"inference.output_prefix={full_path}",
        f"inference.num_designs={num_designs}"
    ]
    
    if chains == "":
        chains = None
    
    # 对称性设置
    if symmetry in ["auto","cyclic","dihedral"]:
        if symmetry == "auto":
            sym, copies = None, 1
            print("模式: 自动检测对称性")
        else:
            sym, copies = {"cyclic":(f"c{order}",order),
                         "dihedral":(f"d{order}",order*2)}[symmetry]
            print(f"模式: {symmetry} 对称，复制数 {copies}")
    else:
        symmetry = None
        sym, copies = None, 1
        print("模式: 无对称")
    
    # contigs解析
    contigs_list = contigs.replace(","," ").replace(":"," ").split()
    is_fixed, is_free = False, False
    fixed_chains = []
    
    for contig in contigs_list:
        for x in contig.split("/"):
            a = x.split("-")[0]
            if a[0].isalpha():
                is_fixed = True
                if a[0] not in fixed_chains:
                    fixed_chains.append(a[0])
            if a.isnumeric():
                is_free = True
    
    if len(contigs_list) == 0 or not is_free:
        mode = "partial"
        print("检测模式: partial (部分扩散)")
    elif is_fixed:
        mode = "fixed"
        print("检测模式: fixed (固定结构)")
    else:
        mode = "free"
        print("检测模式: free (自由生成)")
    
    # PDB处理
    if mode in ["partial","fixed"]:
        print(f"\n--- 处理模板PDB ---")
        pdb_str = pdb_to_string(get_pdb(pdb), chains=chains)
        
        if symmetry == "auto":
            print("--- 自动检测对称性 ---")
            results, pdb_str = run_ananas(pdb_str, path)
            if results:
                group = results["group"]
                if group[0] == "c":
                    symmetry = "cyclic"
                    sym, copies = group, int(group[1:])
                    print(f"检测到循环对称: {sym}")
                elif group[0] == "d":
                    symmetry = "dihedral"
                    sym, copies = group, 2 * int(group[1:])
                    print(f"检测到二面对称: {sym}")
                else:
                    print(f"未支持的对称类型: {group}")
                    symmetry = None
                    sym, copies = None, 1
            else:
                print("对称检测失败，使用无对称模式")
        
        pdb_filename = f"{full_path}/input.pdb"
        with open(pdb_filename, "w") as handle:
            handle.write(pdb_str)
        print(f"输入PDB已保存: {pdb_filename}")
        
        parsed_pdb = parse_pdb(pdb_filename)
        opts.append(f"inference.input_pdb={pdb_filename}")
        
        if mode == "partial":
            iterations = int(80 * (iterations / 200))
            opts.append(f"diffuser.partial_T={iterations}")
            contigs_list = fix_partial_contigs(contigs_list, parsed_pdb)
            print(f"partial模式，调整iterations: {iterations}")
        else:
            opts.append(f"diffuser.T={iterations}")
            contigs_list = fix_contigs(contigs_list, parsed_pdb)
    else:
        opts.append(f"diffuser.T={iterations}")
        parsed_pdb = None
        contigs_list = fix_contigs(contigs_list, parsed_pdb)
    
    # 热点设置
    if hotspot and hotspot.strip():
        opts.append(f"ppi.hotspot_res=[{hotspot}]")
        print(f"设置热点残基: {hotspot}")
    
    # 对称性应用
    if sym:
        sym_opts = ["--config-name symmetry", f"inference.symmetry={sym}"]
        if add_potential:
            sym_opts += [
                "'potentials.guiding_potentials=[\"type:olig_contacts,weight_intra:1,weight_inter:0.1\"]'",
                "potentials.olig_intra_all=True",
                "potentials.olig_inter_all=True",
                "potentials.guide_scale=2",
                "potentials.guide_decay=quadratic"
            ]
            print("启用势能约束")
        opts = sym_opts + opts
        contigs_list = sum([contigs_list] * copies,[])
    
    opts.append(f"'contigmap.contigs=[{' '.join(contigs_list)}]'")
    opts += ["inference.dump_pdb=True","inference.dump_pdb_path=/dev/shm"]

    print(f"\n最终contigs: {contigs_list}")
    print(f"总参数: {len(opts)} 个")

    opts_str = " ".join(opts)
    cmd = f"./RFdiffusion/run_inference.py {opts_str}"
    
    print(f"\n执行命令:")
    print(f"{cmd}")

    # 运行RFdiffusion
    success = run(cmd, iterations, num_designs, visual=visual)

    if success:
        print(f"\n--- 修复PDB文件 ---")
        fixed_count = 0
        for n in range(num_designs):
            files_to_fix = [
                f"outputs/traj/{path}_{n}_pX0_traj.pdb",
                f"outputs/traj/{path}_{n}_Xt-1_traj.pdb",
                f"{full_path}_{n}.pdb"
            ]
            for pdb_file in files_to_fix:
                if os.path.exists(pdb_file):
                    with open(pdb_file, "r") as f:
                        pdb_str = f.read()
                    with open(pdb_file, "w") as f:
                        f.write(fix_pdb(pdb_str, contigs_list))
                    fixed_count += 1
        print(f"已修复 {fixed_count} 个PDB文件")
    else:
        print("PDB修复跳过 (生成失败)")

    return contigs_list, copies

# ================================
# 参数设置和运行
# ================================

# 参数配置
name = "protein_backbone"
contigs = "100"
pdb = ""
iterations = 50
symmetry = "none"
order = 1
hotspot = ""
chains = ""
add_potential = True
num_designs = 20
visual = "image"

print(f"\n=== 参数确认 ===")
print(f"项目名称: {name}")
print(f"contigs: {contigs}")
print(f"PDB模板: {pdb if pdb else '无条件生成'}")
print(f"生成数量: {num_designs}")
print(f"迭代次数: {iterations}")
print(f"对称性: {symmetry}")
if hotspot:
    print(f"热点残基: {hotspot}")
if chains:
    print(f"链筛选: {chains}")
print(f"势能约束: {add_potential}")
print(f"可视化: {visual}")

# 输出路径
path = name
counter = 0
while os.path.exists(f"outputs/{path}_0.pdb"):
    counter += 1
    path = f"{name}_{counter:03d}"
    print(f"\n路径已存在，使用: {path}")

print("\n=== 开始骨架生成 ===")
start_time = time.time()
print(f"开始时间: {time.ctime(start_time)}")

# 运行RFdiffusion
flags = {
    "contigs": contigs,
    "pdb": pdb,
    "order": order,
    "iterations": iterations,
    "symmetry": symmetry,
    "hotspot": hotspot,
    "path": path,
    "chains": chains,
    "add_potential": add_potential,
    "num_designs": num_designs,
    "visual": visual
}

contigs, copies = run_diffusion(**flags)

# ================================
# 结果检查
# ================================
generation_time = time.time() - start_time
print(f"\n=== 生成完成 ===")
print(f"总耗时: {generation_time:.1f} 秒")

import glob
pdb_files = glob.glob(f"outputs/{path}_*.pdb")

if pdb_files:
    print(f"\n✓ 成功生成 {len(pdb_files)} 个骨架文件:")
    total_size = 0
    
    for i, file_path in enumerate(pdb_files):
        size = os.path.getsize(file_path)
        total_size += size
        filename = os.path.basename(file_path)
        status = "✓" if size > 1000 else "⚠"
        print(f"  {status} {i+1:2d}. {filename} ({size:,} bytes)")
    
    print(f"\n总大小: {total_size:,} bytes")
    
    # 3D可视化
    if visual != "none":
        print(f"\n=== 3D结构显示 ===")
        
        def show_structure(file_num=0):
            pdb_file = f"outputs/{path}_{file_num}.pdb"
            if not os.path.exists(pdb_file):
                print(f"文件不存在: {pdb_file}")
                return
            
            view = py3Dmol.view(js='https://3dmol.org/build/3Dmol.js')
            with open(pdb_file, 'r') as f:
                pdb_str = f.read()
            view.addModel(pdb_str, 'pdb')
            view.setStyle({'cartoon': {'colorscheme': {'prop':'b','gradient': 'roygb','min':0.5,'max':0.9}}})
            view.zoomTo()
            view.show()
        
        show_structure(0)
        
        if len(pdb_files) > 1:
            print(f"\n提示: 查看其他结构")
            print(f"show_structure(1)  # 第2个结构")
            print(f"show_structure(2)  # 第3个结构")
            print(f"# 最多到 show_structure({len(pdb_files)-1})")

else:
    print(f"\n✗ 生成失败")
    print("检查错误信息并重试")

# ================================
# 文件打包下载
# ================================
print(f"\n=== 打包下载 ===")

if pdb_files:
    import zipfile
    zip_name = f"{path}.result.zip"
    
    print(f"打包 {len(pdb_files)} 个文件...")
    
    all_files = []
    all_files.extend(pdb_files)
    all_files.extend(glob.glob(f"outputs/traj/{path}*"))
    
    with zipfile.ZipFile(zip_name, 'w') as z:
        for file_path in all_files:
            z.write(file_path, os.path.basename(file_path))
    
    if os.path.exists(zip_name):
        file_size = os.path.getsize(zip_name)
        print(f"✓ 打包完成: {zip_name}")
        print(f"✓ 压缩包大小: {file_size:,} bytes")
        
        files.download(zip_name)
        print(f"✓ 下载已开始")
        print(f"\n文件名说明:")
        print(f"   {path}_0.pdb  - 第1个生成的骨架")
        print(f"   {path}_1.pdb  - 第2个生成的骨架")
        print(f"   ...")
    else:
        print("✗ 打包失败")
else:
    print("✗ 没有文件可打包")

# ================================
# 总结报告
# ================================
print(f"\n" + "="*60)
print("RFdiffusion 执行报告")
print("="*60)
print(f"项目: {name}")
print(f"模式: {contigs}")
print(f"目标: {num_designs} 个骨架")
print(f"耗时: {generation_time:.1f} 秒")

if pdb_files:
    print(f"✓ 成功: {len(pdb_files)} 个骨架")
    if len(pdb_files) == num_designs:
        print("✓ 完全成功")
    elif len(pdb_files) >= num_designs * 0.8:
        print("⚠ 大部分成功")
    else:
        print("⚠ 部分成功")
        
    avg_size = np.mean([os.path.getsize(f) for f in pdb_files])
    print(f"✓ 平均文件大小: {avg_size:.0f} bytes")
else:
    print("✗ 生成失败")
    print("建议检查:")
    print("- contigs参数是否正确")
    print("- PDB模板是否存在")
    print("- GPU内存是否足够")

print(f"\n感谢使用RFdiffusion!")
print("="*60)

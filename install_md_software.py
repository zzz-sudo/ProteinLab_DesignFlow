"""
脚本名: install_md_software.py
作者: Kuroneko
日期: 2025.10.3

功能: 分子动力学软件安装指南
包括: GROMACS, OpenMM, MDAnalysis

运行示例:
python scripts/install_md_software.py
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def print_header():
    print("=" * 80)
    print("分子动力学软件安装指南")
    print("作者: Kuroneko | 日期: 2025.10.3")
    print("=" * 80)

def check_system():
    """检查系统信息"""
    print("\n系统信息:")
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"Python版本: {sys.version}")
    print(f"架构: {platform.machine()}")
    
    # 检查conda
    try:
        result = subprocess.run(['conda', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Conda: {result.stdout.strip()}")
        else:
            print("Conda: 未安装")
    except:
        print("Conda: 未安装")

def install_gromacs():
    """安装GROMACS"""
    print("\n" + "="*60)
    print("GROMACS 安装指南")
    print("="*60)
    
    print("""
GROMACS 是一个高性能的分子动力学模拟软件包。

安装方法:

1. 使用 Conda 安装 (推荐):
   conda install -c conda-forge gromacs
   
2. 使用 pip 安装 (简化版):
   pip install gromacs
   
3. 从源码编译 (高级用户):
   - 下载源码: https://manual.gromacs.org/current/download.html
   - 安装依赖: CMake, GCC, FFTW
   - 编译安装

4. 预编译二进制文件:
   - Windows: 下载预编译版本
   - Linux: 使用包管理器 (apt, yum, dnf)
   - macOS: 使用 Homebrew

验证安装:
   gmx --version
""")

def install_openmm():
    """安装OpenMM"""
    print("\n" + "="*60)
    print("OpenMM 安装指南")
    print("="*60)
    
    print("""
OpenMM 是一个高性能的分子动力学模拟库。

安装方法:

1. 使用 Conda 安装 (推荐):
   conda install -c conda-forge openmm
   
2. 使用 pip 安装:
   pip install openmm
   
3. 安装 CUDA 支持 (可选，用于GPU加速):
   conda install -c conda-forge openmm cudatoolkit
   
4. 验证安装:
   python -c "import openmm; print(openmm.__version__)"
""")

def install_mdanalysis():
    """安装MDAnalysis"""
    print("\n" + "="*60)
    print("MDAnalysis 安装指南")
    print("="*60)
    
    print("""
MDAnalysis 是一个用于分析分子动力学轨迹的Python库。

安装方法:

1. 使用 Conda 安装 (推荐):
   conda install -c conda-forge mdanalysis
   
2. 使用 pip 安装:
   pip install MDAnalysis
   
3. 安装可选依赖:
   pip install MDAnalysis[analysis]  # 包含更多分析工具
   
4. 验证安装:
   python -c "import MDAnalysis; print(MDAnalysis.__version__)"
""")

def install_biopython():
    """安装BioPython"""
    print("\n" + "="*60)
    print("BioPython 安装指南")
    print("="*60)
    
    print("""
BioPython 是生物信息学Python库，用于处理PDB文件。

安装方法:

1. 使用 Conda 安装 (推荐):
   conda install -c conda-forge biopython
   
2. 使用 pip 安装:
   pip install biopython
   
3. 验证安装:
   python -c "from Bio import PDB; print('BioPython 安装成功')"
""")

def auto_install():
    """自动安装所有软件"""
    print("\n" + "="*60)
    print("自动安装所有软件")
    print("="*60)
    
    packages = [
        ("biopython", "BioPython"),
        ("mdanalysis", "MDAnalysis"),
        ("openmm", "OpenMM"),
        ("gromacs", "GROMACS")
    ]
    
    for package, name in packages:
        print(f"\n正在安装 {name}...")
        try:
            if package == "gromacs":
                # GROMACS 需要特殊处理
                result = subprocess.run(['conda', 'install', '-c', 'conda-forge', 'gromacs', '-y'], 
                                      capture_output=True, text=True)
            else:
                result = subprocess.run(['conda', 'install', '-c', 'conda-forge', package, '-y'], 
                                      capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✓ {name} 安装成功")
            else:
                print(f"✗ {name} 安装失败，尝试使用pip...")
                result = subprocess.run(['pip', 'install', package], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"✓ {name} 通过pip安装成功")
                else:
                    print(f"✗ {name} 安装失败")
        except Exception as e:
            print(f"✗ {name} 安装出错: {e}")

def verify_installation():
    """验证安装"""
    print("\n" + "="*60)
    print("验证软件安装")
    print("="*60)
    
    software_checks = [
        ("BioPython", "from Bio import PDB; print('BioPython:', PDB.__version__)"),
        ("MDAnalysis", "import MDAnalysis; print('MDAnalysis:', MDAnalysis.__version__)"),
        ("OpenMM", "import openmm; print('OpenMM:', openmm.__version__)"),
        ("GROMACS", "import subprocess; result = subprocess.run(['gmx', '--version'], capture_output=True, text=True); print('GROMACS:', result.stdout.split('\\n')[0] if result.returncode == 0 else '未安装')")
    ]
    
    for name, check_code in software_checks:
        try:
            exec(check_code)
        except Exception as e:
            print(f"✗ {name}: 未安装或安装有问题")

def main():
    """主函数"""
    print_header()
    check_system()
    
    while True:
        print("\n" + "="*60)
        print("选择操作:")
        print("1. 查看 GROMACS 安装指南")
        print("2. 查看 OpenMM 安装指南")
        print("3. 查看 MDAnalysis 安装指南")
        print("4. 查看 BioPython 安装指南")
        print("5. 自动安装所有软件 (需要conda)")
        print("6. 验证软件安装")
        print("0. 退出")
        print("="*60)
        
        choice = input("请选择 (0-6): ").strip()
        
        if choice == "1":
            install_gromacs()
        elif choice == "2":
            install_openmm()
        elif choice == "3":
            install_mdanalysis()
        elif choice == "4":
            install_biopython()
        elif choice == "5":
            auto_install()
        elif choice == "6":
            verify_installation()
        elif choice == "0":
            print("退出安装指南")
            break
        else:
            print("无效选择，请重新输入")
        
        input("\n按回车键继续...")

if __name__ == "__main__":
    main()

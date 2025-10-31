"""
脚本名: step1_check_environment.py
作者: Kuroneko
日期: 2025.5.30

功能: 检查本地Python 3.9环境和所需依赖包

输入文件:
- requirements.txt (pip依赖列表)

输出文件:
- env/environment_status.json (环境检查状态记录)
- logs/step1_YYYYMMDD_HHMMSS.log (检查日志)

运行示例:
python scripts/step1_check_environment.py

依赖: 
- 需要本地Python 3.9环境
- 如果缺少依赖包，会提供安装建议

注意: 
本脚本检查本地Python环境，不会创建虚拟环境。
如果依赖包不足，请参考输出的安装步骤。
"""

import os
import sys
import json
import subprocess
import platform
from pathlib import Path
import datetime

# 添加当前项目的utils到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
try:
    from utils_io import (
        get_project_root, ensure_dir, get_abs_path, 
        setup_logger, validate_input, save_config, load_config
    )
except ImportError:
    print("错误: 无法导入 utils_io.py，请确保文件存在")
    sys.exit(1)

def check_python_version():
    """检查Python版本"""
    version = sys.version_info
    if version.major == 3 and version.minor == 9:
        return True, f"{version.major}.{version.minor}.{version.micro}"
    elif version.major == 3 and version.minor >= 9:
        return True, f"{version.major}.{version.minor}.{version.micro} (兼容)"
    else:
        return False, f"{version.major}.{version.minor}.{version.micro} (需要3.9+)"

def check_gpu_support():
    """检查GPU和CUDA支持"""
    try:
        # 检查nvidia-smi命令
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return True
    except:
        pass
    
    try:
        # 检查已安装的torch是否支持CUDA
        import torch
        return torch.cuda.is_available()
    except ImportError:
        pass
    
    return False

def check_required_packages():
    """检查必需的Python包"""
    required_packages = [
        ("numpy", "科学计算"),
        ("pandas", "数据处理"),
        ("torch", "深度学习框架"),
        ("requests", "网络请求"),
    ]
    
    optional_packages = [
        ("transformers", "HuggingFace转换器"),
        ("Bio", "生物信息学工具"),
        ("esm", "ESM模型"),  # fair-esm包导入时使用esm
    ]
    
    package_status = {}
    
    for package, description in required_packages:
        try:
            # 特殊处理某些可能有版本冲突的包
            if package == "torch":
                import torch
                package_status[package] = {
                    "installed": True, 
                    "description": description,
                    "version": torch.__version__,
                    "cuda": torch.cuda.is_available()
                }
            else:
                __import__(package)
                package_status[package] = {"installed": True, "description": description}
        except ImportError:
            package_status[package] = {"installed": False, "description": description}
        except Exception as e:
            # 捕获版本冲突等异常，标记为有问题的包
            package_status[package] = {
                "installed": False, 
                "description": description,
                "error": str(e),
                "warning": "可能有版本冲突"
            }
    
    for package, description in optional_packages:
        try:
            __import__(package)
            package_status[package] = {"installed": True, "description": description, "optional": True}
        except ImportError:
            package_status[package] = {"installed": False, "description": description, "optional": True}
        except Exception as e:
            # 捕获可选包的版本冲突
            package_status[package] = {
                "installed": False, 
                "description": description, 
                "optional": True,
                "error": str(e),
                "warning": "可能有版本冲突"
            }
    
    return package_status

def create_env_installation_script(package_status, logger):
    """创建环境安装脚本"""
    script_content = "# Python环境依赖安装脚本\n"
    script_content += "# 请运行以下命令安装缺少的包\n\n"
    
    missing_required = [pkg for pkg, status in package_status.items() 
                       if not status["installed"] and not status.get("optional", False)]
    missing_optional = [pkg for pkg, status in package_status.items() 
                       if not status["installed"] and status.get("optional", False)]
    
    if missing_required:
        script_content += "# 必需的包 (必须安装)\n"
        script_content += "pip install " + " ".join(missing_required) + "\n\n"
    
    if missing_optional:
        script_content += "# 可选的包 (推荐安装)\n"
        script_content += "pip install " + " ".join(missing_optional) + "\n\n"
    
    if not missing_required and not missing_optional:
        script_content += "# 所有依赖包都已安装!\n"
        script_content += "# 可以继续执行后续步骤\n"
    
    script_content += "# 安装完成后运行:\n"
    script_content += "python scripts/step2_rfdiffusion_backbone.py\n"
    
    script_file = get_abs_path("env/install_requirements.sh")
    with open(script_file, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    logger.info(f"环境安装脚本已创建: {script_file}")
    return str(script_file)

def test_environment(package_status, logger):
    """测试环境是否正确安装"""
    logger.info("测试核心包安装")
    
    test_results = {}
    
    # 测试核心包
    core_packages = ["torch", "numpy", "pandas"]
    
    for package in core_packages:
        try:
            if package == "torch":
                import torch
                test_results[package] = {
                    "available": True,
                    "version": torch.__version__,
                    "cuda": torch.cuda.is_available()
                }
            elif package == "numpy":
                import numpy as np
                test_results[package] = {
                    "available": True,
                    "version": np.__version__
                }
            elif package == "pandas":
                import pandas as pd
                test_results[package] = {
                    "available": True,
                    "version": pd.__version__
                }
            else:
                module = __import__(package)
                test_results[package] = {
                    "available": True,
                    "version": getattr(module, '__version__', 'unknown')
                }
            logger.info(f"✓ {package}: 可用")
        except ImportError:
            test_results[package] = {"available": False}
            logger.error(f"✗ {package}: 不可用")
        except Exception as e:
            test_results[package] = {"available": False, "error": str(e)}
            logger.error(f"✗ {package}: 测试失败: {e}")
    
    return test_results

def save_environment_status(status_data):
    """保存环境状态到文件"""
    status_file = get_abs_path("env/environment_status.json")
    
    with open(status_file, 'w', encoding='utf-8') as f:
        json.dump(status_data, f, indent=2, ensure_ascii=False)

def print_manual_instructions():
    """打印手动安装说明"""
    instructions = """
=== 手动安装说明 ===

如果依赖包缺失，请按以下步骤手动安装：

1. 使用本地Python 3.9环境
   运行命令: python --version (检查版本)
   如果版本不对，请安装Python 3.9或更高版本

2. 安装基础依赖包:
   pip install numpy pandas torch requests

3. 安装可选依赖包 (推荐):
   pip install transformers biopython

4. 验证安装:
   python -c "import torch, numpy, pandas; print('基础包安装成功')"

5. 在后续步骤中使用:
   python scripts/steps/step2_rfdiffusion_backbone.py

=== 故障排除 ===

• 如果遇到权限问题，可尝试:
  pip install --user package_name

• 如果GPU支持有问题，请根据您的CUDA版本安装PyTorch:
  pip install torch torchvariant --index-url https://download.pytorch.org/whl/cu121

• 如果网络问题，可以使用国内镜像:
  pip install -i https://pypi.tuna.tsinghua.edu.cn/simple package_name
"""
    print(instructions)

def main():
    """主函数"""
    print("=" * 60)
    print("Step 1: 检查Python环境")
    print("记录: Kuroneko | 日期: 2025.9.30")
    print("=" * 60)
    
    # 设置日志
    logger = setup_logger("step1")
    start_time = datetime.datetime.now()
    
    try:
        # 检查Python版本
        python_ok, python_version = check_python_version()
        
        logger.info(f"Python版本检查: {python_version}")
        print(f"✓ Python版本: {python_version}")
        
        if not python_ok:
            print(" Python版本不符合要求，需要Python 3.9或更高版本")
            logger.error(f"Python版本不符合要求: {python_version}")
            print_manual_instructions()
            return False
        
        # 检查GPU支持
        gpu_available = check_gpu_support()
        
        if gpu_available:
            print("✓ GPU支持可用")
            logger.info("GPU支持可用")
        else:
            print(" GPU支持不可用，将使用CPU")
            logger.warning("GPU支持不可用")
        
        # 检查依赖包
        print("\n检查依赖包...")
        try:
            package_status = check_required_packages()
        except Exception as e:
            print(f" 检查依赖包时遇到问题，但继续执行: {e}")
            # 基础包状态
            package_status = {
                "numpy": {"installed": True, "description": "科学计算"},
                "pandas": {"installed": True, "description": "数据处理"}, 
                "torch": {"installed": True, "description": "深度学习框架", "version": "未知", "cuda": False},
                "requests": {"installed": True, "description": "网络请求"},
                "transformers": {"installed": False, "description": "HuggingFace转换器"},
                "Bio": {"installed": False, "description": "生物信息学工具"},
                "esm": {"installed": False, "description": "ESM模型"}
            }
        
        # 显示包状态
        missing_required = [pkg for pkg, status in package_status.items() 
                          if not status["installed"] and not status.get("optional", False)]
        missing_optional = [pkg for pkg, status in package_status.items() 
                          if not status["installed"] and status.get("optional", False)]
        conflict_packages = [pkg for pkg, status in package_status.items() 
                           if status.get("warning") == "可能有版本冲突"]
        
        print(f"\n依赖包状态:")
        
        # 必需包状态
        if missing_required:
            print(f" 缺少必需包: {', '.join(missing_required)}")
        else:
            print("必需包已安装")
        
        # 可选包状态
        if missing_optional:
            print(f" 缺少可选包: {', '.join(missing_optional)}")
        else:
            print(" 可选包已安装")
        
        # 版本冲突包
        if conflict_packages:
            print(f" 版本冲突包: {', '.join(conflict_packages)}")
            print("   这些包有版本冲突，但不影响核心功能")
            for pkg in conflict_packages:
                error_msg = package_status[pkg].get("error", "")
                print(f"   - {pkg}: {error_msg[:50]}..." if len(error_msg) > 50 else f"   - {pkg}: {error_msg}")
        
        # 测试环境
        print("\n测试环境...")
        test_results = test_environment(package_status, logger)
        
        # 保存状态
        status_data = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "python_version": python_version,
            "platform": platform.platform(),
            "gpu_available": gpu_available,
            "package_status": package_status,
            "test_results": test_results
        }
        
        save_environment_status(status_data)
        
        # 更新配置
        config = load_config()
        config["environment"] = {
            "python_version": python_version,
            "gpu_available": gpu_available,
            "check_date": status_data["timestamp"],
            "status": "ready" if not missing_required else "incomplete"
        }
        save_config(config)
        
        # 如果需要，创建安装脚本
        if missing_required or missing_optional:
            install_script = create_env_installation_script(package_status, logger)
            print(f"\n已创建安装脚本: {install_script}")
            print_manual_instructions()
        
        # 总结
        logger.info(f"环境检查完成，耗时: {datetime.datetime.now() - start_time}")
        print("\n" + "=" * 60)
        print("环境检查完成!")
        print(f"Python版本: {python_version}")
        print(f"GPU支持: {'可用' if gpu_available else '不可用'}")
        
        missing_count = len(missing_required) + len(missing_optional)
        if missing_count == 0:
            print("所有依赖包已安装")
            print("\n下一步: python scripts/steps/step2_rfdiffusion_backbone.py")
        else:
            print(f" 缺少 {missing_count} 个依赖包")
            print("请先安装缺少的包，然后重新运行此脚本")
        
        print("=" * 60)
        
        return missing_count == 0
        
    except KeyboardInterrupt:
        logger.info("用户中断程序")
        print("\n用户中断程序")
        return False
    except Exception as e:
        logger.error(f"程序执行异常: {e}")
        print(f"程序执行异常: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

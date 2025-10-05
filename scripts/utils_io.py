"""
脚本名: utils_io.py
作者: Kuroneko
日期: 2025.9.30

功能: 通用 I/O、日志、文件路径管理、配置读取工具模块

输入文件:
- config.json (项目根目录，JSON格式配置文件)
- 各种数据文件路径 (相对路径)

输出文件:
- logs/stepX_YYYYMMDD_HHMMSS.log (日志文件)
- config.json (更新后的配置文件)

运行示例:
import utils_io as uio
config = uio.load_config()
logger = uio.setup_logger("step1")
uio.ensure_dir("models/huggingface")

依赖: 无特殊依赖，标准库
"""

import os
import json
import logging
import datetime
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# 项目根目录 (相对于脚本所在位置)
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

def get_project_root() -> Path:
    """获取项目根目录路径"""
    return PROJECT_ROOT

def ensure_dir(dir_path: str) -> Path:
    """确保目录存在，不存在则创建
    
    Args:
        dir_path: 相对于项目根的目录路径
        
    Returns:
        Path: 绝对路径对象
    """
    full_path = PROJECT_ROOT / dir_path
    full_path.mkdir(parents=True, exist_ok=True)
    return full_path

def get_abs_path(relative_path: str) -> Path:
    """将相对路径转换为绝对路径
    
    Args:
        relative_path: 相对于项目根的路径
        
    Returns:
        Path: 绝对路径对象
    """
    return PROJECT_ROOT / relative_path

def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """加载配置文件
    
    Args:
        config_path: 配置文件路径（相对于项目根）
        
    Returns:
        Dict: 配置字典
    """
    config_file = get_abs_path(config_path)
    
    # 默认配置
    default_config = {
        "project_name": "de_novo_protein_design",
        "author": "Kuroneko",
        "version": "1.0.0",
        "created_date": "2025.9.30",
        "last_updated": "",
        "current_iteration": 1,
        "max_iterations": 10,
        "parameters": {
            "num_backbones": 200,
            "num_sequences_per_backbone": 50,
            "esmfold_pLDDT_threshold": 60.0,
            "colabfold_use_online_msa": "yes",
            "max_md_ns": 10,
            "rosetta_relax_rounds": 2
        },
        "model_versions": {},
        "iteration_history": []
    }
    
    if config_file.exists():
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            # 合并默认配置和现有配置
            for key, value in default_config.items():
                if key not in config:
                    config[key] = value
        except Exception as e:
            print(f"警告: 无法读取配置文件 {config_file}: {e}")
            print("使用默认配置")
            config = default_config
    else:
        config = default_config
        save_config(config, config_path)
    
    return config

def save_config(config: Dict[str, Any], config_path: str = "config.json") -> None:
    """保存配置文件
    
    Args:
        config: 配置字典
        config_path: 配置文件路径（相对于项目根）
    """
    config_file = get_abs_path(config_path)
    config["last_updated"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"错误: 无法保存配置文件 {config_file}: {e}")

def setup_logger(step_name: str, log_level: str = "INFO") -> logging.Logger:
    """设置日志记录器
    
    Args:
        step_name: 步骤名称 (如 "step1", "step2")
        log_level: 日志级别
        
    Returns:
        logging.Logger: 配置好的日志记录器
    """
    ensure_dir("logs")
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = get_abs_path(f"logs/{step_name}_{timestamp}.log")
    
    logger = logging.getLogger(step_name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # 清除现有处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    logger.info(f"开始执行 {step_name}")
    logger.info(f"项目根目录: {PROJECT_ROOT}")
    logger.info(f"日志文件: {log_file}")
    
    return logger

def validate_input(prompt: str, input_type: type, 
                  valid_range: Optional[tuple] = None, 
                  valid_choices: Optional[list] = None,
                  default_value: Any = None) -> Any:
    """验证用户输入
    
    Args:
        prompt: 提示信息
        input_type: 期望的输入类型 (int, float, str)
        valid_range: 有效范围 (min, max) 仅适用于数值类型
        valid_choices: 有效选择列表
        default_value: 默认值
        
    Returns:
        Any: 验证后的用户输入
    """
    while True:
        try:
            if default_value is not None:
                user_input = input(f"{prompt} (默认: {default_value}): ").strip()
                if not user_input:
                    return default_value
            else:
                user_input = input(f"{prompt}: ").strip()
            
            # 类型转换
            if input_type == int:
                value = int(user_input)
            elif input_type == float:
                value = float(user_input)
            else:
                value = user_input
            
            # 验证范围
            if valid_range and input_type in [int, float]:
                if not (valid_range[0] <= value <= valid_range[1]):
                    print(f"输入值必须在 {valid_range[0]} 到 {valid_range[1]} 之间")
                    continue
            
            # 验证选择
            if valid_choices and value not in valid_choices:
                print(f"输入值必须是以下选项之一: {valid_choices}")
                continue
                
            return value
            
        except ValueError:
            print(f"输入类型错误，期望 {input_type.__name__} 类型")
        except KeyboardInterrupt:
            print("\n用户中断程序")
            sys.exit(1)

def record_model_download(model_name: str, version: str, cache_dir: str) -> None:
    """记录模型下载信息到 manifest.txt
    
    Args:
        model_name: 模型名称
        version: 模型版本
        cache_dir: 缓存目录
    """
    manifest_file = get_abs_path("models/manifest.txt")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(manifest_file, 'a', encoding='utf-8') as f:
        f.write(f"{timestamp} | {model_name} | {version} | {cache_dir}\n")

def get_iteration_dir(step_name: str, iteration: int) -> Path:
    """获取迭代目录路径
    
    Args:
        step_name: 步骤名称
        iteration: 迭代次数
        
    Returns:
        Path: 迭代目录路径
    """
    if step_name.startswith("step2"):  # backbone生成
        base_dir = "backbones"
    elif step_name.startswith("step3"):  # 序列设计
        base_dir = "designs"
    else:
        base_dir = "output"
    
    iter_dir = ensure_dir(f"{base_dir}/iter{iteration}")
    return iter_dir

def list_pdb_files(directory: str) -> list:
    """列出目录中的所有PDB文件
    
    Args:
        directory: 目录路径（相对于项目根）
        
    Returns:
        list: PDB文件路径列表
    """
    dir_path = get_abs_path(directory)
    if not dir_path.exists():
        return []
    
    pdb_files = list(dir_path.glob("*.pdb"))
    return [str(f.relative_to(PROJECT_ROOT)) for f in pdb_files]

def list_json_files(directory: str) -> list:
    """列出目录中的所有JSON文件
    
    Args:
        directory: 目录路径（相对于项目根）
        
    Returns:
        list: JSON文件路径列表
    """
    dir_path = get_abs_path(directory)
    if not dir_path.exists():
        return []
    
    json_files = list(dir_path.glob("*.json"))
    return [str(f.relative_to(PROJECT_ROOT)) for f in json_files]

def check_dependencies() -> Dict[str, bool]:
    """检查依赖包是否安装
    
    Returns:
        Dict[str, bool]: 依赖包安装状态
    """
    dependencies = {
        "torch": False,
        "transformers": False,
        "biopython": False,
        "numpy": False,
        "pandas": False,
        "requests": False
    }
    
    for package in dependencies.keys():
        try:
            __import__(package)
            dependencies[package] = True
        except ImportError:
            dependencies[package] = False
    
    return dependencies

def format_time_elapsed(start_time: datetime.datetime) -> str:
    """格式化耗时
    
    Args:
        start_time: 开始时间
        
    Returns:
        str: 格式化的耗时字符串
    """
    elapsed = datetime.datetime.now() - start_time
    hours, remainder = divmod(elapsed.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

if __name__ == "__main__":
    # 测试工具函数
    print("工具模块测试")
    print(f"项目根目录: {get_project_root()}")
    
    # 测试配置加载
    config = load_config()
    print(f"加载的配置: {json.dumps(config, indent=2, ensure_ascii=False)}")
    
    # 测试日志
    logger = setup_logger("test")
    logger.info("这是一条测试日志")
    
    # 测试依赖检查
    deps = check_dependencies()
    print(f"依赖包状态: {deps}")
    
    print("工具模块测试完成")

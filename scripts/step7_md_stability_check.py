"""
脚本名: step7_md_stability_check.py
作者: Kuroneko
日期: 2025.7.30

功能: 短时分子动力学模拟稳定性检查（如无 MD 环境则使用简化检查）

输入文件:
- preds/iterN/*.pdb (预测结构文件)
- scores/rosetta_scores_iterN.csv (可选，Rosetta评分)
- config.json (全局配置)

输出文件:
- scores/md_stability_iterN.csv (MD 稳定性分析)
- scores/md_trajectories/ (轨迹文件，可选)
- logs/step7_YYYYMMDD_HHMMSS.log (执行日志)

运行示例:
python scripts/step7_md_stability_check.py

依赖: 
- GROMACS/OpenMM（可选，未安装时使用替代检查）
- MDAnalysis（用于轨迹分析）
- 或使用简化的热力学稳定性评估

参数示例:
- max_md_ns: 10 (0-1000，模拟时长纳秒，0=跳过MD)
- md_method: 'gromacs'/'openmm'/'simple'
- temperature: 300 (250-400K)
- top_candidates: 20 (1-100，选择最佳候选进行MD)
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import datetime
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
try:
    from utils_io import (
        get_project_root, ensure_dir, get_abs_path, setup_logger, 
        validate_input, save_config, load_config, format_time_elapsed
    )
except ImportError:
    print("错误: 无法导入 utils_io.py，请确保文件存在")
    sys.exit(1)

def check_md_software() -> Dict[str, Tuple[bool, str]]:
    """检查分子动力学软件可用性"""
    software_status = {}
    
    # 检查 GROMACS
    try:
        result = subprocess.run(['gmx', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            version = result.stdout.split('\n')[0] if result.stdout else "Unknown"
            software_status['gromacs'] = (True, f"GROMACS 可用: {version}")
        else:
            software_status['gromacs'] = (False, "GROMACS 命令失败")
    except Exception:
        software_status['gromacs'] = (False, "GROMACS 未安装")
    
    # 检查 OpenMM
    try:
        import openmm
        software_status['openmm'] = (True, f"OpenMM 可用: {openmm.__version__}")
    except ImportError:
        software_status['openmm'] = (False, "OpenMM 未安装")
    
    # 检查 MDAnalysis
    try:
        import MDAnalysis
        software_status['mdanalysis'] = (True, f"MDAnalysis 可用: {MDAnalysis.__version__}")
    except ImportError:
        software_status['mdanalysis'] = (False, "MDAnalysis 未安装")
    
    return software_status

def select_candidates_for_md(current_iter: int, top_n: int, logger) -> List[Dict]:
    """选择用于MD模拟的候选结构"""
    candidates = []
    
    # 优先使用前N名序列JSON文件（来自第六步）
    top_sequences_file = get_abs_path(f"scores/top_{top_n}_sequences_iter{current_iter}.json")
    if top_sequences_file.exists():
        try:
            with open(top_sequences_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for seq_info in data.get('top_sequences', []):
                pdb_file = get_abs_path(f"scores/{seq_info['pdb_file']}")
                if pdb_file.exists():
                    candidates.append({
                        "sequence_id": seq_info['sequence_id'],
                        "pdb_file": str(pdb_file),
                        "rosetta_score": seq_info.get('total_score', 0),
                        "selection_method": "top_sequences"
                    })
            logger.info(f"从top_sequences文件加载了 {len(candidates)} 个候选结构")
        except Exception as e:
            logger.warning(f"读取top_sequences文件失败: {e}")
    
    # 如果没有足够的候选，使用 Rosetta 评分结果
    if len(candidates) < top_n:
        rosetta_file = get_abs_path(f"scores/rosetta_scores_iter{current_iter}.csv")
        if rosetta_file.exists():
            try:
                df = pd.read_csv(rosetta_file)
                df_sorted = df.sort_values('total_score').head(top_n)  # Rosetta分数越低越好
                
                for _, row in df_sorted.iterrows():
                    # 检查是否已经在候选列表中
                    if row['sequence_id'] not in [c['sequence_id'] for c in candidates]:
                        # 优先在scores文件夹中查找PDB文件
                        pdb_file = get_abs_path(f"scores/{row['pdb_file']}")
                        if pdb_file.exists():
                            candidates.append({
                                "sequence_id": row['sequence_id'],
                                "pdb_file": str(pdb_file),
                                "rosetta_score": row['total_score'],
                                "selection_method": "rosetta_score"
                            })
                        else:
                            # 备选：在prediction_results中查找
                            preds_dir = get_abs_path("output/prediction_results")
                            if preds_dir.exists():
                                for subdir in preds_dir.iterdir():
                                    if subdir.is_dir() and row['sequence_id'] in subdir.name:
                                        potential_pdb = subdir / row['pdb_file']
                                        if potential_pdb.exists():
                                            candidates.append({
                                                "sequence_id": row['sequence_id'],
                                                "pdb_file": str(potential_pdb),
                                                "rosetta_score": row['total_score'],
                                                "selection_method": "rosetta_score"
                                            })
                                            break
                
                logger.info(f"基于 Rosetta 评分选择了 {len(candidates)} 个候选")
                
            except Exception as e:
                logger.warning(f"无法读取 Rosetta 评分: {e}")
    
    # 如果仍然没有足够的候选，使用 ESMFold 结果
    if len(candidates) < top_n:
        esmfold_file = get_abs_path(f"output/prediction_results/esmfold_summary.csv")
        if esmfold_file.exists():
            try:
                df = pd.read_csv(esmfold_file)
                df_filtered = df[df['prediction_success'] == True]
                df_sorted = df_filtered.sort_values('mean_plddt', ascending=False)
                
                needed = top_n - len(candidates)
                for _, row in df_sorted.head(needed).iterrows():
                    if row['sequence_id'] not in [c['sequence_id'] for c in candidates]:
                        # 在prediction_results中查找PDB文件
                        preds_dir = get_abs_path("output/prediction_results")
                        if preds_dir.exists():
                            for subdir in preds_dir.iterdir():
                                if subdir.is_dir() and row['sequence_id'] in subdir.name:
                                    potential_pdb = subdir / row['pdb_file']
                                    if potential_pdb.exists():
                                        candidates.append({
                                            "sequence_id": row['sequence_id'],
                                            "pdb_file": str(potential_pdb),
                                            "esmfold_plddt": row['mean_plddt'],
                                            "selection_method": "esmfold_plddt"
                                        })
                                        break
                
                logger.info(f"额外基于 ESMFold pLDDT 选择了 {len(candidates)} 个候选")
                
            except Exception as e:
                logger.warning(f"无法读取 ESMFold 结果: {e}")
    
    return candidates[:top_n]

def run_simple_stability_check(pdb_file: str, logger) -> Dict[str, float]:
    """运行简化的稳定性检查"""
    stability_scores = {
        "geometric_stability": 0.0,
        "thermal_b_factor": 0.0,
        "surface_exposure": 0.0,
        "loop_flexibility": 0.0,
        "overall_stability": 0.0
    }
    
    try:
        from Bio import PDB
        
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure("protein", pdb_file)
        
        # 提取原子信息
        ca_coords = []
        b_factors = []
        
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.has_id('CA'):
                        ca_atom = residue['CA']
                        ca_coords.append(ca_atom.get_coord())
                        b_factors.append(ca_atom.get_bfactor())
        
        ca_coords = np.array(ca_coords)
        b_factors = np.array(b_factors)
        
        if len(ca_coords) < 5:
            logger.warning(f"结构过短，无法进行稳定性检查: {pdb_file}")
            return stability_scores
        
        # 1. 几何稳定性（基于局部结构规律性）
        geometric_stability = calculate_geometric_stability(ca_coords)
        stability_scores["geometric_stability"] = geometric_stability
        
        # 2. 热力学B因子分析
        thermal_stability = calculate_thermal_stability(b_factors)
        stability_scores["thermal_b_factor"] = thermal_stability
        
        # 3. 表面暴露估算
        surface_exposure = calculate_surface_exposure(ca_coords)
        stability_scores["surface_exposure"] = surface_exposure
        
        # 4. 柔性区域识别
        loop_flexibility = calculate_loop_flexibility(ca_coords)
        stability_scores["loop_flexibility"] = loop_flexibility
        
        # 综合稳定性评分（0-100，越高越稳定）
        stability_scores["overall_stability"] = (
            geometric_stability * 0.3 +
            thermal_stability * 0.3 +
            (100 - surface_exposure) * 0.2 +  # 表面暴露越少越稳定
            (100 - loop_flexibility) * 0.2    # 柔性越小越稳定
        )
        
        logger.info(f"简化稳定性检查完成: {Path(pdb_file).name}, 稳定性: {stability_scores['overall_stability']:.2f}")
        
    except Exception as e:
        logger.error(f"简化稳定性检查失败 {pdb_file}: {e}")
        stability_scores["overall_stability"] = 50.0  # 默认中等稳定性
    
    return stability_scores

def calculate_geometric_stability(coords: np.ndarray) -> float:
    """计算几何稳定性"""
    if len(coords) < 5:
        return 50.0
    
    # 计算相邻CA原子距离的一致性
    distances = []
    for i in range(len(coords) - 1):
        dist = np.linalg.norm(coords[i+1] - coords[i])
        distances.append(dist)
    
    distances = np.array(distances)
    
    # 理想的CA-CA距离约为3.8Å
    ideal_distance = 3.8
    distance_deviations = np.abs(distances - ideal_distance)
    
    # 计算稳定性（偏差越小越稳定）
    mean_deviation = np.mean(distance_deviations)
    stability = max(0, 100 - mean_deviation * 50)  # 标准化到0-100
    
    return stability

def calculate_thermal_stability(b_factors: np.ndarray) -> float:
    """基于B因子计算热力学稳定性"""
    if len(b_factors) == 0:
        return 50.0
    
    # B因子越低表示越稳定
    mean_b_factor = np.mean(b_factors)
    
    # 将B因子转换为稳定性分数（典型B因子范围 10-100）
    stability = max(0, 100 - mean_b_factor)
    stability = min(100, stability)
    
    return stability

def calculate_surface_exposure(coords: np.ndarray) -> float:
    """计算表面暴露度"""
    if len(coords) < 10:
        return 50.0
    
    # 计算每个残基的邻居数量（简化的溶剂可及表面积）
    cutoff = 8.0  # 邻居距离阈值
    exposures = []
    
    for i, coord in enumerate(coords):
        neighbors = 0
        for j, other_coord in enumerate(coords):
            if i != j:
                distance = np.linalg.norm(coord - other_coord)
                if distance < cutoff:
                    neighbors += 1
        
        # 邻居越少，表面暴露度越高
        max_neighbors = min(len(coords) - 1, 20)
        exposure = (max_neighbors - neighbors) / max_neighbors * 100
        exposures.append(exposure)
    
    mean_exposure = np.mean(exposures)
    return mean_exposure

def calculate_loop_flexibility(coords: np.ndarray) -> float:
    """计算柔性/环区域"""
    if len(coords) < 10:
        return 50.0
    
    # 基于局部结构的规律性计算柔性
    flexibilities = []
    window_size = 5
    
    for i in range(window_size, len(coords) - window_size):
        # 计算窗口内的结构规律性
        window_coords = coords[i-window_size:i+window_size+1]
        
        # 计算二阶导数（曲率）
        if len(window_coords) >= 3:
            # 中心差分近似
            curvature = np.linalg.norm(
                window_coords[:-2] - 2*window_coords[1:-1] + window_coords[2:]
            )
            flexibilities.append(curvature)
    
    if not flexibilities:
        return 50.0
    
    mean_flexibility = np.mean(flexibilities)
    
    # 标准化柔性分数
    flexibility_score = min(100, mean_flexibility * 10)
    
    return flexibility_score

def run_openmm_simulation(pdb_file: str, params: Dict, logger) -> Dict[str, float]:
    """使用 OpenMM 运行简短MD模拟"""
    logger.info(f"使用 OpenMM 运行MD模拟: {Path(pdb_file).name}")
    
    md_results = {
        "simulation_time_ns": 0.0,
        "final_rmsd": 0.0,
        "average_rmsd": 0.0,
        "potential_energy": 0.0,
        "simulation_success": False
    }
    
    try:
        import openmm as mm
        import openmm.app as app
        import openmm.unit as unit
        from openmm import LangevinMiddleIntegrator
        
        # 加载PDB结构
        pdb = app.PDBFile(pdb_file)
        
        # 创建力场
        forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
        
        # 创建系统
        system = forcefield.createSystem(
            pdb.topology,
            nonbondedMethod=app.PME,
            nonbondedCutoff=1*unit.nanometer,
            constraints=app.HBonds
        )
        
        # 创建积分器
        integrator = LangevinMiddleIntegrator(
            params.get('temperature', 300)*unit.kelvin,
            1/unit.picosecond,
            0.002*unit.picoseconds
        )
        
        # 创建模拟
        simulation = app.Simulation(pdb.topology, system, integrator)
        simulation.context.setPositions(pdb.positions)
        
        # 能量最小化
        simulation.minimizeEnergy()
        
        # 平衡
        simulation.context.setVelocitiesToTemperature(params.get('temperature', 300)*unit.kelvin)
        simulation.step(1000)  # 2ps 平衡
        
        # 记录初始位置
        initial_positions = simulation.context.getState(getPositions=True).getPositions()
        
        # 生产模拟
        simulation_ns = params.get('max_md_ns', 1.0)
        steps = int(simulation_ns * 1000 / 0.002)  # 2fs步长
        
        rmsds = []
        
        for step in range(0, steps, 1000):  # 每2ps记录一次
            simulation.step(1000)
            
            current_positions = simulation.context.getState(getPositions=True).getPositions()
            
            # 计算RMSD
            rmsd = calculate_rmsd_openmm(initial_positions, current_positions)
            rmsds.append(rmsd)
        
        # 计算结果
        md_results.update({
            "simulation_time_ns": simulation_ns,
            "final_rmsd": rmsds[-1] if rmsds else 0.0,
            "average_rmsd": np.mean(rmsds) if rmsds else 0.0,
            "potential_energy": simulation.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole),
            "simulation_success": True
        })
        
        logger.info(f"OpenMM 模拟成功: RMSD={md_results['average_rmsd']:.2f} Å")
        
    except Exception as e:
        logger.error(f"OpenMM 模拟失败 {pdb_file}: {e}")
        md_results["simulation_success"] = False
    
    return md_results

def calculate_rmsd_openmm(pos1, pos2):
    """计算两组坐标的RMSD"""
    import openmm.unit as unit
    
    # 转换为numpy数组
    coords1 = np.array([[pos.x, pos.y, pos.z] for pos in pos1]) * 10  # nm to Å
    coords2 = np.array([[pos.x, pos.y, pos.z] for pos in pos2]) * 10
    
    # 计算RMSD
    diff = coords1 - coords2
    rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
    
    return rmsd

def save_md_results(results: List[Dict], current_iter: int, output_dir: Path) -> str:
    """保存MD分析结果"""
    if not results:
        return ""
    
    df = pd.DataFrame(results)
    csv_file = output_dir / f"md_stability_iter{current_iter}.csv"
    df.to_csv(csv_file, index=False)
    
    return str(csv_file)

def main():
    """主函数"""
    print("=" * 60)
    print("Step 7: 分子动力学稳定性检查")
    print("作者: Kuroneko | 日期: 2025.9.30")
    print("=" * 60)
    
    logger = setup_logger("step7")
    start_time = datetime.datetime.now()
    
    try:
        # 检查MD软件
        software_status = check_md_software()
        logger.info("MD软件检查结果:")
        for software, (available, info) in software_status.items():
            logger.info(f"  {software}: {info}")
            print(f"✓ {software}: {info}")
        
        # 加载配置
        config = load_config()
        current_iter = config.get("current_iteration", 1)
        
        # 获取用户参数
        print("\n请输入MD参数:")
        
        max_md_ns = validate_input(
            "MD模拟时长(纳秒) - 0表示跳过MD使用简化检查",
            int,
            valid_range=(0, 1000),
            default_value=config["parameters"].get("max_md_ns", 10)
        )
        
        if max_md_ns == 0:
            print("选择跳过MD模拟，将使用简化稳定性检查")
            md_method = "simple"
        else:
            available_methods = ["simple"]
            if software_status['openmm'][0]:
                available_methods.append("openmm")
            if software_status['gromacs'][0]:
                available_methods.append("gromacs")
            
            md_method = validate_input(
                "MD方法",
                str,
                valid_choices=available_methods,
                default_value="openmm" if "openmm" in available_methods else "simple"
            )
        
        top_candidates = validate_input(
            "选择最佳候选进行MD检查",
            int,
            valid_range=(1, 100),
            default_value=20
        )
        
        temperature = validate_input(
            "模拟温度(K)",
            int,
            valid_range=(250, 400),
            default_value=300
        ) if md_method != "simple" else 300
        
        # 选择候选结构
        print(f"\n选择前 {top_candidates} 个候选结构...")
        candidates = select_candidates_for_md(current_iter, top_candidates, logger)
        
        if not candidates:
            print("错误: 未找到合适的候选结构")
            print("请先运行 step4-step6 生成预测和评分结果")
            return False
        
        print(f"✓ 选择了 {len(candidates)} 个候选结构")
        
        # 设置输出目录
        output_dir = ensure_dir("scores")
        
        # 准备参数
        params = {
            "max_md_ns": max_md_ns,
            "md_method": md_method,
            "temperature": temperature,
            "top_candidates": top_candidates,
            "iteration": current_iter
        }
        
        logger.info(f"MD参数: {json.dumps(params, indent=2)}")
        
        # 开始MD检查
        print(f"\n开始稳定性检查 (方法: {md_method})...")
        
        results = []
        
        for i, candidate in enumerate(candidates):
            print(f"处理候选 {i+1}/{len(candidates)}: {candidate['sequence_id']}")
            logger.info(f"处理候选: {candidate['sequence_id']}")
            
            pdb_file = candidate['pdb_file']
            if not Path(pdb_file).exists():
                logger.warning(f"PDB文件不存在: {pdb_file}")
                continue
            
            # 执行稳定性检查
            if md_method == "openmm" and software_status['openmm'][0]:
                md_result = run_openmm_simulation(pdb_file, params, logger)
            elif md_method == "gromacs" and software_status['gromacs'][0]:
                # GROMACS实现可以在这里添加
                logger.warning("GROMACS 实现尚未完成，使用简化检查")
                md_result = {"simulation_success": False}
            else:
                md_result = {"simulation_success": False}
            
            # 简化稳定性检查（作为备选或补充）
            stability_result = run_simple_stability_check(pdb_file, logger)
            
            # 合并结果
            result = {
                "sequence_id": candidate['sequence_id'],
                "pdb_file": Path(pdb_file).name,
                "md_method": md_method,
                **candidate,
                **stability_result,
                **md_result
            }
            
            results.append(result)
            
            if md_result.get("simulation_success", False):
                print(f"  MD成功: RMSD={md_result.get('average_rmsd', 0):.2f}Å, 稳定性={stability_result.get('overall_stability', 0):.1f}")
            else:
                print(f"  简化检查: 稳定性={stability_result.get('overall_stability', 0):.1f}")
        
        # 保存结果
        if results:
            csv_file = save_md_results(results, current_iter, output_dir)
            logger.info(f"MD结果已保存: {csv_file}")
            
            # 统计信息
            df = pd.DataFrame(results)
            mean_stability = df['overall_stability'].mean()
            successful_md = len(df[df.get('simulation_success', False) == True])
            
            print(f"\n稳定性检查统计:")
            print(f"平均稳定性: {mean_stability:.2f}")
            print(f"成功MD模拟: {successful_md}/{len(results)}")
        
        # 更新配置
        config["parameters"]["max_md_ns"] = max_md_ns
        
        # 记录迭代历史
        iteration_record = {
            "iteration": current_iter,
            "step": "step7_md_stability",
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "parameters": params,
            "results": {
                "candidates_processed": len(results),
                "md_method": md_method,
                "successful_simulations": len([r for r in results if r.get('simulation_success', False)]),
                "execution_time": (datetime.datetime.now() - start_time).total_seconds()
            }
        }
        config["iteration_history"].append(iteration_record)
        save_config(config)
        
        # 总结
        print("\n" + "=" * 60)
        print("Step 7 执行完成!")
        print(f"检查方法: {md_method}")
        print(f"处理候选: {len(results)} 个")
        print(f"输出目录: {output_dir.relative_to(get_project_root())}")
        print(f"执行时间: {format_time_elapsed(start_time)}")
        
        if results:
            print("✓ 稳定性检查成功")
            print("\n下一步: python scripts/step8_iterate_and_select.py")
        else:
            print("✗ 稳定性检查失败")
        
        print("=" * 60)
        
        return len(results) > 0
        
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

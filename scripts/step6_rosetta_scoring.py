"""
脚本名: step6_rosetta_scoring.py
作者: Kuroneko
日期: 2025.7.30

功能: 使用 Rosetta 能量打分与 relax（如无 Rosetta 环境则提供替代评分）

输入文件:
- preds/iterN/*.pdb (ESMFold/ColabFold 预测结构)
- config.json (全局配置)

输出文件:
- scores/rosetta_scores_iterN.csv (Rosetta 能量评分)
- scores/relaxed_structures/ (优化后的结构，可选)
- logs/step6_YYYYMMDD_HHMMSS.log (执行日志)

运行示例:
python scripts/step6_rosetta_scoring.py

依赖: 
- Rosetta 软件包（可选，未安装时使用替代方法）
- PyRosetta（可选）
- 或使用简化的几何/物理评分

参数示例:
- rosetta_relax_rounds: 2 (0-5，优化轮数)
- scoring_method: 'rosetta'/'alternative'/'both'
- max_structures: 50 (1-500，最大处理结构数)
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

def check_rosetta_available() -> Tuple[bool, str]:
    """检查 Rosetta 是否可用"""
    try:
        # 尝试 PyRosetta
        import pyrosetta
        return True, "PyRosetta 可用"
    except ImportError:
        pass
    
    try:
        # 尝试命令行 Rosetta
        result = subprocess.run(['score_jd2', '-help'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0 or "Rosetta" in result.stderr:
            return True, "Rosetta 命令行工具可用"
    except Exception:
        pass
    
    return False, "Rosetta 未安装"

def check_biopython_available() -> bool:
    """检查 BioPython 是否可用"""
    try:
        from Bio import PDB
        return True
    except ImportError:
        return False

def calculate_alternative_scores(pdb_file: Path, logger) -> Dict[str, float]:
    """使用替代方法计算结构评分"""
    scores = {
        "clash_score": 0.0,
        "rama_score": 0.0,
        "solvation_score": 0.0,
        "compactness_score": 0.0,
        "total_alternative_score": 0.0
    }
    
    try:
        if not check_biopython_available():
            logger.warning("BioPython 不可用，使用简化评分")
            # 基于文件大小的简化评分
            file_size = pdb_file.stat().st_size
            scores["total_alternative_score"] = max(0, 100 - file_size / 1000)
            return scores
        
        from Bio import PDB
        
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure("protein", pdb_file)
        
        # 提取原子坐标
        coords = []
        residues = []
        
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.has_id('CA'):
                        ca_atom = residue['CA']
                        coords.append(ca_atom.get_coord())
                        residues.append(residue)
        
        coords = np.array(coords)
        
        if len(coords) < 3:
            logger.warning(f"结构过短，无法评分: {pdb_file.name}")
            return scores
        
        # 1. 原子冲突评分 (clash score)
        clash_score = calculate_clash_score(coords)
        scores["clash_score"] = clash_score
        
        # 2. 拉马钱德兰评分 (simplified)
        rama_score = calculate_rama_score(residues)
        scores["rama_score"] = rama_score
        
        # 3. 溶剂化评分 (simplified)
        solvation_score = calculate_solvation_score(coords)
        scores["solvation_score"] = solvation_score
        
        # 4. 紧密度评分
        compactness_score = calculate_compactness_score(coords)
        scores["compactness_score"] = compactness_score
        
        # 综合评分（越低越好，模拟 Rosetta）
        scores["total_alternative_score"] = (
            clash_score * 2.0 +
            rama_score * 1.5 +
            solvation_score * 1.0 +
            compactness_score * 0.5
        )
        
        logger.info(f"替代评分完成: {pdb_file.name}, 总分: {scores['total_alternative_score']:.2f}")
        
    except Exception as e:
        logger.error(f"替代评分失败 {pdb_file.name}: {e}")
        scores["total_alternative_score"] = 999.0  # 高分表示差结构
    
    return scores

def calculate_clash_score(coords: np.ndarray) -> float:
    """计算原子冲突评分"""
    if len(coords) < 2:
        return 0.0
    
    # 计算所有原子对距离
    from scipy.spatial.distance import pdist
    distances = pdist(coords)
    
    # 冲突阈值（CA-CA 距离应 > 3.8 Å）
    clash_threshold = 3.8
    clashes = np.sum(distances < clash_threshold)
    
    # 标准化为 0-100 分数
    clash_score = min(100.0, clashes * 10.0)
    
    return clash_score

def calculate_rama_score(residues: List) -> float:
    """计算简化的拉马钱德兰评分"""
    if len(residues) < 3:
        return 0.0
    
    # 简化：基于相邻 CA 原子的夹角
    angles = []
    
    try:
        for i in range(1, len(residues) - 1):
            if (residues[i-1].has_id('CA') and 
                residues[i].has_id('CA') and 
                residues[i+1].has_id('CA')):
                
                ca1 = residues[i-1]['CA'].get_coord()
                ca2 = residues[i]['CA'].get_coord()
                ca3 = residues[i+1]['CA'].get_coord()
                
                # 计算夹角
                v1 = ca1 - ca2
                v2 = ca3 - ca2
                
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle) * 180.0 / np.pi
                
                angles.append(angle)
        
        if not angles:
            return 0.0
        
        angles = np.array(angles)
        
        # 理想的 CA-CA-CA 角度约为 90-120 度
        ideal_range = (90, 120)
        bad_angles = np.sum((angles < ideal_range[0]) | (angles > ideal_range[1]))
        
        rama_score = (bad_angles / len(angles)) * 100.0
        
    except Exception:
        rama_score = 50.0  # 默认中等分数
    
    return rama_score

def calculate_solvation_score(coords: np.ndarray) -> float:
    """计算简化的溶剂化评分"""
    if len(coords) < 4:
        return 0.0
    
    # 计算蛋白质的表面积/体积比
    center = np.mean(coords, axis=0)
    distances_to_center = np.linalg.norm(coords - center, axis=1)
    
    # 计算回转半径
    radius_of_gyration = np.sqrt(np.mean(distances_to_center**2))
    
    # 理想的回转半径 (rough estimate for globular proteins)
    expected_rg = 2.2 * (len(coords) ** 0.38)  # Empirical formula
    
    # 偏差越大分数越高（越差）
    rg_deviation = abs(radius_of_gyration - expected_rg) / expected_rg
    solvation_score = min(100.0, rg_deviation * 100.0)
    
    return solvation_score

def calculate_compactness_score(coords: np.ndarray) -> float:
    """计算紧密度评分"""
    if len(coords) < 4:
        return 0.0
    
    # 计算所有原子对的平均距离
    from scipy.spatial.distance import pdist
    distances = pdist(coords)
    mean_distance = np.mean(distances)
    
    # 期望的平均距离（基于蛋白质大小）
    n_residues = len(coords)
    expected_distance = 8.0 + 2.0 * np.log(n_residues)  # Empirical
    
    # 偏差评分
    distance_deviation = abs(mean_distance - expected_distance) / expected_distance
    compactness_score = min(100.0, distance_deviation * 50.0)
    
    return compactness_score

def run_pyrosetta_scoring(pdb_files: List[Path], params: Dict, logger) -> List[Dict]:
    """使用 PyRosetta 进行评分"""
    logger.info("使用 PyRosetta 进行评分")
    
    try:
        import pyrosetta
        pyrosetta.init("-mute all")
        
        from pyrosetta import pose_from_pdb
        from pyrosetta.rosetta.core.scoring import get_score_function
        
        # 创建评分函数
        scorefxn = get_score_function()
        
        results = []
        
        for pdb_file in pdb_files:
            try:
                # 加载结构
                pose = pose_from_pdb(str(pdb_file))
                
                # 评分
                score = scorefxn(pose)
                
                # 如果需要 relax
                if params.get("relax_rounds", 0) > 0:
                    from pyrosetta.rosetta.protocols.relax import FastRelax
                    
                    relax = FastRelax()
                    relax.set_scorefxn(scorefxn)
                    relax.apply(pose)
                    
                    relaxed_score = scorefxn(pose)
                    
                    # 保存 relaxed 结构
                    relaxed_dir = ensure_dir("scores/relaxed_structures")
                    relaxed_file = relaxed_dir / f"relaxed_{pdb_file.name}"
                    pose.dump_pdb(str(relaxed_file))
                    
                    logger.info(f"Relax 完成: {pdb_file.name}, {score:.2f} -> {relaxed_score:.2f}")
                    score = relaxed_score
                
                result = {
                    "pdb_file": pdb_file.name,
                    "sequence_id": pdb_file.stem.replace("_esmfold", "").replace("_colabfold", ""),
                    "total_score": score,
                    "scoring_method": "pyrosetta"
                }
                
                results.append(result)
                logger.info(f"PyRosetta 评分: {pdb_file.name} = {score:.2f}")
                
            except Exception as e:
                logger.error(f"PyRosetta 评分失败 {pdb_file.name}: {e}")
                continue
        
        return results
        
    except Exception as e:
        logger.error(f"PyRosetta 初始化失败: {e}")
        return []

def run_alternative_scoring(pdb_files: List[Path], params: Dict, logger) -> List[Dict]:
    """使用替代方法进行评分"""
    logger.info("使用替代方法进行评分")
    
    results = []
    
    for pdb_file in pdb_files:
        try:
            scores = calculate_alternative_scores(pdb_file, logger)
            
            result = {
                "pdb_file": pdb_file.name,
                "sequence_id": pdb_file.stem.replace("_esmfold", "").replace("_colabfold", ""),
                "total_score": scores["total_alternative_score"],
                "clash_score": scores["clash_score"],
                "rama_score": scores["rama_score"],
                "solvation_score": scores["solvation_score"],
                "compactness_score": scores["compactness_score"],
                "scoring_method": "alternative"
            }
            
            results.append(result)
            
        except Exception as e:
            logger.error(f"替代评分失败 {pdb_file.name}: {e}")
            continue
    
    return results

def save_scoring_results(results: List[Dict], current_iter: int, output_dir: Path) -> str:
    """保存评分结果"""
    if not results:
        return ""
    
    df = pd.DataFrame(results)
    csv_file = output_dir / f"rosetta_scores_iter{current_iter}.csv"
    df.to_csv(csv_file, index=False)
    
    return str(csv_file)

def extract_top_sequences(scores_df, preds_dir, iteration, top_n=10, logger=None):
    """提取前N名最佳序列并保存为JSON格式"""
    # 按分数排序（分数越低越好）
    top_sequences = scores_df.nsmallest(top_n, 'total_score')
    
    # 构建序列信息
    top_sequences_data = []
    
    for _, row in top_sequences.iterrows():
        sequence_id = row['sequence_id']
        pdb_file = row['pdb_file']
        
        # 查找对应的序列信息
        sequence_info = {
            'rank': len(top_sequences_data) + 1,
            'sequence_id': sequence_id,
            'pdb_file': pdb_file,
            'total_score': float(row['total_score']),
            'clash_score': float(row.get('clash_score', 0)),
            'rama_score': float(row.get('rama_score', 0)),
            'solvation_score': float(row.get('solvation_score', 0)),
            'compactness_score': float(row.get('compactness_score', 0)),
            'scoring_method': row.get('scoring_method', 'unknown')
        }
        
        # 尝试从PDB文件或metrics.json中获取序列
        sequence = extract_sequence_from_pdb(preds_dir, pdb_file, sequence_id, logger)
        if sequence:
            sequence_info['sequence'] = sequence
            sequence_info['length'] = len(sequence)
        
        top_sequences_data.append(sequence_info)
    
    # 保存为JSON到scores文件夹
    scores_dir = get_project_root() / "scores"
    scores_dir.mkdir(exist_ok=True)
    
    json_file = scores_dir / f"top_{top_n}_sequences_iter{iteration}.json"
    
    result_data = {
        'extraction_info': {
            'timestamp': datetime.datetime.now().isoformat(),
            'iteration': iteration,
            'total_sequences_analyzed': len(scores_df),
            'top_n': top_n,
            'scoring_method': top_sequences_data[0]['scoring_method'] if top_sequences_data else 'unknown'
        },
        'top_sequences': top_sequences_data
    }
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    
    # 复制前N名的PDB文件到scores文件夹
    copy_top_pdb_files(top_sequences_data, preds_dir, scores_dir, logger)
    
    if logger:
        logger.info(f"前{top_n}名序列已保存: {json_file}")
    return top_sequences_data

def extract_sequence_from_pdb(preds_dir, pdb_file, sequence_id, logger=None):
    """从PDB文件或metrics.json中提取序列"""
    try:
        # 方法1: 从metrics.json中获取序列
        metrics_file = None
        for subdir in preds_dir.iterdir():
            if subdir.is_dir() and sequence_id in subdir.name:
                metrics_file = subdir / "metrics.json"
                if metrics_file.exists():
                    break
        
        if metrics_file and metrics_file.exists():
            with open(metrics_file, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
                if 'sequence' in metrics:
                    return metrics['sequence']
        
        # 方法2: 从PDB文件中提取序列
        pdb_path = None
        for subdir in preds_dir.iterdir():
            if subdir.is_dir():
                potential_pdb = subdir / pdb_file
                if potential_pdb.exists():
                    pdb_path = potential_pdb
                    break
        
        if pdb_path and pdb_path.exists():
            sequence = extract_sequence_from_pdb_file(pdb_path, logger)
            if sequence:
                return sequence
        
        # 方法3: 从merged_input_sequences.json中查找
        merged_file = get_project_root() / "output" / "prediction_results" / "merged_input_sequences.json"
        if merged_file.exists():
            with open(merged_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for seq in data.get('sequences', []):
                    if seq.get('id') == sequence_id:
                        return seq.get('sequence')
        
        return None
        
    except Exception as e:
        if logger:
            logger.warning(f"无法提取序列 {sequence_id}: {e}")
        return None

def extract_sequence_from_pdb_file(pdb_path, logger=None):
    """从PDB文件中提取序列"""
    try:
        sequence = ""
        with open(pdb_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                    res_name = line[17:20].strip()
                    # 将三字母氨基酸代码转换为单字母
                    aa_map = {
                        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
                        'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
                        'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
                        'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
                    }
                    if res_name in aa_map:
                        sequence += aa_map[res_name]
        return sequence if sequence else None
    except Exception as e:
        if logger:
            logger.warning(f"从PDB文件提取序列失败 {pdb_path}: {e}")
        return None

def copy_top_pdb_files(top_sequences_data, preds_dir, scores_dir, logger=None):
    """复制前N名的PDB文件到scores文件夹"""
    import shutil
    
    copied_count = 0
    for seq_info in top_sequences_data:
        sequence_id = seq_info['sequence_id']
        pdb_file = seq_info['pdb_file']
        
        # 查找源PDB文件
        source_pdb = None
        for subdir in preds_dir.iterdir():
            if subdir.is_dir():
                potential_pdb = subdir / pdb_file
                if potential_pdb.exists():
                    source_pdb = potential_pdb
                    break
        
        if source_pdb and source_pdb.exists():
            # 复制到scores文件夹
            dest_pdb = scores_dir / pdb_file
            try:
                shutil.copy2(source_pdb, dest_pdb)
                copied_count += 1
                if logger:
                    logger.info(f"复制PDB文件: {pdb_file}")
            except Exception as e:
                if logger:
                    logger.warning(f"复制PDB文件失败 {pdb_file}: {e}")
        else:
            if logger:
                logger.warning(f"未找到PDB文件: {pdb_file}")
    
    if logger:
        logger.info(f"成功复制 {copied_count} 个PDB文件到scores文件夹")

def main():
    """主函数"""
    print("=" * 60)
    print("Step 6: Rosetta 能量评分与优化")
    print("作者: Kuroneko | 日期: 2025.9.30")
    print("=" * 60)
    
    logger = setup_logger("step6")
    start_time = datetime.datetime.now()
    
    try:
        # 检查 Rosetta 可用性
        rosetta_available, rosetta_info = check_rosetta_available()
        logger.info(f"Rosetta 状态: {rosetta_info}")
        print(f"✓ Rosetta: {rosetta_info}")
        
        # 加载配置
        config = load_config()
        current_iter = config.get("current_iteration", 1)
        
        # 查找预测结构文件
        # 优先查找 output/prediction_results（第四步结果）
        preds_dir = get_abs_path("output/prediction_results")
        if not preds_dir.exists():
            # 备选：查找 preds/iterN（第五步结果）
            preds_dir = get_abs_path(f"preds/iter{current_iter}")
            if not preds_dir.exists():
                print(f"错误: 未找到预测结构目录")
                print(f"  已检查: output/prediction_results")
                print(f"  已检查: preds/iter{current_iter}")
                print("请先运行 step4 或 step5")
            return False
        
        # 查找 PDB 文件（可能在子目录中）
        pdb_files = []
        
        # 1. 直接在根目录查找
        pdb_files.extend(list(preds_dir.glob("*.pdb")))
        
        # 2. 在子目录中查找
        for subdir in preds_dir.iterdir():
            if subdir.is_dir():
                pdb_files.extend(list(subdir.glob("*.pdb")))
        
        # 3. 查找 ColabFold 结果
        colabfold_dir = preds_dir / "colabfold"
        if colabfold_dir.exists():
            pdb_files.extend(list(colabfold_dir.glob("*.pdb")))
        
        if not pdb_files:
            print("错误: 未找到结构文件")
            return False
        
        logger.info(f"找到 {len(pdb_files)} 个结构文件")
        print(f"✓ 找到 {len(pdb_files)} 个结构文件")
        
        # 获取用户参数
        print("\n请输入评分参数:")
        
        max_structures = validate_input(
            f"最大处理结构数量 (总共 {len(pdb_files)} 个)",
            int,
            valid_range=(1, len(pdb_files)),
            default_value=min(50, len(pdb_files))
        )
        
        if rosetta_available:
            scoring_method = validate_input(
                "评分方法",
                str,
                valid_choices=["rosetta", "alternative", "both"],
                default_value="rosetta"
            )
        else:
            print("Rosetta 不可用，将使用替代评分方法")
            scoring_method = "alternative"
        
        rosetta_relax_rounds = validate_input(
            "Rosetta relax 轮数 (0=仅评分)",
            int,
            valid_range=(0, 5),
            default_value=config["parameters"].get("rosetta_relax_rounds", 2)
        )
        
        # 设置输出目录
        output_dir = ensure_dir("scores")
        logger.info(f"输出目录: {output_dir}")
        
        # 准备参数
        params = {
            "scoring_method": scoring_method,
            "relax_rounds": rosetta_relax_rounds,
            "max_structures": max_structures,
            "iteration": current_iter
        }
        
        logger.info(f"评分参数: {json.dumps(params, indent=2)}")
        
        # 选择要处理的文件
        pdb_files_to_process = pdb_files[:max_structures]
        print(f"\n开始评分 {len(pdb_files_to_process)} 个结构...")
        
        all_results = []
        
        # Rosetta 评分
        if scoring_method in ["rosetta", "both"] and rosetta_available:
            print("执行 Rosetta 评分...")
            rosetta_results = run_pyrosetta_scoring(pdb_files_to_process, params, logger)
            all_results.extend(rosetta_results)
            print(f"✓ Rosetta 评分完成: {len(rosetta_results)} 个结构")
        
        # 替代评分
        if scoring_method in ["alternative", "both"] or not rosetta_available:
            print("执行替代评分...")
            alt_results = run_alternative_scoring(pdb_files_to_process, params, logger)
            all_results.extend(alt_results)
            print(f"✓ 替代评分完成: {len(alt_results)} 个结构")
        
        # 保存结果
        if all_results:
            csv_file = save_scoring_results(all_results, current_iter, output_dir)
            logger.info(f"评分结果已保存: {csv_file}")
            
            # 统计信息
            df = pd.DataFrame(all_results)
            mean_score = df['total_score'].mean()
            min_score = df['total_score'].min()
            max_score = df['total_score'].max()
            
            print(f"\n评分统计:")
            print(f"平均分数: {mean_score:.2f}")
            print(f"最佳分数: {min_score:.2f}")
            print(f"最差分数: {max_score:.2f}")
            
            # 提取前10名最佳序列
            top_10_sequences = extract_top_sequences(df, preds_dir, current_iter, logger=logger)
            
            print(f"\n前10名最佳序列:")
            for i, seq in enumerate(top_10_sequences, 1):
                print(f"  {i}. {seq['sequence_id']} - 分数: {seq['total_score']:.2f}")
        
        # 更新配置
        config["parameters"]["rosetta_relax_rounds"] = rosetta_relax_rounds
        
        # 记录迭代历史
        iteration_record = {
            "iteration": current_iter,
            "step": "step6_rosetta_scoring",
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "parameters": params,
            "results": {
                "structures_processed": len(all_results),
                "scoring_methods": scoring_method,
                "rosetta_available": rosetta_available,
                "execution_time": (datetime.datetime.now() - start_time).total_seconds()
            }
        }
        config["iteration_history"].append(iteration_record)
        save_config(config)
        
        # 总结
        execution_time = (datetime.datetime.now() - start_time).total_seconds()
        print("\n" + "=" * 60)
        print("Step 6 执行完成!")
        print(f"评分方法: {scoring_method}")
        print(f"处理结构: {len(all_results)} 个")
        print(f"输出目录: {output_dir.relative_to(get_project_root())}")
        print(f"执行时间: {format_time_elapsed(start_time)}")
        
        if all_results:
            print("✓ 能量评分成功")
            print("\n下一步: python scripts/step7_md_stability_check.py (可选)")
            print("或直接: python scripts/step8_iterate_and_select.py")
        else:
            print("✗ 能量评分失败")
        
        print("=" * 60)
        
        return len(all_results) > 0
        
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

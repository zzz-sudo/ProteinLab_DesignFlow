"""
脚本名: step8_iterate_and_select.py
作者: Kuroneko
日期: 2025.8.3

功能: 迭代筛选控制脚本，评估结果并决定下一步操作

输入文件:
- preds/iterN/esmfold_summary.csv (ESMFold 预测结果)
- preds/iterN/colabfold_summary.csv (ColabFold 高精度结果，可选)
- scores/rosetta_*.csv (Rosetta 评分，可选)
- config.json (全局配置)

输出文件:
- scores/summary_top_candidates.csv (最终候选排名)
- scores/top_candidates/ (最佳候选结构文件)
- scores/iter_history.csv (迭代历史记录)
- logs/step8_YYYYMMDD_HHMMSS.log (执行日志)

运行示例:
python scripts/step8_iterate_and_select.py

依赖: 
- 需要前面步骤的预测结果
- 可选择回退到 step2 或 step3 重新设计

参数示例:
- selection_criteria: 'plddt'/'combined'/'manual'
- top_n_candidates: 5 (1-20)
- min_plddt_threshold: 70.0 (50.0-90.0)
- action_choice: 'accept'/'retry_step2'/'retry_step3'/'analyze'
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import datetime
import shutil
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

def load_all_results(current_iter: int, logger) -> Dict[str, pd.DataFrame]:
    """加载所有可用的结果文件"""
    results = {}
    
    # ESMFold 结果
    esmfold_file = get_abs_path(f"preds/iter{current_iter}/esmfold_summary.csv")
    if esmfold_file.exists():
        results['esmfold'] = pd.read_csv(esmfold_file)
        logger.info(f"加载 ESMFold 结果: {len(results['esmfold'])} 个预测")
    else:
        logger.warning("未找到 ESMFold 结果文件")
    
    # ColabFold 结果
    colabfold_file = get_abs_path(f"preds/iter{current_iter}/colabfold_summary.csv")
    if colabfold_file.exists():
        results['colabfold'] = pd.read_csv(colabfold_file)
        logger.info(f"加载 ColabFold 结果: {len(results['colabfold'])} 个预测")
    else:
        logger.info("未找到 ColabFold 结果文件（可选）")
    
    # Rosetta 评分结果
    rosetta_files = list(get_abs_path("scores").glob(f"*iter{current_iter}*.csv"))
    if rosetta_files:
        # 合并所有 Rosetta 结果
        rosetta_dfs = []
        for file in rosetta_files:
            try:
                df = pd.read_csv(file)
                rosetta_dfs.append(df)
                logger.info(f"加载 Rosetta 结果: {file.name}")
            except Exception as e:
                logger.warning(f"读取 Rosetta 文件失败 {file}: {e}")
        
        if rosetta_dfs:
            results['rosetta'] = pd.concat(rosetta_dfs, ignore_index=True)
            logger.info(f"合并 Rosetta 结果: {len(results['rosetta'])} 个评分")
    else:
        logger.info("未找到 Rosetta 评分文件（可选）")
    
    return results

def calculate_combined_scores(results: Dict[str, pd.DataFrame], 
                             weights: Dict[str, float], logger) -> pd.DataFrame:
    """计算综合评分"""
    logger.info("计算综合评分")
    
    # 以 ESMFold 结果为基础
    if 'esmfold' not in results:
        logger.error("缺少 ESMFold 基础结果")
        return pd.DataFrame()
    
    combined_df = results['esmfold'].copy()
    combined_df['combined_score'] = 0.0
    combined_df['score_components'] = ""
    
    # 标准化分数到 0-1 范围
    def normalize_score(series, reverse=False):
        """标准化分数，reverse=True 表示越小越好的分数"""
        if len(series) == 0 or series.isna().all():
            return series
        
        min_val, max_val = series.min(), series.max()
        if max_val == min_val:
            return pd.Series([0.5] * len(series), index=series.index)
        
        normalized = (series - min_val) / (max_val - min_val)
        return (1 - normalized) if reverse else normalized
    
    # ESMFold pLDDT 权重
    if weights.get('plddt', 0) > 0:
        plddt_norm = normalize_score(combined_df['mean_plddt'])
        combined_df['combined_score'] += weights['plddt'] * plddt_norm
        combined_df['score_components'] += f"pLDDT({weights['plddt']:.2f})"
    
    # ColabFold 置信度权重
    if 'colabfold' in results and weights.get('colabfold_confidence', 0) > 0:
        # 合并 ColabFold 结果
        colabfold_scores = results['colabfold'].set_index('sequence_id')['colabfold_confidence']
        combined_df['colabfold_confidence'] = combined_df['sequence_id'].map(colabfold_scores).fillna(0)
        
        conf_norm = normalize_score(combined_df['colabfold_confidence'])
        combined_df['combined_score'] += weights['colabfold_confidence'] * conf_norm
        combined_df['score_components'] += f" ColabFold({weights['colabfold_confidence']:.2f})"
    
    # Rosetta 能量权重
    if 'rosetta' in results and weights.get('rosetta_score', 0) > 0:
        # 合并 Rosetta 结果（假设越负越好）
        rosetta_scores = results['rosetta'].set_index('sequence_id')['total_score']
        combined_df['rosetta_score'] = combined_df['sequence_id'].map(rosetta_scores).fillna(0)
        
        # Rosetta 分数越负越好，所以 reverse=True
        rosetta_norm = normalize_score(combined_df['rosetta_score'], reverse=True)
        combined_df['combined_score'] += weights['rosetta_score'] * rosetta_norm
        combined_df['score_components'] += f" Rosetta({weights['rosetta_score']:.2f})"
    
    # 按综合分数排序
    combined_df = combined_df.sort_values('combined_score', ascending=False)
    
    logger.info(f"综合评分计算完成，最高分: {combined_df['combined_score'].max():.3f}")
    
    return combined_df

def select_top_candidates(scored_df: pd.DataFrame, top_n: int, 
                         min_plddt: float, logger) -> pd.DataFrame:
    """选择最佳候选"""
    logger.info(f"选择前 {top_n} 个候选，pLDDT 阈值: {min_plddt}")
    
    # 应用质量过滤
    filtered_df = scored_df[
        (scored_df['prediction_success'] == True) & 
        (scored_df['mean_plddt'] >= min_plddt)
    ].copy()
    
    if len(filtered_df) == 0:
        logger.warning(f"没有候选通过质量过滤（pLDDT >= {min_plddt}）")
        return pd.DataFrame()
    
    # 选择前 N 个
    top_candidates = filtered_df.head(top_n)
    
    logger.info(f"选择了 {len(top_candidates)} 个最佳候选")
    
    return top_candidates

def copy_top_structures(top_candidates: pd.DataFrame, current_iter: int, 
                       output_dir: Path, logger) -> List[str]:
    """复制最佳候选的结构文件"""
    logger.info("复制最佳候选结构文件")
    
    copied_files = []
    
    for idx, candidate in top_candidates.iterrows():
        seq_id = candidate['sequence_id']
        
        # 查找对应的 PDB 文件
        pdb_sources = [
            get_abs_path(f"preds/iter{current_iter}/{candidate.get('pdb_file', '')}"),
            get_abs_path(f"preds/iter{current_iter}/colabfold/{seq_id}_relaxed_rank_001_*.pdb"),
            get_abs_path(f"preds/iter{current_iter}/{seq_id}_esmfold.pdb")
        ]
        
        source_file = None
        for src in pdb_sources:
            if isinstance(src, str) and Path(src).exists():
                source_file = Path(src)
                break
            elif hasattr(src, 'glob'):
                matches = list(src.parent.glob(src.name))
                if matches:
                    source_file = matches[0]
                    break
        
        if source_file and source_file.exists():
            dest_file = output_dir / f"rank_{idx+1:02d}_{seq_id}.pdb"
            shutil.copy2(source_file, dest_file)
            copied_files.append(str(dest_file))
            logger.info(f"复制结构文件: {source_file.name} -> {dest_file.name}")
        else:
            logger.warning(f"未找到序列 {seq_id} 的结构文件")
    
    return copied_files

def save_iteration_history(config: Dict, current_iter: int, 
                          top_candidates: pd.DataFrame, action: str):
    """保存迭代历史"""
    history_file = get_abs_path("scores/iter_history.csv")
    
    # 准备历史记录
    history_record = {
        "iteration": current_iter,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_candidates": len(top_candidates),
        "best_plddt": top_candidates['mean_plddt'].max() if len(top_candidates) > 0 else 0,
        "best_combined_score": top_candidates['combined_score'].max() if len(top_candidates) > 0 else 0,
        "action_taken": action,
        "parameters": json.dumps(config.get("parameters", {}))
    }
    
    # 读取现有历史或创建新的
    if history_file.exists():
        history_df = pd.read_csv(history_file)
        history_df = pd.concat([history_df, pd.DataFrame([history_record])], ignore_index=True)
    else:
        history_df = pd.DataFrame([history_record])
    
    # 保存历史
    history_df.to_csv(history_file, index=False)

def display_results_summary(top_candidates: pd.DataFrame, logger):
    """显示结果摘要"""
    if len(top_candidates) == 0:
        print("❌ 没有找到合格的候选")
        return
    
    print("\n" + "="*60)
    print("🏆 最佳候选摘要")
    print("="*60)
    
    for idx, candidate in top_candidates.head(10).iterrows():
        rank = idx + 1
        seq_id = candidate['sequence_id']
        plddt = candidate['mean_plddt']
        combined_score = candidate.get('combined_score', 0)
        backbone_id = candidate.get('backbone_id', 'Unknown')
        
        print(f"排名 {rank:2d}: {seq_id}")
        print(f"         pLDDT: {plddt:.2f} | 综合分数: {combined_score:.3f}")
        print(f"         骨架: {backbone_id}")
        print(f"         序列: {candidate['sequence'][:50]}...")
        print()
    
    print("="*60)

def get_user_action_choice(top_candidates: pd.DataFrame, current_iter: int, 
                          max_iter: int, logger) -> str:
    """获取用户的行动选择"""
    print(f"\n当前迭代: {current_iter}/{max_iter}")
    print(f"候选数量: {len(top_candidates)}")
    
    if len(top_candidates) > 0:
        best_plddt = top_candidates['mean_plddt'].max()
        print(f"最佳 pLDDT: {best_plddt:.2f}")
        
        action_choices = ["accept", "analyze", "retry_step2", "retry_step3", "adjust_params"]
        
        if current_iter >= max_iter:
            print("已达到最大迭代次数")
            action_choices = ["accept", "analyze"]
        
    else:
        print("没有合格候选，建议重新设计")
        action_choices = ["retry_step2", "retry_step3", "adjust_params", "abort"]
    
    print("\n可选操作:")
    print("- accept: 接受当前结果，结束设计流程")
    print("- analyze: 详细分析当前结果，不采取行动")
    print("- retry_step2: 回退到骨架生成，尝试新参数")
    print("- retry_step3: 回退到序列设计，调整设计策略")
    print("- adjust_params: 调整筛选参数，重新评估")
    print("- abort: 终止设计流程")
    
    action = validate_input(
        "请选择下一步操作",
        str,
        valid_choices=action_choices,
        default_value="accept" if len(top_candidates) > 0 else "retry_step2"
    )
    
    return action

def main():
    """主函数"""
    print("=" * 60)
    print("Step 8: 迭代筛选与决策")
    print("作者: Kuroneko | 日期: 2025.9.30")
    print("=" * 60)
    
    logger = setup_logger("step8")
    start_time = datetime.datetime.now()
    
    try:
        # 加载配置
        config = load_config()
        current_iter = config.get("current_iteration", 1)
        max_iter = config.get("max_iterations", 10)
        
        logger.info(f"当前迭代: {current_iter}/{max_iter}")
        
        # 加载所有结果
        print("加载预测结果...")
        results = load_all_results(current_iter, logger)
        
        if not results:
            print("错误: 未找到任何预测结果")
            print("请先运行前面的步骤生成预测结果")
            return False
        
        # 获取用户参数
        print("\n请输入筛选参数:")
        
        selection_criteria = validate_input(
            "评分方式",
            str,
            valid_choices=["plddt", "combined", "manual"],
            default_value="combined"
        )
        
        top_n_candidates = validate_input(
            "选择最佳候选数量",
            int,
            valid_range=(1, 20),
            default_value=5
        )
        
        min_plddt_threshold = validate_input(
            "最低 pLDDT 阈值",
            float,
            valid_range=(50.0, 90.0),
            default_value=70.0
        )
        
        # 设置评分权重
        if selection_criteria == "combined":
            weights = config.get("scoring_weights", {
                "plddt": 0.4,
                "colabfold_confidence": 0.3,
                "rosetta_score": 0.2,
                "md_stability": 0.1
            })
        elif selection_criteria == "plddt":
            weights = {"plddt": 1.0}
        else:  # manual
            print("手动设置权重:")
            weights = {}
            weights["plddt"] = validate_input("pLDDT 权重", float, valid_range=(0.0, 1.0), default_value=0.5)
            if 'colabfold' in results:
                weights["colabfold_confidence"] = validate_input("ColabFold 权重", float, valid_range=(0.0, 1.0), default_value=0.3)
            if 'rosetta' in results:
                weights["rosetta_score"] = validate_input("Rosetta 权重", float, valid_range=(0.0, 1.0), default_value=0.2)
        
        # 计算综合评分
        print("\n计算综合评分...")
        scored_df = calculate_combined_scores(results, weights, logger)
        
        if scored_df.empty:
            print("错误: 无法计算评分")
            return False
        
        # 选择最佳候选
        top_candidates = select_top_candidates(scored_df, top_n_candidates, 
                                             min_plddt_threshold, logger)
        
        # 显示结果摘要
        display_results_summary(top_candidates, logger)
        
        # 设置输出目录
        scores_dir = ensure_dir("scores")
        top_candidates_dir = ensure_dir("scores/top_candidates")
        
        # 保存最佳候选列表
        if not top_candidates.empty:
            summary_file = scores_dir / "summary_top_candidates.csv"
            top_candidates.to_csv(summary_file, index=False)
            logger.info(f"最佳候选摘要已保存: {summary_file}")
            
            # 复制结构文件
            copied_files = copy_top_structures(top_candidates, current_iter, 
                                             top_candidates_dir, logger)
            print(f"✓ 复制了 {len(copied_files)} 个结构文件到 scores/top_candidates/")
        
        # 获取用户决策
        action = get_user_action_choice(top_candidates, current_iter, max_iter, logger)
        
        # 保存迭代历史
        save_iteration_history(config, current_iter, top_candidates, action)
        
        # 执行用户选择的操作
        if action == "accept":
            print("\n✅ 用户接受当前结果，设计流程完成")
            config["design_status"] = "completed"
            config["final_iteration"] = current_iter
            
        elif action == "retry_step2":
            print("\n🔄 准备回退到 Step 2 重新生成骨架")
            config["current_iteration"] = current_iter + 1
            print("请调整参数后运行: python scripts/step2_rfdiffusion_backbone.py")
            
        elif action == "retry_step3":
            print("\n🔄 准备回退到 Step 3 重新设计序列")
            print("请调整参数后运行: python scripts/step3_proteinmpnn_design.py")
            
        elif action == "adjust_params":
            print("\n⚙️ 请调整筛选参数后重新运行本脚本")
            
        elif action == "analyze":
            print("\n📊 结果分析完成，请查看输出文件进行详细分析")
            
        elif action == "abort":
            print("\n❌ 用户终止设计流程")
            config["design_status"] = "aborted"
        
        # 更新配置
        config["parameters"].update({
            "selection_criteria": selection_criteria,
            "top_n_candidates": top_n_candidates,
            "min_plddt_threshold": min_plddt_threshold,
            "scoring_weights": weights
        })
        
        # 记录迭代历史
        iteration_record = {
            "iteration": current_iter,
            "step": "step8_iterate_select",
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "parameters": {
                "selection_criteria": selection_criteria,
                "top_n_candidates": top_n_candidates,
                "min_plddt_threshold": min_plddt_threshold
            },
            "results": {
                "total_evaluated": len(scored_df),
                "candidates_selected": len(top_candidates),
                "action_taken": action,
                "execution_time": (datetime.datetime.now() - start_time).total_seconds()
            }
        }
        config["iteration_history"].append(iteration_record)
        save_config(config)
        
        # 总结
        print("\n" + "=" * 60)
        print("Step 8 执行完成!")
        print(f"评估了 {len(scored_df)} 个候选")
        print(f"选择了 {len(top_candidates)} 个最佳候选")
        print(f"用户选择: {action}")
        print(f"执行时间: {format_time_elapsed(start_time)}")
        
        if action == "accept":
            print("\n🎉 恭喜！蛋白质设计流程成功完成！")
            print("📁 最终结果位于: scores/top_candidates/")
            print("📊 详细摘要: scores/summary_top_candidates.csv")
        elif action in ["retry_step2", "retry_step3"]:
            print(f"\n🔄 准备进入下一轮迭代 (iter {config['current_iteration']})")
        
        print("=" * 60)
        
        return len(top_candidates) > 0
        
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

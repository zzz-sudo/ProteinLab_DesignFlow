#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step6_pipeline_driver.py - 完整分析流程驱动脚本
作者: Kuroneko
日期: 2025.9.16

功能说明:
    按顺序执行完整的生物序列和结构分析流程：
    1. 环境设置与数据准备（step1）
    2. 序列质量检查与SEG掩蔽（step2）
    3. 远程BLAST搜索（step3，可选）
    4. GFF/GenBank注释解析（step4，可选）
    5. PDB结构处理（step5，可选）
    6. 生成最终报告

输入输出:
    输入: 用户交互选择要运行的步骤
    输出: 所有子步骤的输出文件 + 最终综合报告

运行示例:
    python step6_pipeline_driver.py

设计决策:
    - 作为流程编排器，不重复实现功能
    - 通过导入其他步骤的main函数或使用subprocess调用
    - 提供灵活的步骤选择（允许跳过某些步骤）
    - 生成综合报告汇总所有结果
"""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

# 导入工具模块
try:
    from step7_utils import (
        setup_logger, get_user_input, confirm_action,
        print_section_header, OUTPUT_DIR, PROJECT_ROOT
    )
except ImportError:
    print("错误: 无法导入 step7_utils 模块")
    print("请确保 step7_utils.py 与本脚本在同一目录下")
    sys.exit(1)


# 定义流程步骤
PIPELINE_STEPS = [
    {
        'id': 'step1',
        'name': '环境设置与数据准备',
        'script': 'step1_setup_and_download.py',
        'required': True,
        'description': '创建目录结构，下载或生成示例数据'
    },
    {
        'id': 'step2',
        'name': '序列质量检查与SEG掩蔽',
        'script': 'step2_seq_check_and_seg.py',
        'required': True,
        'description': '检查序列有效性，执行低复杂度掩蔽'
    },
    {
        'id': 'step3',
        'name': '远程BLAST搜索',
        'script': 'step3_blast_remote_and_parse.py',
        'required': False,
        'description': '同源序列搜索（需要网络，耗时较长）'
    },
    {
        'id': 'step4',
        'name': 'GFF/GenBank注释解析',
        'script': 'step4_gff_genbank_parse.py',
        'required': False,
        'description': '解析基因注释文件（需要GFF或GenBank文件）'
    },
    {
        'id': 'step5',
        'name': 'PDB结构处理',
        'script': 'step5_pdb_processing.py',
        'required': False,
        'description': '解析蛋白结构文件（需要PDB文件）'
    }
]


def run_step_script(script_name: str, scripts_dir: Path, logger) -> bool:
    """
    运行步骤脚本
    
    参数:
        script_name: 脚本文件名
        scripts_dir: 脚本目录
        logger: 日志记录器
    
    返回:
        True 表示成功，False 表示失败
    """
    script_path = scripts_dir / script_name
    
    if not script_path.exists():
        logger.error(f"脚本不存在: {script_path}")
        return False
    
    logger.info(f"运行脚本: {script_name}")
    
    try:
        # 使用 subprocess 运行脚本
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(PROJECT_ROOT),
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            logger.info(f"  {script_name} 执行成功")
            return True
        else:
            logger.error(f"  {script_name} 执行失败，返回码: {result.returncode}")
            return False
            
    except Exception as e:
        logger.error(f"  运行脚本时出错: {e}")
        return False


def generate_final_report(results: dict, output_file: Path, logger):
    """
    生成最终综合报告
    
    参数:
        results: 各步骤执行结果
        output_file: 输出文件
        logger: 日志记录器
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("生物序列与结构分析 - 综合报告\n")
        f.write("="*70 + "\n")
        f.write(f"作者: Kuroneko\n")
        f.write(f"日期: 2025.10.04\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*70 + "\n\n")
        
        f.write("项目概述\n")
        f.write("-"*70 + "\n")
        f.write("本项目提供本地生物序列和结构的综合分析功能，包括：\n")
        f.write("  - 序列质量检查与低复杂度掩蔽\n")
        f.write("  - 远程 BLAST 同源搜索\n")
        f.write("  - GFF/GenBank 注释解析\n")
        f.write("  - PDB 蛋白结构处理\n\n")
        
        f.write("执行步骤与结果\n")
        f.write("-"*70 + "\n")
        
        for step_id, step_result in results.items():
            status = "成功" if step_result['success'] else "失败/跳过"
            f.write(f"\n[{step_id}] {step_result['name']}\n")
            f.write(f"  状态: {status}\n")
            if step_result.get('description'):
                f.write(f"  说明: {step_result['description']}\n")
        
        f.write("\n\n生成文件\n")
        f.write("-"*70 + "\n")
        f.write(f"输出目录: {OUTPUT_DIR}\n\n")
        
        # 列出所有输出文件
        if OUTPUT_DIR.exists():
            output_files = sorted(OUTPUT_DIR.glob("*"))
            if output_files:
                f.write("文件列表:\n")
                for file in output_files:
                    if file.is_file():
                        size_kb = file.stat().st_size / 1024
                        f.write(f"  - {file.name} ({size_kb:.1f} KB)\n")
            else:
                f.write("  （无输出文件）\n")
        
        f.write("\n\n后续分析建议\n")
        f.write("-"*70 + "\n")
        f.write("1. 查看 seq_check.csv 了解序列质量问题\n")
        f.write("2. 如果执行了 BLAST，查看 blast_results.csv 了解同源序列\n")
        f.write("3. 如果有注释文件，查看 gff_parsed.csv 或 gbk_parsed.csv\n")
        f.write("4. 如果有结构文件，查看 pdb_summary.csv 了解结构信息\n")
        f.write("5. 所有详细日志保存在 logs/ 目录\n\n")
        
        f.write("="*70 + "\n")
        f.write("分析完成\n")
        f.write("="*70 + "\n")
    
    logger.info(f"综合报告已保存: {output_file}")


def main():
    """
    主函数
    """
    print_section_header("步骤 6: 完整分析流程驱动")
    print("作者: Kuroneko")
    print("日期: 2025.10.04\n")
    
    # 设置日志
    logger = setup_logger('step6_pipeline')
    
    # 显示流程步骤
    print_section_header("流程步骤")
    
    print("\n本流程包含以下步骤:\n")
    for i, step in enumerate(PIPELINE_STEPS, 1):
        required_tag = "[必需]" if step['required'] else "[可选]"
        print(f"{i}. {required_tag} {step['name']}")
        print(f"   {step['description']}\n")
    
    # 询问用户执行模式
    print("执行模式:")
    print("  1. 完整执行（运行所有步骤）")
    print("  2. 选择性执行（跳过某些可选步骤）")
    print("  3. 仅必需步骤（跳过所有可选步骤）")
    
    mode = get_user_input("\n请选择执行模式", default="1", valid_options=['1', '2', '3'])
    
    # 确定要执行的步骤
    steps_to_run = []
    
    if mode == '1':
        steps_to_run = PIPELINE_STEPS
        print("\n将执行所有步骤")
    elif mode == '3':
        steps_to_run = [s for s in PIPELINE_STEPS if s['required']]
        print(f"\n将执行 {len(steps_to_run)} 个必需步骤")
    else:  # mode == '2'
        print("\n请选择要执行的步骤:")
        for step in PIPELINE_STEPS:
            if step['required']:
                print(f"  {step['name']}: 必需")
                steps_to_run.append(step)
            else:
                if confirm_action(f"  执行 {step['name']}?", default_yes=False):
                    steps_to_run.append(step)
    
    if not steps_to_run:
        print("\n没有选择任何步骤，退出")
        sys.exit(0)
    
    # 执行流程
    print_section_header(f"开始执行流程（共 {len(steps_to_run)} 个步骤）")
    
    scripts_dir = Path(__file__).parent
    results = {}
    
    for i, step in enumerate(steps_to_run, 1):
        print(f"\n{'='*70}")
        print(f"执行步骤 {i}/{len(steps_to_run)}: {step['name']}")
        print(f"{'='*70}\n")
        
        success = run_step_script(step['script'], scripts_dir, logger)
        
        results[step['id']] = {
            'name': step['name'],
            'success': success,
            'description': step['description']
        }
        
        if not success and step['required']:
            logger.error(f"必需步骤 {step['name']} 失败，终止流程")
            print(f"\n错误: 必需步骤失败，流程终止")
            break
        
        if i < len(steps_to_run):
            print(f"\n{'='*70}")
            print(f"步骤 {i} 完成，准备下一步...")
            print(f"{'='*70}\n")
    
    # 生成综合报告
    print_section_header("生成综合报告")
    
    report_file = OUTPUT_DIR / f"final_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    generate_final_report(results, report_file, logger)
    
    print(f"\n综合报告已保存: {report_file}")
    
    # 完成
    print_section_header("流程执行完成")
    
    success_count = sum(1 for r in results.values() if r['success'])
    total_count = len(results)
    
    print(f"\n执行统计:")
    print(f"  成功: {success_count}/{total_count}")
    print(f"  失败/跳过: {total_count - success_count}/{total_count}")
    
    print(f"\n所有结果保存在: {OUTPUT_DIR}")
    print(f"所有日志保存在: {PROJECT_ROOT / 'logs'}")
    
    print("\n感谢使用本项目！")
    print("作者: Kuroneko")
    print("日期: 2025.10.04")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断操作")
        sys.exit(0)
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


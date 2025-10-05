#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
作者: Kuroneko
日期: 2025-10-04
功能: 批量处理本地 PDB 文件，提取 protein.pdb 和 ligand.pdb
说明:
    - 输入本地 PDB 文件夹
    - 每个 PDB 文件生成独立子文件夹
    - 输出 protein.pdb 和 ligand.pdb，可直接作为 LigandMPNN 输入
"""

import os
from Bio.PDB import PDBParser, PDBIO, Select

# -------------------------------
# Step 1: 配置
# -------------------------------
# 输入文件夹（本地已有 PDB 文件夹）
input_pdb_dir = "local_pdb_folder"  # 请改成你的本地 PDB 文件夹路径
# 输出文件夹
output_dir = "pdb_batch_output"
os.makedirs(output_dir, exist_ok=True)

# -------------------------------
# Step 2: 定义选择器
# -------------------------------
class ProteinSelect(Select):
    def accept_residue(self, residue):
        # 保留标准氨基酸
        return residue.id[0] == " "

class LigandSelect(Select):
    def accept_residue(self, residue):
        # 非氨基酸小分子
        return residue.id[0] != " "

# -------------------------------
# Step 3: 批量处理每个 PDB 文件
# -------------------------------
parser = PDBParser(QUIET=True)
io = PDBIO()

# 列出所有 PDB 文件
pdb_files = [f for f in os.listdir(input_pdb_dir) if f.lower().endswith(".pdb")]

if not pdb_files:
    print("[WARN] 输入文件夹没有 PDB 文件，请检查路径。")
else:
    for pdb_file in pdb_files:
        pdb_id = os.path.splitext(pdb_file)[0]
        pdb_path = os.path.join(input_pdb_dir, pdb_file)
        print(f"[INFO] 处理 {pdb_file} ...")

        # 创建子文件夹
        sub_dir = os.path.join(output_dir, pdb_id)
        os.makedirs(sub_dir, exist_ok=True)

        # 解析结构
        structure = parser.get_structure(pdb_id, pdb_path)

        # 输出 protein.pdb
        protein_file = os.path.join(sub_dir, f"{pdb_id}_protein.pdb")
        io.set_structure(structure)
        io.save(protein_file, ProteinSelect())
        print(f"[INFO] 蛋白质文件保存: {protein_file}")

        # 输出 ligand.pdb
        ligand_file = os.path.join(sub_dir, f"{pdb_id}_ligand.pdb")
        io.set_structure(structure)
        io.save(ligand_file, LigandSelect())
        print(f"[INFO] 配体文件保存: {ligand_file}")

    print("[INFO] 批量提取完成，可用于 LigandMPNN 输入。")

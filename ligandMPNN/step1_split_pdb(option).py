#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
作者: Kuroneko
日期: 2025-10-04
功能: 从蛋白-配体复合物 PDB 自动提取蛋白和配体
说明:
    - 下载示例 PDB (1BRS)
    - 使用 Biopython 分离蛋白质和小分子配体
    - 输出 protein.pdb 和 ligand.pdb 到本地
"""

import os
from Bio.PDB import PDBList, PDBParser, PDBIO, Select

# -------------------------------
# Step 1: 创建项目目录
# -------------------------------
project_dir = "./pdb_project"
os.makedirs(project_dir, exist_ok=True)

# -------------------------------
# Step 2: 下载 PDB 文件
# -------------------------------
pdb_id = "1BRS"  # 示例公开复合物
pdbl = PDBList()
pdb_file = pdbl.retrieve_pdb_file(pdb_id, pdir=project_dir, file_format="pdb")
print(f"[INFO] PDB 文件下载完成: {pdb_file}")

# -------------------------------
# Step 3: 解析 PDB
# -------------------------------
parser = PDBParser(QUIET=True)
structure = parser.get_structure(pdb_id, pdb_file)

# -------------------------------
# Step 4: 定义选择器
# -------------------------------
class ProteinSelect(Select):
    def accept_residue(self, residue):
        # 保留标准氨基酸
        if residue.id[0] == " ":
            return True
        return False

class LigandSelect(Select):
    def accept_residue(self, residue):
        # 非氨基酸小分子
        if residue.id[0] != " ":
            return True
        return False

# -------------------------------
# Step 5: 输出 protein.pdb
# -------------------------------
io = PDBIO()
protein_file = os.path.join(project_dir, f"{pdb_id}_protein.pdb")
io.set_structure(structure)
io.save(protein_file, ProteinSelect())
print(f"蛋白质文件保存: {protein_file}")

# -------------------------------
# Step 6: 输出 ligand.pdb
# -------------------------------
ligand_file = os.path.join(project_dir, f"{pdb_id}_ligand.pdb")
io.set_structure(structure)
io.save(ligand_file, LigandSelect())
print(f"配体文件保存: {ligand_file}")

# -------------------------------
# 完成
# -------------------------------
print("提取完成，可用于 LigandMPNN 输入。")

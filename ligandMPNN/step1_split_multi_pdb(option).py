#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
作者: Kuroneko
日期: 2025-10-04
功能: 批量下载复合物 PDB，并提取 protein.pdb 和 ligand.pdb
说明:
    - 支持多个 PDB ID 批量处理
    - 每个 PDB 文件生成单独子文件夹
    - 输出 protein.pdb 和 ligand.pdb，可直接作为 LigandMPNN 输入
"""

import os
from Bio.PDB import PDBList, PDBParser, PDBIO, Select

# -------------------------------
# Step 1: 配置
# -------------------------------
pdb_ids = ["1BRS", "1HVR", "2VB1"]  # 你可以在这里扩展更多 PDB
project_dir = "./pdb_batch_project"
os.makedirs(project_dir, exist_ok=True)

# -------------------------------
# Step 2: 下载 PDB 文件
# -------------------------------
pdbl = PDBList()
downloaded_files = []

for pdb_id in pdb_ids:
    print(f"[INFO] 下载 PDB {pdb_id} ...")
    pdb_file = pdbl.retrieve_pdb_file(pdb_id, pdir=project_dir, file_format="pdb")
    downloaded_files.append((pdb_id, pdb_file))

# -------------------------------
# Step 3: 定义选择器
# -------------------------------
class ProteinSelect(Select):
    def accept_residue(self, residue):
        if residue.id[0] == " ":
            return True
        return False

class LigandSelect(Select):
    def accept_residue(self, residue):
        if residue.id[0] != " ":
            return True
        return False

# -------------------------------
# Step 4: 批量处理每个 PDB
# -------------------------------
parser = PDBParser(QUIET=True)
io = PDBIO()

for pdb_id, pdb_file in downloaded_files:
    print(f"[INFO] 处理 {pdb_id} ...")
    # 创建子文件夹
    sub_dir = os.path.join(project_dir, pdb_id)
    os.makedirs(sub_dir, exist_ok=True)

    # 解析结构
    structure = parser.get_structure(pdb_id, pdb_file)

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

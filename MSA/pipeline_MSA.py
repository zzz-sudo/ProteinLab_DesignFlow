# ===============================================
# Author: Kuroneko
# Date: 2025.10.6
# Description:
# 自动处理多链 PDB 文件，将 query MSA 序列比对到每条链，
# 并生成每条链对应的 fixed_positions_<chain>.json，
# 可直接用于 ProteinMPNN 或 LigandMPNN 冻结保守位点。
# 同时生成 WebLogo 可视化图和保守性 CSV。
# 输入文件: step1_input.a3m, pdb_file.pdb
# 输出文件: cleaned.aln, conservation_scores.csv, fixed_positions_<chain>.json, weblogo.png
# ===============================================

import os
import json
import math
from Bio import SeqIO, AlignIO
from Bio.Align import PairwiseAligner
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.PDB import PDBParser, PPBuilder
import matplotlib.pyplot as plt

# -----------------------------
# Step1: 清理 A3M 文件
# -----------------------------
def step1_clean_a3m(input_a3m, output_aln):
    records = []
    with open(input_a3m) as f:
        seq_id, seq = None, ''
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if seq_id:
                    records.append(SeqRecord(Seq(seq), id=seq_id, description=''))
                seq_id = line[1:].split()[0]
                seq = ''
            else:
                clean_seq = ''.join([c for c in line if c.isupper() or c == '-'])
                seq += clean_seq
        if seq_id:
            records.append(SeqRecord(Seq(seq), id=seq_id, description=''))
    SeqIO.write(records, output_aln, "fasta")
    print(f"[Step1] Cleaned MSA saved to {output_aln}")
    return output_aln

# -----------------------------
# Step2: 计算保守性
# -----------------------------
def step2_compute_conservation(aln_file, output_csv):
    alignment = AlignIO.read(aln_file, "fasta")
    n_positions = alignment.get_alignment_length()
    
    def shannon_entropy(column):
        freq = {}
        for aa in column:
            if aa == '-':
                continue
            freq[aa] = freq.get(aa, 0) + 1
        total = sum(freq.values())
        entropy = 0.0
        for aa_count in freq.values():
            p = aa_count / total
            entropy -= p * math.log2(p)
        return entropy

    entropies = [shannon_entropy(alignment[:,i]) for i in range(n_positions)]
    
    with open(output_csv, 'w') as f:
        f.write("position,entropy\n")
        for idx, val in enumerate(entropies):
            f.write(f"{idx+1},{val:.4f}\n")
    print(f"[Step2] Conservation scores saved to {output_csv}")
    return entropies

# -----------------------------
# Step3: 提取 PDB 所有链序列
# -----------------------------
def step3_extract_all_chains(pdb_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('pdb', pdb_file)
    ppb = PPBuilder()
    chain_sequences = {}
    for model in structure:
        for chain in model:
            seq = ''
            for pp in ppb.build_peptides(chain):
                seq += str(pp.get_sequence())
            if seq:
                chain_sequences[chain.id] = seq
                print(f"[Step3] Extracted chain {chain.id} sequence: length {len(seq)}")
        break  # 只处理第一个 model
    return chain_sequences

# -----------------------------
# Step4: 映射高保守位点到每条 PDB 链
# -----------------------------
def step4_map_fixed_positions_multi_chain(query_seq, chain_sequences, entropies, threshold=1.0, output_dir="."):
    aligner = PairwiseAligner()
    aligner.mode = 'global'
    fixed_positions_files = {}
    for chain_id, pdb_seq in chain_sequences.items():
        alignment = aligner.align(query_seq, pdb_seq)[0]
        query_aligned = alignment.seqA
        pdb_aligned = alignment.seqB

        fixed_positions = []
        query_index = 0
        for i, (q_res, p_res) in enumerate(zip(query_aligned, pdb_aligned)):
            if q_res != '-':
                query_index += 1
            if p_res != '-' and q_res != '-':
                if entropies[query_index-1] < threshold:
                    fixed_positions.append(i+1)  # 1-based PDB numbering

        output_json = os.path.join(output_dir, f"fixed_positions_{chain_id}.json")
        with open(output_json, 'w') as f:
            json.dump({"fixed_positions": fixed_positions}, f, indent=4)
        fixed_positions_files[chain_id] = output_json
        print(f"[Step4] Chain {chain_id} fixed_positions saved to {output_json}, count={len(fixed_positions)}")
    return fixed_positions_files

# -----------------------------
# Step5: WebLogo 可视化
# -----------------------------
def step5_weblogo_visualization(aln_file, output_logo="weblogo.png"):
    alignment = AlignIO.read(aln_file, "fasta")
    n_positions = alignment.get_alignment_length()
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    freqs_per_position = []
    for i in range(n_positions):
        column = alignment[:, i]
        counts = {aa:0 for aa in amino_acids}
        total = 0
        for aa in column:
            if aa in amino_acids:
                counts[aa] += 1
                total += 1
        freqs = {aa: counts[aa]/total if total>0 else 0 for aa in amino_acids}
        freqs_per_position.append(freqs)
    
    fig, ax = plt.subplots(figsize=(max(10, n_positions/2),6))
    bottom = [0]*n_positions
    colors = {
        'A':'green','C':'yellow','D':'red','E':'red','F':'blue','G':'green',
        'H':'blue','I':'blue','K':'red','L':'blue','M':'blue','N':'red',
        'P':'green','Q':'red','R':'red','S':'green','T':'green','V':'blue',
        'W':'blue','Y':'blue'
    }
    for aa in amino_acids:
        heights = [freqs_per_position[i][aa] for i in range(n_positions)]
        ax.bar(range(1,n_positions+1), heights, bottom=bottom, color=colors.get(aa,'gray'), edgecolor='black', width=1.0, label=aa)
        bottom = [bottom[i]+heights[i] for i in range(n_positions)]
    ax.set_xlabel("Position")
    ax.set_ylabel("Frequency")
    ax.set_title("WebLogo-like Amino Acid Frequency")
    ax.set_xlim(0,n_positions+1)
    ax.set_ylim(0,1)
    ax.legend(ncol=5, bbox_to_anchor=(1.05,1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_logo, dpi=150)
    plt.close()
    print(f"[Step5] WebLogo saved to {output_logo}")

# -----------------------------
# 主流程
# -----------------------------
if __name__ == "__main__":
    input_a3m = "step1_input.a3m"
    cleaned_aln = "cleaned.aln"
    conservation_csv = "conservation_scores.csv"
    pdb_file = "pdb_file.pdb"
    output_dir = "."
    weblogo_png = "weblogo.png"
    threshold = 1.0

    # Step1
    step1_clean_a3m(input_a3m, cleaned_aln)
    # Step2
    entropies = step2_compute_conservation(cleaned_aln, conservation_csv)
    # Step3
    chain_sequences = step3_extract_all_chains(pdb_file)
    # Step4
    query_seq = str(SeqIO.read(cleaned_aln, "fasta").seq)  # 默认第一个序列为 query
    step4_map_fixed_positions_multi_chain(query_seq, chain_sequences, entropies, threshold, output_dir)
    # Step5
    step5_weblogo_visualization(cleaned_aln, weblogo_png)

    print("Pipeline 完成，fixed_positions_<chain>.json 可直接用于 ProteinMPNN 冻结保守位点。")

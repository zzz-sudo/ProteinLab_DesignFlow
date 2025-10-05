ip# 生物序列与结构综合分析工具

**作者**: Kuroneko  
**日期**: 2025.10.04

---

## 项目简介

本项目是一个多脚本 Python 工具集，用于本地生物序列和蛋白结构的综合分析。所有脚本均可独立运行，同时共享工具模块以复用代码。

### 主要功能

- **序列质量检查**: 检测非标准字符、序列长度、低复杂度区域
- **SEG 掩蔽**: 对蛋白序列进行低复杂度区域掩蔽
- **远程 BLAST 搜索**: 使用 NCBI BLAST Web 服务进行同源序列搜索
- **注释文件解析**: 解析 GFF3 和 GenBank 格式的基因注释
- **PDB 结构处理**: 解析 PDB 蛋白结构文件，提取结构信息
- **自动化流程**: 提供流程驱动脚本，一键执行完整分析

---

## 项目结构

```
project_root/
├── data/                           # 数据目录
│   ├── input_sequences.fasta       # 默认输入序列文件
│   ├── example_public_uniprot.fasta      # 公开 UniProt 数据（若下载）
│   ├── example_public_genbank.gb         # 公开 GenBank 数据（若下载）
│   ├── example_public_structure.pdb      # 公开 PDB 结构（若下载）
│   ├── example_synthetic.fasta           # 合成测试数据（若生成）
│   ├── download_manifest.txt             # 下载清单
│   └── synthetic_manifest.txt            # 合成数据清单
├── outputs/                        # 输出目录
│   ├── seq_check.csv               # 序列检查报告
│   ├── seg_masked.fasta            # SEG 掩蔽后序列
│   ├── blast_results.csv           # BLAST 结果摘要
│   ├── gff_parsed.csv              # GFF 解析结果
│   ├── gbk_parsed.csv              # GenBank 解析结果
│   ├── pdb_summary.csv             # PDB 结构摘要
│   └── final_report_*.txt          # 综合报告
├── logs/                           # 日志目录
├── scripts/                        # 脚本目录
│   ├── step1_setup_and_download.py       # 环境设置与数据准备
│   ├── step2_seq_check_and_seg.py        # 序列检查与SEG掩蔽
│   ├── step3_blast_remote_and_parse.py   # BLAST搜索与解析
│   ├── step4_gff_genbank_parse.py        # 注释文件解析
│   ├── step5_pdb_processing.py           # PDB结构处理
│   ├── step6_pipeline_driver.py          # 流程驱动脚本
│   └── step7_utils.py                    # 工具模块
├── requirements.txt                # Python 依赖
└── README.md                       # 项目文档
```

---

## 安装与环境配置

### 1. Python 环境要求

- Python 3.9 或更高版本
- 推荐使用虚拟环境（venv 或 conda）

### 2. 安装依赖

```bash
# 克隆或下载项目后，进入项目根目录
cd /path/to/project_root

# 安装依赖包
pip install -r requirements.txt
```

### 3. 验证安装

```bash
python scripts/step7_utils.py
```

如果看到工具模块的说明信息，表示安装成功。

---

## 使用方法

### 快速开始

#### 方式一：运行完整流程（推荐新手）

```bash
# 1. 首先运行 step1 准备数据和环境
python scripts/step1_setup_and_download.py

# 2. 然后运行 step6 执行完整流程
python scripts/step6_pipeline_driver.py
```

#### 方式二：逐步运行各个脚本

```bash
# 步骤 1: 环境设置与数据准备
python scripts/step1_setup_and_download.py

# 步骤 2: 序列质量检查与 SEG 掩蔽
python scripts/step2_seq_check_and_seg.py

# 步骤 3: BLAST 同源搜索（可选，需要网络）
python scripts/step3_blast_remote_and_parse.py

# 步骤 4: 注释文件解析（可选，需要 GFF/GenBank 文件）
python scripts/step4_gff_genbank_parse.py

# 步骤 5: PDB 结构处理（可选，需要 PDB 文件）
python scripts/step5_pdb_processing.py

# 步骤 6: 完整流程驱动
python scripts/step6_pipeline_driver.py
```

### 脚本说明

#### step1_setup_and_download.py - 环境设置与数据准备

**功能**:
- 创建项目所需的目录结构
- 提供两种数据获取方式：
  - **选项 A**: 下载公开真实数据（推荐）
  - **选项 B**: 生成合成测试数据

**交互选项**:
- 是否下载公开数据（Y/N）
- 如果选择下载，可以自定义 accession/ID，或使用默认值

**默认公开数据示例**:
- **UniProt**: P69905（人血红蛋白亚基 alpha）
  - 来源: https://rest.uniprot.org/uniprotkb/P69905.fasta
- **GenBank**: NM_000518.5（人 HBB 基因 mRNA）
  - 来源: https://www.ncbi.nlm.nih.gov/nuccore/NM_000518.5
- **PDB**: 1CRN（Crambin 蛋白，46残基）
  - 来源: https://files.rcsb.org/download/1CRN.pdb

**合成数据说明**:
- 如果选择生成合成数据，将创建 4 条测试序列：
  1. `SYNTH_SHORT_001`: 短肽链（25 aa）
  2. `SYNTH_MEDIUM_002`: 中等长度蛋白（150 aa）
  3. `SYNTH_LONG_003`: 长蛋白含重复片段（约1200 aa）
  4. `SYNTH_LOWCOMP_004`: 低复杂度序列（poly-A 和 PG-repeats）
- 随机种子: 42（确保可复现）
- 生成方法: 从标准20种氨基酸中随机选择

#### step2_seq_check_and_seg.py - 序列质量检查与 SEG 掩蔽

**功能**:
- 读取 FASTA 格式序列
- 检查序列有效性（非标准字符、长度、类型）
- 计算序列复杂度
- 执行 SEG 算法掩蔽低复杂度区域

**输出**:
- `outputs/seq_check.csv`: 序列检查报告
- `outputs/seg_masked.fasta`: SEG 掩蔽后的序列
- `outputs/sample_seg_examples.txt`: 掩蔽前后对比示例

#### step3_blast_remote_and_parse.py - BLAST 搜索与解析

**功能**:
- 使用 NCBI BLAST Web 服务进行远程同源搜索
- 解析 BLAST 结果并提取关键信息
- 生成 CSV 格式结果摘要

**重要提示**:
- ⚠️ **隐私警告**: 序列数据将上传到 NCBI 服务器
- ⚠️ **耗时提醒**: 每个序列可能需要数分钟
- 对于合成序列，脚本会额外提示确认

**输出**:
- `outputs/blast_results.csv`: BLAST 结果摘要
- `outputs/blast_details_*.txt`: 详细结果

#### step4_gff_genbank_parse.py - 注释文件解析

**功能**:
- 解析 GFF3 格式基因注释文件
- 解析 GenBank 格式记录文件
- 提取基因特征（gene, CDS, mRNA, exon等）

**输入要求**:
- 将 GFF/GenBank 文件放置在 `data/` 目录
- 支持文件扩展名: `.gff`, `.gff3`, `.gb`, `.gbk`

**输出**:
- `outputs/gff_parsed.csv`: GFF 解析结果
- `outputs/gbk_parsed.csv`: GenBank 解析结果

#### step5_pdb_processing.py - PDB 结构处理

**功能**:
- 解析 PDB 格式蛋白结构文件
- 提取结构统计信息（链、残基、原子数量）
- 计算 B 因子统计和几何中心

**输入要求**:
- 将 PDB 文件放置在 `data/` 目录
- 支持文件扩展名: `.pdb`

**输出**:
- `outputs/pdb_summary.csv`: 结构摘要
- `outputs/pdb_details_*.txt`: 详细信息

#### step6_pipeline_driver.py - 流程驱动脚本

**功能**:
- 按顺序执行完整分析流程
- 提供三种执行模式：
  1. 完整执行（所有步骤）
  2. 选择性执行（跳过某些可选步骤）
  3. 仅必需步骤（step1 + step2）

**输出**:
- `outputs/final_report_*.txt`: 综合报告

---

## 数据来源与示例

### 公开真实数据（推荐用于学习）

本项目支持从以下公共数据库下载真实生物数据：

| 数据类型 | 默认示例 | Accession/ID | 来源 URL |
|---------|---------|--------------|----------|
| **蛋白序列** | 人血红蛋白 alpha 链 | P69905 | https://rest.uniprot.org/uniprotkb/P69905.fasta |
| **基因序列** | 人 HBB 基因 mRNA | NM_000518.5 | https://www.ncbi.nlm.nih.gov/nuccore/NM_000518.5 |
| **蛋白结构** | Crambin 蛋白 | 1CRN | https://files.rcsb.org/download/1CRN.pdb |

**优点**:
- 真实的生物学数据，适合学习
- 来源可追溯，符合学术规范
- BLAST 搜索能找到真实同源序列

**注意事项**:
- 需要网络连接
- 下载时间取决于网络速度
- 所有下载记录保存在 `data/download_manifest.txt`

### 合成测试数据（用于快速测试）

如果不需要网络或只是测试软件功能，可以选择生成合成数据：

| 序列 ID | 类型 | 长度 | 用途 |
|---------|------|------|------|
| SYNTH_SHORT_001 | 短肽链 | 25 aa | 测试短序列处理 |
| SYNTH_MEDIUM_002 | 中等蛋白 | 150 aa | 标准长度测试 |
| SYNTH_LONG_003 | 长蛋白 | ~1200 aa | 长序列与重复片段测试 |
| SYNTH_LOWCOMP_004 | 低复杂度序列 | ~140 aa | SEG 掩蔽效果测试 |

**生成方法**:
- 使用 Python `random.choices()` 从20种标准氨基酸随机生成
- 随机种子固定为 42，确保可复现
- 详细记录保存在 `data/synthetic_manifest.txt`

**注意事项**:
- 合成序列是计算机生成的，非真实生物序列
- 仅用于软件功能测试
- BLAST 搜索不太可能找到同源序列

---

## 常见问题

### Q1: 如何切换数据源？

**A**: 重新运行 `step1_setup_and_download.py`，选择不同的数据获取方式。

### Q2: BLAST 搜索需要多长时间？

**A**: 每个序列通常需要 2-5 分钟，取决于：
- 序列长度
- NCBI 服务器负载
- 网络速度

为节省时间，脚本默认限制最多查询 3 条序列。

### Q3: 如何使用自己的数据？

**A**: 
1. 将 FASTA 格式序列文件放置为 `data/input_sequences.fasta`
2. 将 GFF/GenBank 文件放置在 `data/` 目录
3. 将 PDB 文件放置在 `data/` 目录
4. 运行对应的脚本

### Q4: 遇到 "需要安装 Biopython" 错误怎么办？

**A**: 运行安装命令：
```bash
pip install biopython
```

### Q5: 合成数据是否可以用于发表？

**A**: **不可以**。合成数据仅用于软件测试，不代表真实生物学信息，不应用于科研发表。

---

## 隐私与数据安全

### 使用远程 BLAST 的注意事项

- ⚠️ 您的序列将上传到 NCBI 服务器
- ⚠️ NCBI 可能保留查询序列用于统计
- ⚠️ 请勿上传敏感、专有或未发表的数据
- ⚠️ 建议使用公开序列进行测试
- ⚠️ 如需处理敏感数据，请使用本地 BLAST

脚本会在上传前显示隐私警告并要求确认。

---

## 技术细节

### SEG 算法实现

本项目实现了简化版 SEG 算法：
- 使用滑动窗口（默认 12 个残基）
- 检测窗口内单一氨基酸高频区域（阈值 50%）
- 将低复杂度位置标记为小写字母

**标准 SEG vs 简化版**:
- 标准 SEG: 基于序列熵的复杂算法
- 简化版: 基于单字符频率，纯 Python 实现
- 适用场景: 简化版适合教学和快速筛选

### 设计决策

1. **纯 Python 实现**: 无需编译外部程序
2. **模块化设计**: 每个脚本独立可运行
3. **交互式操作**: 所有参数通过 `input()` 获取
4. **错误处理**: 完善的异常捕获和用户提示
5. **日志记录**: 所有操作记录到日志文件

---

## 开发与贡献

### 代码规范

- 编码: UTF-8
- 缩进: 4 空格
- 文档字符串: Google 风格
- 注释: 中文

### 文件头部格式

每个脚本文件包含：
- 作者: Kuroneko
- 日期: 2025.10.04
- 功能说明
- 输入输出说明
- 运行示例

---

## 许可与引用

本项目用于教学和研究目的。使用时请注意：

1. **依赖工具的引用**:
   - Biopython: Cock et al. (2009) Bioinformatics
   - NCBI BLAST: Altschul et al. (1990) J Mol Biol

2. **公开数据的引用**:
   - UniProt: The UniProt Consortium
   - NCBI GenBank: NCBI Resource Coordinators
   - RCSB PDB: Berman et al. (2000) Nucleic Acids Res

---

## 更新日志

### 2025.10.04 - 初始版本
- 完成全部 7 个脚本（step1-step7）
- 支持公开数据下载与合成数据生成
- 实现序列检查、SEG 掩蔽、BLAST 搜索
- 实现 GFF/GenBank 解析和 PDB 处理
- 添加完整流程驱动脚本

---

## 联系方式

**作者**: Kuroneko  
**日期**: 2025.10.04

如有问题或建议，欢迎反馈。

---

**祝您使用愉快！**


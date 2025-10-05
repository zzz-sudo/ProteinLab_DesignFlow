# JSON 配置文件详细使用指南

**作者**: Kuroneko  
**日期**: 2025.9.30  

## 目录概述

本目录包含了项目的所有配置文件，这些JSON文件控制着整个蛋白质设计流程的各个方面。通过修改这些文件，您可以精确控制设计过程，实现个性化的蛋白质设计目标。

## 文件功能概览

```
json/
├── example_config.json              # 项目主配置模板
├── mini_protein_targets.json       # 预定义设计目标
├── design_constraints.json         # 设计约束模板库
└── custom_constraints_template.json # 自定义约束构建指南
```

---

## 1. example_config.json - 项目主配置

### 文件作用
这是整个项目的**中央配置文件**，控制所有设计步骤的默认参数。

### 详细参数说明

#### A. 基本信息区域
```json
{
  "project_name": "de_novo_protein_design",
  "author": "Kuroneko", 
  "version": "1.0.0",
  "created_date": "2025.9.30",
  "current_iteration": 1,
  "max_iterations": 10
}
```

**可调节参数**:
- `project_name`: 项目名称，会影响输出文件命名
- `current_iteration`: 当前迭代轮次，控制输出目录
- `max_iterations`: 最大迭代次数，防止无限循环

**小白指南**:
- 一般不需要修改基本信息
- `current_iteration`会自动更新，无需手动改

#### B. 核心设计参数区域
```json
"parameters": {
  "num_backbones": 200,                    // 可调节
  "num_sequences_per_backbone": 50,        // 可调节  
  "esmfold_pLDDT_threshold": 60.0,        // 可调节
  "colabfold_use_online_msa": "yes",      // 可调节
  "max_md_ns": 10,                        // 可调节
  "rosetta_relax_rounds": 2,              // 可调节
  "rfdiffusion_seed": 42,                 // 可调节
  "target_length": 100,                   // 可调节
  "sampling_temperature": 0.1             // 可调节
}
```

**详细调节指南**:

##### num_backbones (骨架生成数量)
```json
"num_backbones": 200
```
**作用**: 控制RFdiffusion生成多少个不同的蛋白质骨架
**推荐值**:
- **测试阶段**: 10-50（快速验证流程）
- **正常设计**: 100-300（平衡质量和效率）
- **大规模筛选**: 500-1000（追求最佳结果）

**影响因素**:
- 数量越多，找到优质骨架的概率越高
- 但计算时间和存储需求也相应增加
- GPU内存限制可能影响实际可生成数量

**调节示例**:
```json
// 快速测试
"num_backbones": 20

// 标准设计  
"num_backbones": 200

// 深度搜索
"num_backbones": 800
```

##### num_sequences_per_backbone (每个骨架的序列数)
```json
"num_sequences_per_backbone": 50
```
**作用**: 对每个骨架结构，ProteinMPNN生成多少个不同序列
**推荐值**:
- **快速测试**: 5-20
- **标准设计**: 30-100  
- **深度优化**: 100-500

**计算关系**:
```
总序列数 = num_backbones × num_sequences_per_backbone
例如: 200个骨架 × 50个序列 = 10,000个候选序列
```

**调节考虑**:
- 序列数量越多，找到优质序列的概率越高
- 但ESMFold预测时间会相应增加
- 建议根据计算资源调整

##### esmfold_pLDDT_threshold (质量阈值)
```json
"esmfold_pLDDT_threshold": 60.0
```
**作用**: ESMFold预测的置信度阈值，过滤低质量预测
**阈值含义**:
- **90+**: 接近实验结构质量（极高要求）
- **80-90**: 高质量，适合功能研究
- **70-80**: 良好质量，适合大多数应用
- **60-70**: 可接受质量，适合初步筛选
- **50-60**: 较低质量，仅用于探索性研究

**调节指南**:
```json
// 小蛋白(<50残基) - 要求高质量
"esmfold_pLDDT_threshold": 75.0

// 功能蛋白 - 平衡质量和数量
"esmfold_pLDDT_threshold": 70.0

// 大蛋白(>150残基) - 适当降低要求
"esmfold_pLDDT_threshold": 60.0

// 探索性设计 - 宽松过滤
"esmfold_pLDDT_threshold": 50.0
```

##### sampling_temperature (采样温度)
```json
"sampling_temperature": 0.1
```
**作用**: 控制ProteinMPNN序列生成的随机性
**温度效果**:
- **0.01-0.1**: 保守采样，序列相似性高，质量稳定
- **0.1-0.5**: 平衡采样，适度多样性
- **0.5-1.0**: 激进采样，高多样性
- **1.0-2.0**: 极端随机，可能产生不稳定序列

**使用策略**:
```json
// 功能蛋白设计 - 保守确保功能
"sampling_temperature": 0.05

// 平衡设计 - 默认推荐
"sampling_temperature": 0.1  

// 探索性设计 - 获得多样性
"sampling_temperature": 0.3

// 创新性设计 - 高风险高回报
"sampling_temperature": 0.8
```

##### colabfold_use_online_msa (在线MSA开关)
```json
"colabfold_use_online_msa": "yes"
```
**作用**: 是否使用在线多序列比对提高预测精度

**选项详解**:
- `"yes"`: 
  - 优点: 显著提高预测精度（通常提升10-20%）
  - 缺点: 需要上传序列到远程服务器
  - 适用: 非敏感序列，追求最高精度

- `"no"`: 
  - 优点: 完全本地计算，保护数据隐私
  - 缺点: 预测精度较低
  - 适用: 敏感序列，专利相关设计

##### max_md_ns (分子动力学时长)
```json
"max_md_ns": 10
```
**作用**: 分子动力学模拟的时长（纳秒）
**时长建议**:
- **0**: 跳过MD（快速设计）
- **1-5**: 快速稳定性检查
- **10-50**: 标准稳定性评估
- **100+**: 深度稳定性研究

**调节考虑**:
```json
// 快速设计流程
"max_md_ns": 0

// 平衡质量和速度
"max_md_ns": 10

// 高精度验证
"max_md_ns": 50
```

#### C. 评分权重系统
```json
"scoring_weights": {
  "pLDDT": 0.4,           // ← 结构质量权重
  "rmsd": 0.3,            // ← 结构相似性权重
  "rosetta_score": 0.2,   // ← 能量稳定性权重
  "md_stability": 0.1     // ← 动力学稳定性权重
}
```

**权重调节策略**:

**结构研究优先**:
```json
{
  "pLDDT": 0.7,           // 强调结构质量
  "rmsd": 0.2,            // 适度考虑相似性
  "rosetta_score": 0.1,   // 基本能量要求
  "md_stability": 0.0     // 跳过MD节省时间
}
```

**功能蛋白优先**:
```json
{
  "pLDDT": 0.2,           // 降低结构要求
  "rosetta_score": 0.5,   // 强调能量稳定性
  "md_stability": 0.3     // 重视动力学稳定
}
```

**平衡模式**（推荐）:
```json
{
  "pLDDT": 0.4,           // 平衡结构质量
  "rmsd": 0.3,            // 适度结构要求
  "rosetta_score": 0.2,   // 基本能量稳定
  "md_stability": 0.1     // 简单动力学检查
}
```

---

## 2. mini_protein_targets.json - 预定义设计目标

### 文件作用
包含5个完整的蛋白质设计项目，每个都有详细的生物学背景和技术规格。

### 目标详细解析

#### A. 锌指蛋白设计
```json
{
  "target_id": "zinc_finger_design",
  "name": "经典锌指结构域",
  "description": "设计能够结合特定DNA序列的锌指蛋白",
  "target_length": 30,
  "secondary_structure": "beta-beta-alpha",
  "key_residues": {
    "zinc_coordinating": [3, 6, 23, 26],    // 锌配位残基
    "dna_contacting": [10, 13, 17, 20]      // DNA接触残基
  }
}
```

**生物学背景**:
锌指蛋白是最重要的DNA结合蛋白家族之一，在基因调控中发挥关键作用。每个锌指结构域识别3-4个DNA碱基。

**结构组成**:
- **beta发夹** (残基1-12): 包含两个锌配位的半胱氨酸
- **alpha螺旋** (残基13-30): 包含两个锌配位的组氨酸，直接接触DNA

**关键约束解释**:
```json
"constraints": {
  "fixed_positions": [3, 6, 23, 26],           // 绝对不能改变
  "required_residues": {"3": "C", "6": "C", "23": "H", "26": "H"},
  "forbidden_residues": ["P"],                 // 脯氨酸会破坏结构
  "target_motifs": ["CXXCXXXXXXXXXXXXHXXH"]    // 锌指共识序列
}
```

**可调节部分**:
- **DNA接触残基** (位置10,13,17,20): 改变这些可以调整DNA识别特异性
- **连接子区域** (位置7-12): 可以优化以改善稳定性
- **C端尾部** (位置27-30): 可以添加功能域或稳定性残基

**调节示例**:
```json
// 识别不同的DNA序列
"dna_contact_modifications": {
  "10": "R",    // 原来可能是K，改为R增强结合
  "13": "S",    // 原来可能是T，改为S调整特异性
  "17": "N",    // 原来可能是Q，改为N改变氢键模式
  "20": "K"     // 保持正电荷进行DNA骨架结合
}
```

#### B. WW结构域设计
```json
{
  "target_id": "ww_domain_design", 
  "name": "WW结构域变体",
  "description": "设计能够识别脯氨酸富集序列的WW域",
  "target_length": 38,
  "secondary_structure": "beta-turn-beta-turn-beta"
}
```

**生物学功能**:
WW结构域是蛋白质相互作用模块，专门识别和结合脯氨酸富集的序列（如PPXY基序）。

**结构特征**:
- **三股反平行beta折叠**
- **两个beta发夹转角**
- **脯氨酸结合槽**

**关键残基功能**:
```json
"key_residues": {
  "binding_groove": [9, 11, 26, 28],      // 形成脯氨酸结合槽
  "structural_core": [6, 15, 22, 33]      // 维持beta折叠结构
}
```

**可调节策略**:
```json
// 改变结合特异性
"binding_modifications": {
  "11": ["Y", "F"],     // 芳香残基，调整结合槽大小
  "28": ["P", "A"]      // 调整结合槽形状
}

// 增强结构稳定性
"stability_modifications": {
  "6": ["I", "L", "V"],   // 增强疏水核心
  "15": ["F", "W", "Y"],  // 芳香相互作用
  "22": ["L", "I", "V"],  // 疏水稳定
  "33": ["K", "R"]        // 表面静电稳定
}
```

#### C. 卷曲螺旋设计
```json
{
  "target_id": "coiled_coil_design",
  "name": "双链卷曲螺旋",
  "description": "设计稳定的同源二聚体卷曲螺旋",
  "target_length": 35
}
```

**结构原理**:
卷曲螺旋是两条或多条alpha螺旋缠绕形成的结构，遵循七联体重复模式。

**七联体模式详解**:
```
位置标记:  a b c d e f g | a b c d e f g | a b c d e f g | a b c d e f g | a b c
残基示例:  M K Q L E D K | V E L L S K N | Y H L E N E V | A R L K K L V | G E R
功能说明:  疏 + 表 疏 静 表 静 | 疏 + 表 疏 静 表 静 | 静 表 疏 + 表 静 + 表 | + 表 疏 + 表 + 疏 | 表 +
```

**位置功能**:
- **a,d位**: 疏水核心，两条螺旋间的主要接触
- **e,g位**: 静电相互作用，调节特异性和稳定性
- **b,c,f位**: 表面暴露，影响溶解性

**关键约束**:
```json
"heptad_repeat": {
  "a_positions": [1, 8, 15, 22, 29],      // 必须疏水
  "d_positions": [4, 11, 18, 25, 32],     // 必须疏水
  "e_positions": [5, 12, 19, 26, 33],     // 偏好带电
  "g_positions": [7, 14, 21, 28, 35]      // 偏好带电
}
```

**调节指南**:
```json
// 增强稳定性 - 更多疏水接触
"a_d_preferences": ["L", "I", "V", "F"]

// 调节特异性 - e,g位电荷配对
"electrostatic_strategy": {
  "homodimer": {"e": ["E"], "g": ["K"]},     // 同源二聚体
  "heterodimer": {"e": ["K"], "g": ["E"]}    // 异源二聚体
}

// 改善溶解性 - 表面极性
"surface_modifications": {
  "b_c_f_positions": ["S", "T", "N", "Q", "K", "R", "E", "D"]
}
```

#### D. 人工酶设计
```json
{
  "target_id": "enzyme_active_site",
  "name": "简化酶活性位点",
  "description": "设计包含催化三联体的迷你酶",
  "target_length": 65
}
```

**催化机制**:
模拟丝氨酸蛋白酶的催化三联体：Ser-His-Asp

**活性位点几何**:
```json
"active_site_geometry": {
  "ser15_his42_distance": 3.5,             // 丝氨酸-组氨酸距离
  "his42_asp58_distance": 3.0,             // 组氨酸-天冬氨酸距离
  "catalytic_angle": 105.0                 // 催化角度
}
```

**可调节部分**:
```json
// 底物结合位点
"substrate_binding_modifications": {
  "12": ["F", "W", "Y"],                   // 芳香堆积结合
  "18": ["K", "R"],                        // 静电结合
  "39": ["N", "Q"],                        // 氢键供体
  "45": ["S", "T"]                         // 羟基相互作用
}

// 催化效率优化
"catalytic_optimization": {
  "14": ["G"],                             // Ser15前的柔性
  "41": ["G"],                             // His42前的柔性
  "57": ["G"]                              // Asp58前的柔性
}
```

#### E. 膜蛋白设计
```json
{
  "target_id": "membrane_protein_helix",
  "name": "单次跨膜螺旋",
  "description": "设计稳定的跨膜alpha螺旋结构",
  "target_length": 45
}
```

**膜蛋白设计原理**:
跨膜蛋白必须适应脂质双分子层环境，需要精确的疏水性梯度。

**分域设计**:
```json
"domains": {
  "extracellular": {
    "positions": [1, 10],                  // 胞外段
    "environment": "aqueous",
    "preferred_residues": ["S", "T", "N", "Q", "E", "D", "K", "R"]
  },
  "transmembrane": {
    "positions": [11, 40],                 // 跨膜段  
    "environment": "lipid_bilayer",
    "preferred_residues": ["A", "L", "I", "V", "F", "Y", "W"],
    "forbidden_residues": ["E", "D", "K", "R", "P"]
  },
  "intracellular": {
    "positions": [41, 45],                 // 胞内段
    "environment": "aqueous",
    "preferred_residues": ["S", "T", "N", "Q", "E", "D", "K", "R"]
  }
}
```

**疏水性要求**:
```json
"hydrophobicity_requirements": {
  "transmembrane_min": 0.8,               // 跨膜区至少80%疏水
  "terminal_max": 0.3                     // 末端区域最多30%疏水
}
```

---

## 3. design_constraints.json - 约束模板库

### 约束类型完全解析

#### A. 位置固定约束
```json
"fixed_positions": {
  "3": "C",     // 第3位必须是半胱氨酸
  "6": "C",     // 第6位必须是半胱氨酸
  "23": "H",    // 第23位必须是组氨酸
  "26": "H"     // 第26位必须是组氨酸
}
```

**应用场景详解**:

**金属结合位点**:
```json
// 锌指蛋白
"zinc_coordination": {"4": "C", "7": "C", "23": "H", "26": "H"}

// 铁硫簇结合
"iron_sulfur_cluster": {"15": "C", "18": "C", "45": "C", "48": "C"}

// 血红蛋白样
"heme_binding": {"20": "H", "64": "H"}
```

**催化位点**:
```json
// 丝氨酸蛋白酶三联体
"catalytic_triad": {"35": "S", "78": "H", "102": "D"}

// 半胱氨酸蛋白酶
"cysteine_protease": {"25": "C", "68": "H", "158": "N"}

// 金属蛋白酶
"metalloprotease": {"146": "H", "150": "H", "166": "E"}
```

**结构关键残基**:
```json
// 二硫键形成
"disulfide_bonds": {"8": "C", "24": "C", "41": "C", "55": "C"}

// 脯氨酸铰链
"proline_hinges": {"45": "P", "78": "P"}

// 甘氨酸柔性
"flexible_regions": {"23": "G", "67": "G", "89": "G"}
```

#### B. 位置偏好约束
```json
"position_preferences": {
  "5": ["L", "I", "V"],                    // 疏水核心偏好
  "10": ["K", "R", "H"],                   // 正电荷偏好
  "15": ["D", "E"],                        // 负电荷偏好
  "20": ["F", "W", "Y"]                    // 芳香残基偏好
}
```

**偏好类型与功能**:

**疏水核心偏好**:
```json
"hydrophobic_core": {
  "branched": ["I", "L", "V"],             // 分支疏水，紧密堆积
  "aromatic": ["F", "W", "Y"],             // 芳香堆积，强相互作用
  "flexible": ["A", "M"],                  // 灵活疏水，适应性好
  "large": ["F", "W", "Y", "L"]            // 大疏水，填充空间
}
```

**表面残基偏好**:
```json
"surface_residues": {
  "positive": ["K", "R", "H"],             // 与核酸结合
  "negative": ["D", "E"],                  // 与阳离子结合
  "polar": ["S", "T", "N", "Q"],           // 氢键网络
  "flexible": ["G", "S", "T"]              // 适应性结合
}
```

**功能特异性偏好**:
```json
"functional_preferences": {
  "dna_binding": ["K", "R", "H", "S", "T"], // DNA大沟结合
  "rna_binding": ["K", "R", "F", "W"],       // RNA结合，需芳香性
  "protein_binding": ["L", "I", "V", "F"],   // 蛋白质界面，疏水为主
  "metal_coordination": ["H", "C", "D", "E"] // 金属配位残基
}
```

#### C. 距离约束系统
```json
"distance_constraints": [
  {
    "atom1": {"residue": 3, "atom": "SG"},      // 半胱氨酸硫原子
    "atom2": {"residue": 23, "atom": "NE2"},    // 组氨酸氮原子
    "distance_range": [2.0, 3.5],              // 允许距离范围
    "description": "zinc_coordination"          // 约束类型说明
  }
]
```

**原子类型速查表**:
```
常用主链原子:
- N: 氨基氮原子
- CA: alpha碳原子  
- C: 羰基碳原子
- O: 羰基氧原子

常用侧链原子:
- CB: beta碳原子（除甘氨酸外都有）
- SG: 半胱氨酸硫原子
- NE2: 组氨酸咪唑氮原子
- OG: 丝氨酸羟基氧原子
- NZ: 赖氨酸氨基氮原子
- CZ: 精氨酸胍基碳原子
```

**距离约束应用**:

**共价键约束**:
```json
// 二硫键
{"atom1": {"residue": 5, "atom": "SG"}, 
 "atom2": {"residue": 28, "atom": "SG"},
 "distance_range": [2.0, 2.1]}           // 标准二硫键距离

// 金属配位键
{"atom1": {"residue": 12, "atom": "NE2"}, 
 "atom2": {"residue": 15, "atom": "SG"},
 "distance_range": [2.2, 2.8]}           // 锌配位距离
```

**非共价相互作用**:
```json
// 氢键
{"atom1": {"residue": 20, "atom": "OG"}, 
 "atom2": {"residue": 45, "atom": "NE2"},
 "distance_range": [2.5, 3.2]}           // 氢键距离

// 疏水接触
{"atom1": {"residue": 15, "atom": "CB"}, 
 "atom2": {"residue": 22, "atom": "CB"},
 "distance_range": [3.5, 5.0]}           // 疏水接触距离

// 芳香堆积
{"atom1": {"residue": 18, "atom": "CZ"}, 
 "atom2": {"residue": 35, "atom": "CZ"},
 "distance_range": [3.5, 4.5]}           // 芳香环中心距离
```

---

## 4. custom_constraints_template.json - 自定义约束指南

### 自定义约束构建完整教程

#### 基础约束构建
```json
{
  "constraint_id": "my_protein_v1",          // 您的蛋白质版本标识
  "name": "我的第一个设计蛋白",                // 显示名称
  "description": "学习用的简单设计案例",        // 详细描述
  "target_length": 40                       // 目标残基数
}
```

#### 固定位点设置详解
```json
"fixed_positions": {
  "5": "C",      // 第5位固定为半胱氨酸
  "12": "H",     // 第12位固定为组氨酸
  "25": "S",     // 第25位固定为丝氨酸
  "35": "D"      // 第35位固定为天冬氨酸
}
```

**设置原则**:
- **功能必需**: 催化残基、金属配位残基
- **结构关键**: 二硫键半胱氨酸、脯氨酸转角
- **特异性决定**: DNA接触残基、底物识别残基

**常用固定组合**:
```json
// 二硫键对
"disulfide_pair": {"8": "C", "35": "C"}

// 催化三联体
"serine_protease": {"25": "S", "45": "H", "78": "D"}

// 锌指配位
"zinc_finger": {"4": "C", "7": "C", "23": "H", "26": "H"}

// 血红蛋白配位
"heme_coordination": {"18": "H", "87": "H"}
```

#### 禁用残基策略
```json
"forbidden_residues": ["P", "G", "C"]
```

**禁用原因分析**:
- **脯氨酸(P)**: 破坏alpha螺旋，引入刚性转角
- **甘氨酸(G)**: 过度柔性，可能导致结构失序
- **半胱氨酸(C)**: 意外二硫键形成，结构错误
- **蛋氨酸(M)**: 易氧化，影响稳定性
- **色氨酸(W)**: 过大，可能造成空间冲突

**场景化禁用策略**:
```json
// 膜蛋白设计
"membrane_forbidden": ["E", "D", "K", "R", "P"]  // 跨膜区禁用带电残基

// 抗菌肽设计  
"antimicrobial_forbidden": ["C", "M", "P"]       // 避免氧化和结构问题

// 酶设计
"enzyme_forbidden": ["P", "G"]                   // 活性位点附近保持刚性

// 高温环境蛋白
"thermostable_forbidden": ["Q", "N", "C", "M"]   // 避免热不稳定残基
```

#### 位点偏好高级设置
```json
"position_preferences": {
  "core_hydrophobic": {
    "positions": [8, 15, 22, 29],
    "residues": ["L", "I", "V", "F"],
    "rationale": "疏水核心稳定"
  },
  "surface_polar": {
    "positions": [3, 18, 25, 32],
    "residues": ["S", "T", "N", "Q"],
    "rationale": "表面极性增加溶解性"
  },
  "electrostatic_network": {
    "positions": [12, 27],
    "residues": ["K", "R", "E", "D"],
    "rationale": "静电相互作用稳定"
  }
}
```

## 实际修改练习

### 练习1: 设计稳定的小蛋白

**目标**: 基于Trp-cage设计一个更稳定的20残基蛋白

**原序列**: `NLYIQWLKDGGPSSGRPPPS`

**分析关键点**:
- W6: 绝对关键，不能改变
- G10,G11: 转角必需，不能改变
- P12,P17,P18,P19: beta股刚性，可以适当调整

**改进策略**:
```json
{
  "fixed_positions": {"6": "W"},           // 保持关键色氨酸
  "position_preferences": {
    "2": ["I", "V"],                       // L→I/V增强疏水性
    "4": ["L", "F"],                       // I→L/F增强疏水核心
    "16": ["K"]                            // R→K调整电荷分布
  },
  "forbidden_residues": ["M", "C"]         // 避免氧化和二硫键
}
```

### 练习2: 设计DNA结合蛋白变体

**目标**: 基于锌指设计识别新DNA序列的蛋白

**原序列**: `PYKCPECGKSFSQSSDLVKHQRIHTGEKP`
**原识别序列**: 5'-GCG-3'
**目标识别**: 5'-ATG-3'

**设计思路**:
```json
{
  "fixed_positions": {
    "4": "C", "7": "C", "23": "H", "26": "H"  // 锌配位不变
  },
  "dna_contact_modifications": {
    "16": "K",                              // 增强A-T结合
    "19": "N",                              // 与T形成氢键
    "20": "S"                               // 与G形成氢键
  },
  "stability_enhancements": {
    "8": "K",                               // 增强DNA骨架结合
    "14": "F"                               // 增加疏水稳定性
  }
}
```

### 练习3: 设计抗菌肽

**设计要求**:
- 长度: 20-30残基
- 净正电荷: +3到+6
- 疏水性: 40-60%
- 结构: alpha螺旋

**约束设计**:
```json
{
  "target_length": 25,
  "charge_distribution": {
    "positive_positions": [1, 5, 9, 13, 17, 21, 25],  // 均匀分布正电荷
    "preferred_residues": ["K", "R"]
  },
  "hydrophobic_pattern": {
    "hydrophobic_positions": [3, 7, 11, 15, 19, 23],  // 疏水面
    "preferred_residues": ["A", "L", "I", "V", "F"]
  },
  "amphipathic_helix": {
    "polar_face": [2, 6, 10, 14, 18, 22],            // 极性面
    "hydrophobic_face": [3, 7, 11, 15, 19, 23]       // 疏水面
  },
  "forbidden_residues": ["C", "M", "P", "G"]          // 避免不稳定因素
}
```

## 高级调节技巧

### 多目标优化
```json
{
  "multi_objective": {
    "primary_goal": "stability",           // 主要目标
    "secondary_goal": "binding_affinity",  // 次要目标
    "constraints": {
      "stability_requirements": {
        "min_hydrophobic_core": 6,         // 至少6个疏水核心残基
        "max_surface_hydrophobic": 3       // 最多3个表面疏水残基
      },
      "binding_requirements": {
        "key_contact_residues": [15, 18, 22], // 关键接触残基
        "electrostatic_complementarity": true  // 静电互补性
      }
    }
  }
}
```

### 进化压力模拟
```json
{
  "evolutionary_constraints": {
    "conserved_regions": [10, 15, 20, 25],   // 高度保守区域
    "variable_regions": [5, 30, 35],         // 允许变化区域
    "purifying_selection": {
      "hydrophobic_core": "strict",          // 疏水核心严格保守
      "surface_loops": "relaxed"             // 表面环区域宽松
    }
  }
}
```

### 环境适应性设计
```json
{
  "environmental_adaptation": {
    "temperature_stability": {
      "thermophilic": {                      // 高温环境
        "preferred": ["I", "V", "F", "W"],   // 增加疏水相互作用
        "forbidden": ["Q", "N", "C", "M"]    // 避免热不稳定残基
      },
      "psychrophilic": {                     // 低温环境
        "preferred": ["G", "S", "T", "A"],   // 增加柔性
        "forbidden": ["P"]                   // 避免过度刚性
      }
    },
    "ph_stability": {
      "acidic": {                            // 酸性环境
        "preferred": ["K", "R", "H"],        // 增加碱性残基
        "forbidden": ["D", "E"]              // 减少酸性残基
      },
      "basic": {                             // 碱性环境
        "preferred": ["D", "E"],             // 增加酸性残基
        "forbidden": ["K", "R"]              // 减少碱性残基
      }
    }
  }
}
```

## 错误排查指南

### 常见配置错误

#### 错误1: 约束冲突
```json
// 错误示例
{
  "fixed_positions": {"5": "K"},           // 固定为赖氨酸
  "forbidden_residues": ["K"],             // 又禁用赖氨酸
  "position_preferences": {
    "5": ["D", "E"]                        // 又偏好酸性残基
  }
}

// 正确示例
{
  "fixed_positions": {"5": "K"},           // 固定为赖氨酸
  "forbidden_residues": ["P", "G"],        // 禁用其他残基
  "position_preferences": {
    "6": ["D", "E"]                        // 在相邻位点设置偏好
  }
}
```

#### 错误2: 过度约束
```json
// 错误示例 - 约束过严
{
  "target_length": 30,
  "fixed_positions": {
    "1": "M", "2": "K", "3": "Q", "4": "L", "5": "E",
    "6": "D", "7": "K", "8": "V", "9": "E", "10": "E"  // 固定了33%
  }
}

// 正确示例 - 合理约束
{
  "target_length": 30,
  "fixed_positions": {
    "5": "C", "28": "C"                    // 仅固定关键的二硫键
  },
  "position_preferences": {
    "10": ["L", "I", "V"],                 // 其他位点使用偏好
    "15": ["K", "R"],
    "20": ["F", "W", "Y"]
  }
}
```

#### 错误3: 不现实的成功标准
```json
// 错误示例 - 标准过高
{
  "success_criteria": {
    "plddt_threshold": 98.0,               // 几乎不可能达到
    "rosetta_score_max": -200.0,           // 极不现实
    "binding_affinity": "< 1 pM"           // 过于严格
  }
}

// 正确示例 - 合理标准
{
  "success_criteria": {
    "plddt_threshold": 70.0,               // 良好质量
    "rosetta_score_max": -50.0,            // 可达到的稳定性
    "binding_affinity": "< 100 nM"         // 实用的结合力
  }
}
```

### 性能优化建议

#### 计算资源有限时
```json
{
  "resource_limited_config": {
    "num_backbones": 50,                   // 减少骨架数
    "num_sequences_per_backbone": 20,      // 减少序列数
    "esmfold_pLDDT_threshold": 55.0,      // 降低过滤标准
    "max_md_ns": 0,                       // 跳过MD模拟
    "colabfold_use_online_msa": "no"      // 仅本地计算
  }
}
```

#### 追求高质量时
```json
{
  "high_quality_config": {
    "num_backbones": 500,                  // 大量骨架搜索
    "num_sequences_per_backbone": 100,     // 充分序列采样
    "esmfold_pLDDT_threshold": 80.0,      // 高质量过滤
    "sampling_temperature": 0.05,          // 保守采样
    "max_md_ns": 50,                       // 深度MD验证
    "rosetta_relax_rounds": 5              // 充分结构优化
  }
}
```

## 实验验证准备

### 合成可行性检查
```json
{
  "synthesis_considerations": {
    "avoid_difficult_sequences": {
      "tandem_prolines": "PP",              // 避免连续脯氨酸
      "poly_glycine": "GGG",                // 避免多个甘氨酸
      "charge_clusters": "KKKK",            // 避免电荷簇集
      "hydrophobic_patches": "FFFFFF"       // 避免疏水簇集
    },
    "expression_optimization": {
      "avoid_rare_codons": true,            // 避免稀有密码子
      "optimize_for_ecoli": true,           // 针对大肠杆菌优化
      "add_purification_tags": {
        "his_tag": "HHHHHH",               // 组氨酸标签
        "flag_tag": "DYKDDDDK"             // FLAG标签
      }
    }
  }
}
```

### 功能验证设计
```json
{
  "functional_assays": {
    "binding_assay": {
      "method": "EMSA",                     // 电泳迁移率变化
      "target": "specific_DNA_sequence",
      "controls": ["original_protein", "negative_control"]
    },
    "enzymatic_assay": {
      "method": "colorimetric",             // 比色法检测
      "substrate": "synthetic_peptide",
      "expected_activity": "> 10% of natural enzyme"
    },
    "stability_assay": {
      "thermal": "CD_spectroscopy",         // 圆二色光谱
      "chemical": "GuHCl_denaturation",     // 盐酸胍变性
      "proteolytic": "trypsin_resistance"   // 胰蛋白酶抗性
    }
  }
}
```

## 进阶应用

### 批量设计配置
```json
{
  "batch_design": [
    {
      "batch_id": "length_series",
      "description": "不同长度的蛋白质系列",
      "variants": [
        {"target_length": 30, "name": "mini_v1"},
        {"target_length": 50, "name": "medium_v1"},
        {"target_length": 80, "name": "large_v1"}
      ]
    },
    {
      "batch_id": "stability_series", 
      "description": "不同稳定性的变体",
      "variants": [
        {"sampling_temperature": 0.05, "name": "conservative"},
        {"sampling_temperature": 0.2, "name": "moderate"},
        {"sampling_temperature": 0.5, "name": "diverse"}
      ]
    }
  ]
}
```

### 机器学习辅助配置
```json
{
  "ml_assisted_design": {
    "feature_importance": {
      "sequence_features": 0.4,             // 序列特征权重
      "structural_features": 0.3,           // 结构特征权重
      "evolutionary_features": 0.2,         // 进化特征权重
      "physicochemical_features": 0.1       // 理化特征权重
    },
    "active_learning": {
      "initial_samples": 100,               // 初始采样数
      "acquisition_function": "expected_improvement",
      "stopping_criteria": "convergence"
    }
  }
}
```

## 总结

这些JSON配置文件是您控制蛋白质设计的强大工具。掌握它们的使用方法，您可以：

1. **精确控制设计过程** - 从骨架生成到最终筛选
2. **实现特定功能要求** - 通过约束实现特定生物学功能
3. **优化设计策略** - 根据目标调整参数权重
4. **批量处理设计** - 系统性地探索设计空间
5. **为实验做准备** - 考虑合成和验证的实际需求

**学习路径建议**:
1. **理解基础** - 从默认配置开始，理解每个参数的作用
2. **小幅修改** - 改变1-2个参数，观察结果变化
3. **模板应用** - 使用预定义的约束模板
4. **自定义设计** - 根据特定需求构建自己的约束
5. **优化迭代** - 基于结果反馈持续改进

记住：蛋白质设计是科学和艺术的结合，需要理论知识、实践经验和创新思维。这些配置文件为您提供了坚实的起点，剩下的就是发挥您的创造力！

---

**相关文档**: 
- 查看 `../examples/EXAMPLES_GUIDE.md` 了解具体的蛋白质案例
- 查看项目根目录的 `README.md` 了解完整流程
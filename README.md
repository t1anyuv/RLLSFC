# LETI

## 项目概述

本项目旨在通过强化学习优化 TShape 索引的编码设计，学习一个全局遍历顺序 O，使得相似度高的节点编码连续，从而最小化期望查询代价。

### TShape 索引原理

1. **扩大元素表示**：用一个合适的扩大元素（α×β cells）表示一条轨迹
2. **编码方式**：深度优先遍历顺序（从层级1到层级g），在Cell中使用Z曲线顺序
3. **查询方法**：找到所有相交/覆盖的扩大元素，经过形状过滤后合并为连续区间，执行扫描和refinement

### 优化思路

#### 1. 成本模型

给定合并后的查询区间 $R_Q(O) = {[s_1, e_1), [s_2, e_2), ..., [s_m, e_m)}$，定义：

- 总覆盖长度：$L_Q = Σ(e_i - s_i)$
- 成本模型：$cost(Q;O) = τ_loc × m + τ_scan × L_Q$
  - $m$: 连续区间数量
  - $L_Q$: 每个区间内的索引数量

**目标**：$min_O E_Q[cost(Q;O)]$

#### 2. 强化学习设计

**状态 (State)**

- 当前访问位置
- 已访问节点集合
- 当前空间分布特征

**动作 (Action)**
从"未访问节点集合"中选择下一个节点，限制在：

- 当前节点同层的最多8个节点（8个方位）
- 当前节点下一层的4个子节点
- 8个邻居节点的父节点

**奖励 (Reward)**

- **邻近性奖励**：基于几何距离，衡量空间邻近性
- **相似性奖励**：基于Jaccard相似度，统计两个节点所包含轨迹的相似度
- **全局目标**：与深度优先Z曲线baseline比较，必须优于原始设计才采纳

## 快速开始

### 1. 安装依赖

```bash
# 安装项目（开发模式）
pip install -e .

# 或安装所有依赖
pip install -r requirements.txt
```

### 2. 运行实验

```bash
# 调试训练（快速测试）
python -m scripts.experiments.train_debug --config resource/default.yaml

# 运行完整训练流水线
python -m scripts.experiments.run_pipeline --config resource/default.yaml --name my_experiment
```

## 项目结构

```
LearnedTShape/
│
├── resource/                  # 资源目录
│   ├── shared/                # 共享资源
│   │   └── similarity/        # 相似度矩阵
│   ├── queries/               # 查询数据集
│   │   ├── uniform/
│   │   ├── gaussian/
│   │   └── skewed/
│   ├── temp/                  # 临时文件
│   │   ├── curves/            # 训练曲线
│   │   └── orders/            # 遍历顺序
│   ├── experiments/           # 实验输出
│   │   ├── debug/
│   │   ├── gaussian/
│   │   └── skewed/
│   └── default.yaml           # 默认配置
│
├── src/                       # 源代码
│   ├── config.py              # 统一配置管理
│   ├── core/                  # 基础几何结构（BoundingBox等）
│   ├── data/                  # 数据加载（T-Drive、合成数据）
│   ├── indexing/              # 四叉树索引与编码
│   ├── features/              # 特征构建与轨迹统计
│   ├── reward/                # 奖励计算
│   ├── rl/                    # 强化学习（环境、Agent、PPO）
│   ├── storage/               # 轨迹存储设计
│   ├── training/              # 训练调度与流水线
│   ├── evaluation/            # 评估工具
│   └── utils/                 # 工具函数
│
├── scripts/                   # 脚本
│   ├── experiments/           # 实验脚本
│   │   ├── run_pipeline.py    # 完整训练流水线
│   │   └── train_debug.py     # 调试训练
│   ├── preprocess/            # 数据预处理
│   │   ├── clean_tdrive.py    # 清洗T-Drive数据
│   │   ├── clean_cdtaxi.py    # 清洗CD-Taxi数据
│   │   ├── generate_query_dataset.py  # 生成查询数据集
│   │   ├── similarity_matrix.py       # 计算相似度矩阵
│   │   ├── split_query_datasets.py    # 划分数据集
│   │   └── augment_trajectories.py    # 轨迹增强
│   └── analyze/               # 结果分析
│       ├── analyze_dataset.py         # 数据集分析
│       ├── raw_traj_distribution.py   # 原始轨迹分布
│       └── prune_traj_distribution.py # 剪枝后分布
│
├── docs/                      # 文档
│   ├── gpu_acceleration_guide.md       # GPU加速指南
│   ├── paper_introduction_optimized.md # 论文介绍
│   ├── trajectory_storage_design.md    # 轨迹存储设计
│   └── traversal_training_flowchart.md # 训练流程图
│
├── tests/                     # 测试
│   ├── conftest.py            # pytest配置
│   ├── test_config.py         # 配置测试
│   └── test_generate_queries.py # 查询生成测试
│
├── pyproject.toml             # 项目配置
├── requirements.txt           # 依赖列表
└── pytest.ini               # pytest配置
```

## 配置系统

项目使用统一的 YAML 配置文件，位于 `resource/default.yaml`：

```yaml
experiment:
  name: gaussian               # 实验名称
  description: "高斯查询实验"   # 实验说明

index:
  max_level: 8                 # 四叉树最大层级
  alpha: 3                     # 横向划分数
  beta: 3                      # 纵向划分数

data:
  num_trajectories: -1         # 轨迹数量（-1表示全部）
  use_tdrive_data: true        # 是否使用真实数据

query:
  type: gaussian               # 查询类型: uniform/gaussian/skewed
  dataset: gaussian_1000m      # 查询数据集名称
  size: 200                    # 查询数量

reward:
  tau_loc: 1.0                 # 定位成本系数
  tau_scan: 0.5                # 扫描成本系数
  gamma: 0.99                  # 折扣因子

train:
  num_episodes: 400            # 训练轮数
  lr_actor: 0.0003             # Actor学习率
  lr_critic: 0.0003            # Critic学习率

network:
  hidden_dims: [256, 256]      # 网络隐藏层
  dropout: 0.1                 # Dropout率
```

## 数据预处理

### 1. 清洗原始数据

```bash
# 清洗T-Drive数据
python -m scripts.preprocess.clean_tdrive

# 清洗CD-Taxi数据
python -m scripts.preprocess.clean_cdtaxi
```

### 2. 生成查询数据集

```bash
# 生成不同类型查询
python -m scripts.preprocess.generate_query_dataset --type uniform --size 200
python -m scripts.preprocess.generate_query_dataset --type gaussian --size 200
python -m scripts.preprocess.generate_query_dataset --type skewed --size 200
```

### 3. 计算相似度矩阵

```bash
python -m scripts.preprocess.similarity_matrix --config resource/default.yaml
```

### 4. 分析数据集

```bash
# 分析轨迹分布
python -m scripts.analyze.analyze_dataset

# 查看原始分布
python -m scripts.analyze.raw_traj_distribution
```

## 实验管理

### 创建新实验

1. 复制默认配置：
```bash
cp resource/default.yaml resource/experiments/my_exp/config.yaml
```

2. 编辑配置文件修改参数

3. 运行实验：
```bash
python -m scripts.experiments.run_pipeline --config resource/experiments/my_exp/config.yaml --name my_exp
```

### 实验输出结构

每个实验的输出保存在 `resource/experiments/<experiment_name>/` 目录：

```
resource/experiments/my_exp/
├── config.yaml              # 配置副本
├── models/                  # 模型检查点
│   ├── model_ep_100.pth
│   └── final_model.pth
├── orders/                  # 学习到的遍历顺序
│   └── learned_order.json
├── curves/                  # 训练曲线
│   └── training_curve.png
└── logs/                    # 训练日志
```

## 数据格式

### TDrive 数据

TDrive数据文件格式：

```
3644-3644_1-MULTIPOINT Z((116.37497 39.85789 1201930859000), (116.37542 39.85764 1201930931000), ...)
```

每行代表一条轨迹，包含：

- 轨迹ID：`ID-ID_SEQ`
- 轨迹点：`(lon lat time)` 格式的点序列

### CD-Taxi 数据

CD-Taxi（成都出租车）数据文件格式：

```
[7dceae818438b836e3d306296b4ccfbd,[["2018-09-30 19:15:38.0",104.04235,30.69204],["2018-09-30 19:18:23.0",104.04389,30.69443]]]
```

每行代表一条轨迹，包含：

- 轨迹ID：MD5哈希字符串
- 轨迹点：`[timestamp, lon, lat]` 格式的点序列

**地理范围**：成都市区域（约 102.0°E - 105.5°E, 29.5°N - 32.0°N）

### 环境变量

```bash
# TDrive数据路径
export TDRIVE_DATA_DIR=/path/to/tdrive
export TDRIVE_DATA_PATH=/path/to/tdrive/data.txt

# CD-Taxi数据路径
export CDTAXI_DATA_DIR=/path/to/cdtaxi
export CDTAXI_DATA_PATH=/path/to/cdtaxi/data.txt

# 资源目录
export RESOURCE_BASE_DIR=resource

# 项目根目录（可选）
export PROJECT_ROOT=/path/to/project
```

## 核心组件

### 1. 配置管理 (`src/config.py`)

- `TShapeConfig`: 统一配置类
- 支持 YAML/JSON 格式
- 模块化配置：experiment, index, data, query, reward, train, network

### 2. 索引系统 (`src/indexing/`)

- `QuadTreeIndex`: 四叉树索引构建
- `TShapeEncoder`: TShape编码器
- `ZOrderEncoder`: Z曲线编码

### 3. 强化学习 (`src/rl/`)

- `TraversalEnv`: 遍历环境
- `ActorCriticAgent`: Actor-Critic智能体
- `PPOUpdater`: PPO更新器
- `RolloutBuffer`: 经验回放缓冲区

### 4. 训练系统 (`src/training/`)

- `TraversalTrainer`: 训练调度器
- `TrainingPipeline`: 完整流水线
- `CheckpointManager`: 检查点管理

### 5. 评估系统 (`src/evaluation/`)

- `TraversalEvaluator`: 遍历顺序评估
- `LSFCEvaluator`: LSFC性能评估

### 6. 数据与存储 (`src/data/`, `src/storage/`)

- `TDriveLoader`: T-Drive数据加载
- `SyntheticTrajectoryFactory`: 合成轨迹生成
- `TrajectoryStore`: 轨迹存储管理


## 运行测试

```bash
# 运行所有测试
pytest

# 运行特定测试
pytest tests/test_config.py
pytest tests/test_generate_queries.py
```

## 引用

如果你在研究中使用了本项目，请引用：

```
[添加引用信息]
```

# LearnedTShape

## 1. 环境准备

```bash
pip install -e .
```

如果不走开发模式，也可以：

```bash
pip install -r requirements.txt
```

可选环境变量：

```bash
set DATASET_TDRIVE_PATH=D:\dataset\Trajectory\TDrive\complete_clean\tdrive.txt
set DATASET_CHENGDU_PATH=D:\dataset\Trajectory\Chengdu\cleaned_cd_taxi.txt
set OUTPUT_DIR=D:\projects\LearnedTShape\outputs
```

## 2. 目录与资源

### 2.1 配置目录

- `configs/experiments/default/config.yaml`
- `configs/experiments/debug/config.yaml`
- `configs/experiments/test/config.yaml`
- `configs/experiments/formal/config.yaml`
- `configs/experiments/gaussian/config.yaml`
- `configs/experiments/skewed/config.yaml`
- `configs/experiments/uniform/config.yaml`

参数扫描默认配置来源：

- `configs/order_sweeps/tdrive/<distribution>/config.yaml`
- `configs/order_sweeps/cd_taxi/<distribution>/config.yaml`

### 2.2 输入资源目录

- `resource/queries/tdrive/`
- `resource/queries/chengdu/`
- `resource/matrices/similarity/`
- `resource/orders/`

其中：

- `resource/queries/<dataset>/` 保存查询数据
- `resource/matrices/similarity/` 保存共享相似度矩阵
- `resource/orders/` 保存共享导出顺序，主要用于 XZ 顺序和批量扫参导出

### 2.3 输出目录

- `outputs/experiments/<experiment_name>/checkpoints/`
- `outputs/experiments/<experiment_name>/results/`
- `outputs/experiments/<experiment_name>/logs/`
- `outputs/experiments/<experiment_name>/figures/`
- `outputs/experiments/param_orders/`

其中：

- `checkpoints/` 保存模型与 metrics
- `results/` 保存顺序 JSON、metadata、best_model_record
- `logs/` 保存训练日志与 summary
- `figures/` 保存训练曲线图
- `param_orders/` 保存扫参汇总结果

## 3. 数据要求

### 3.1 轨迹数据

配置里按数据集读取轨迹文件：

- `tdrive`
- `cdtaxi`

轨迹路径来源有两种：

- 在 YAML 里直接写 `datasets.profiles.<dataset>.trajectory_path`
- 通过环境变量 `DATASET_TDRIVE_PATH` / `DATASET_CHENGDU_PATH`

### 3.2 查询数据

训练和评估要求查询目录下存在以下结构：

```text
resource/queries/tdrive/
  gaussian/
    queries_train.json
    queries_val.json
    queries_test.json
  skewed/
    queries_train.json
    queries_val.json
    queries_test.json
  uniform/
    queries_train.json
    queries_val.json
    queries_test.json
```

`chengdu` 同理。

### 3.3 相似度矩阵

如果配置里启用了 `use_similarity_matrix: true`，脚本会读取：

- 显式指定的 `data.similarity_matrix_path`
- 或默认共享矩阵目录 `resource/matrices/similarity/`

默认共享矩阵文件名规则：

- `sim_mtx_<dataset>_<query_distribution_type>_R<resolution>_M<minTrajs>_A<alpha>_B<beta>_T<num_trajectories>.npz`

## 4. 各脚本运行命令

### 4.1 生成查询数据

按内置数据集配置生成：

```bash
python scripts/preprocess/generate_queries.py --dataset tdrive
python scripts/preprocess/generate_queries.py --dataset chengdu
```

手动指定边界、轨迹文件和输出目录：

```bash
python scripts/preprocess/generate_queries.py ^
  --min-lon 115.29 ^
  --min-lat 39.00 ^
  --max-lon 117.83 ^
  --max-lat 41.50 ^
  --traj-path D:\dataset\Trajectory\TDrive\complete_clean\tdrive.txt ^
  --output-dir resource/queries/tdrive
```

产物：

- `resource/queries/<dataset>/range/<distribution>/*.txt`
- `resource/queries/<dataset>/<distribution>/queries_train.json`
- `resource/queries/<dataset>/<distribution>/queries_val.json`
- `resource/queries/<dataset>/<distribution>/queries_test.json`

### 4.2 划分查询数据

按默认目录处理：

```bash
python scripts/preprocess/split_query_datasets.py --all
```

指定目录和划分比例：

```bash
python scripts/preprocess/split_query_datasets.py ^
  --categories gaussian skewed uniform ^
  --train-ratio 0.6 ^
  --val-ratio 0.2 ^
  --queries-dir resource/queries/tdrive ^
  --output-dir resource/queries/tdrive
```

### 4.3 预计算相似度矩阵

```bash
python -m scripts.preprocess.similarity_matrix --config configs/experiments/default/config.yaml
```

指定数据集、输出位置和 worker 数：

```bash
python -m scripts.preprocess.similarity_matrix ^
  --config configs/experiments/default/config.yaml ^
  --dataset tdrive ^
  --output-file resource/matrices/similarity/sim_mtx_tdrive_skewed_R8_M4_A2_B2_T-1.npz ^
  --num-workers 8 ^
  --force
```

### 4.4 调试训练

```bash
python -m scripts.experiments.train_debug --config configs/experiments/debug/config.yaml
```

输出位置：

- `outputs/experiments/debug/logs/`

### 4.5 运行完整实验

测试实验：

```bash
python -m scripts.experiments.run_pipeline --config configs/experiments/test/config.yaml --name test
```

正式实验：

```bash
python -m scripts.experiments.run_pipeline --config configs/experiments/formal/config.yaml --name formal
```

按分布基准配置运行：

```bash
python -m scripts.experiments.run_pipeline --config configs/experiments/gaussian/config.yaml --name gaussian
python -m scripts.experiments.run_pipeline --config configs/experiments/skewed/config.yaml --name skewed
python -m scripts.experiments.run_pipeline --config configs/experiments/uniform/config.yaml --name uniform
```

输出位置：

- `outputs/experiments/<name>/checkpoints/`
- `outputs/experiments/<name>/results/`
- `outputs/experiments/<name>/logs/`
- `outputs/experiments/<name>/figures/`

### 4.6 导出默认 XZ 顺序

使用默认配置：

```bash
python -m scripts.experiments.export_pruned_xz_order --config configs/experiments/default/config.yaml
```

覆盖关键参数：

```bash
python -m scripts.experiments.export_pruned_xz_order ^
  --config configs/experiments/default/config.yaml ^
  --dataset tdrive ^
  --max-level 8 ^
  --min-trajs 4 ^
  --num-trajectories -1 ^
  --output-file resource/orders/tdrive/skewed/pruned_xz_order.json
```

### 4.7 导出自适应划分

```bash
python -m scripts.experiments.export_adaptive_partitions --config configs/experiments/default/config.yaml
```

指定输出文件：

```bash
python -m scripts.experiments.export_adaptive_partitions ^
  --config configs/experiments/default/config.yaml ^
  --dataset tdrive ^
  --output-file resource/orders/tdrive/skewed/adaptive_partitions.json
```

### 4.8 合并顺序文件与划分信息

```bash
python -m scripts.experiments.merge_order_with_partitions ^
  --order-file resource/orders/tdrive/skewed/pruned_xz_order.json ^
  --partitions-file resource/orders/tdrive/skewed/adaptive_partitions.json
```

另存为新文件：

```bash
python -m scripts.experiments.merge_order_with_partitions ^
  --order-file resource/orders/tdrive/skewed/pruned_xz_order.json ^
  --partitions-file resource/orders/tdrive/skewed/adaptive_partitions.json ^
  --output-file resource/orders/tdrive/skewed/pruned_xz_order_merged.json
```

### 4.9 批量生成参数顺序

XZ 批量导出：

```bash
python -m scripts.experiments.generate_param_orders ^
  --distribution skewed ^
  --dataset tdrive ^
  --order-mode xz
```

RL 批量导出：

```bash
python -m scripts.experiments.generate_param_orders ^
  --distribution gaussian ^
  --dataset cdtaxi ^
  --order-mode rl ^
  --device auto ^
  --tag rl_batch
```

全网格扫参：

```bash
python -m scripts.experiments.generate_param_orders ^
  --distribution uniform ^
  --dataset tdrive ^
  --order-mode xz ^
  --full-grid ^
  --resolutions 6 7 8 9 10 ^
  --min-trajs 2 4 6 8 ^
  --alpha 2 ^
  --beta 2 ^
  --force
```

参数扫描配置默认读取：

- `configs/order_sweeps/tdrive/<distribution>/config.yaml`
- `configs/order_sweeps/cd_taxi/<distribution>/config.yaml`

产物位置：

- 汇总输出：`outputs/experiments/param_orders/<distribution_timestamp[_tag]>/`
- XZ 顺序文件：`resource/orders/<dataset>/<distribution>/`
- RL 单实验输出：`outputs/experiments/<case_name>/`

## 5. 推荐执行顺序

### 5.1 标准训练流程

```bash
python scripts/preprocess/generate_queries.py --dataset tdrive
python -m scripts.preprocess.similarity_matrix --config configs/experiments/default/config.yaml --dataset tdrive
python -m scripts.experiments.run_pipeline --config configs/experiments/test/config.yaml --name test
```

### 5.2 只导出非 RL 顺序

```bash
python -m scripts.experiments.export_pruned_xz_order --config configs/experiments/default/config.yaml --dataset tdrive
python -m scripts.experiments.export_adaptive_partitions --config configs/experiments/default/config.yaml --dataset tdrive
python -m scripts.experiments.merge_order_with_partitions --order-file resource/orders/tdrive/skewed/pruned_xz_order.json --partitions-file resource/orders/tdrive/skewed/adaptive_partitions.json
```

### 5.3 做参数扫描

```bash
python -m scripts.experiments.generate_param_orders --distribution skewed --dataset tdrive --order-mode xz
```

## 6. 故障检查

### 6.1 查询文件缺失

检查：

- `resource/queries/<dataset>/<distribution>/queries_train.json`
- `resource/queries/<dataset>/<distribution>/queries_val.json`
- `resource/queries/<dataset>/<distribution>/queries_test.json`

### 6.2 数据集路径缺失

检查：

- YAML 中的 `datasets.profiles.<dataset>.trajectory_path`
- 或环境变量 `DATASET_TDRIVE_PATH` / `DATASET_CHENGDU_PATH`

### 6.3 输出目录确认

当前统一规则：

- 输入资源看 `resource/`
- 实验结果看 `outputs/experiments/`
- 批量汇总看 `outputs/experiments/param_orders/`

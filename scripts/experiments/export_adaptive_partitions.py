"""Export signature-optimized adaptive partition information without traversal order."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from src.config import TShapeConfig
from src.rl.partition_formatter import AdaptivePartitionFormatter
from src.training.component_factory import TrainingComponentFactory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出自适应划分信息")
    parser.add_argument("--config", type=str, default="configs/experiments/default/config.yaml", help="YAML 配置文件路径")
    parser.add_argument("--dataset", type=str, default=None, help="覆盖激活数据集，例如 tdrive / cdtaxi")
    parser.add_argument("--max-level", type=int, default=None, help="覆盖四叉树层级")
    parser.add_argument("--min-trajs", type=int, default=None, help="覆盖 min_cell_trajs")
    parser.add_argument("--num-trajectories", type=int, default=None, help="覆盖轨迹数量")
    parser.add_argument("--source", type=str, choices=["dataset", "synthetic"], default=None, help="覆盖数据来源")
    parser.add_argument("--disable-optimize", action="store_true", help="关闭 signature optimize，仅导出全局划分")
    parser.add_argument("--export-prefix", type=str, default="adaptive_partitions", help="输出文件前缀")
    parser.add_argument("--output-file", type=str, default=None, help="显式输出路径")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> TShapeConfig:
    config = TShapeConfig.from_yaml(args.config)
    if args.dataset:
        config.datasets.active = args.dataset
    if args.max_level is not None:
        config.index.max_level = args.max_level
    if args.min_trajs is not None:
        config.index.min_cell_trajs = args.min_trajs
        config.index.use_prune = args.min_trajs > 0
    if args.num_trajectories is not None:
        config.data.num_trajectories = args.num_trajectories
    if args.source is not None:
        config.data.source = args.source
    if args.disable_optimize:
        config.index.enable_sig_optimize = False
    return config


def main() -> None:
    args = parse_args()
    config = build_config(args)

    factory = TrainingComponentFactory(config)
    quadtree = factory.create_quadtree()
    trajectories = factory.load_trajectories(quadtree)
    print(f"开始分配轨迹，共 {len(trajectories)} 条")
    for trajectory_id, points in trajectories:
        quadtree.assign_trajectory(trajectory_id, points)

    if config.index.use_prune and config.index.min_cell_trajs is not None:
        quadtree.post_prune_tree(config.index.min_cell_trajs)
    quadtree.compute_signatures(config.index.enable_sig_optimize)

    formatter = AdaptivePartitionFormatter(config=config)
    payload = formatter.export(quadtree)

    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        formatter.output_dir = output_path.parent
        filename = output_path.name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = (
            f"{args.export_prefix}_{config.datasets.active}_"
            f"res{config.index.max_level}_min{config.index.min_cell_trajs}_{timestamp}.json"
        )

    save_path = formatter.save(payload, filename)
    print(f"adaptive_partition_file: {save_path}")


if __name__ == "__main__":
    main()

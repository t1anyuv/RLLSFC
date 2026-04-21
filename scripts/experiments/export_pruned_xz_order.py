"""Export a pruned + shape-optimized traversal order using the default XZ order.

This flow skips RL entirely and only performs:
1. trajectory assignment
2. post-pruning
3. signature / shape optimization
4. export using the default XZ(Z-curve DFS) order
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from src.config import TShapeConfig
from src.indexing import TraversalOrderEncoder
from src.rl.order_formatter import TrajectoryOrderFormatter
from src.training.component_factory import TrainingComponentFactory
from src.utils.path_manager import get_path_manager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出仅剪枝+形状优化后的默认 XZ 顺序")
    parser.add_argument("--config", type=str, default="default.yaml", help="YAML 配置文件路径")
    parser.add_argument("--dataset", type=str, default=None, help="覆盖数据集，例如 tdrive / cdtaxi")
    parser.add_argument("--max-level", type=int, default=None, help="覆盖四叉树层级")
    parser.add_argument("--min-trajs", type=int, default=None, help="覆盖 min_cell_trajs")
    parser.add_argument("--num-trajectories", type=int, default=None, help="覆盖轨迹数量")
    parser.add_argument("--source", type=str, choices=["dataset", "synthetic"], default=None, help="覆盖数据来源")
    parser.add_argument("--disable-prune", action="store_true", help="关闭剪枝")
    parser.add_argument("--disable-optimize", action="store_true", help="关闭形状优化")
    parser.add_argument("--export-prefix", type=str, default="pruned_xz_order", help="输出文件前缀")
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
    if args.disable_prune:
        config.index.use_prune = False
        config.index.min_cell_trajs = 0
    if args.disable_optimize:
        config.index.enable_sig_optimize = False
    return config


def main() -> None:
    args = parse_args()
    config = build_config(args)
    config.paths.apply_to_path_manager()
    get_path_manager().set_experiment_name(config.experiment.name)

    factory = TrainingComponentFactory(config)
    quadtree = factory.create_quadtree()
    trajectories = factory.load_trajectories(quadtree)

    print(f"开始分配轨迹，共 {len(trajectories)} 条")
    for trajectory_id, points in trajectories:
        quadtree.assign_trajectory(trajectory_id, points)

    if config.index.use_prune and config.index.min_cell_trajs is not None:
        prune_stats = quadtree.post_prune_tree(config.index.min_cell_trajs)
        print(f"剪枝完成: active={prune_stats['after']}, muted={prune_stats['muted']}")
    else:
        print("跳过剪枝，保留原始 XZ 结构")

    optimize_stats = quadtree.compute_signatures(config.index.enable_sig_optimize)
    print(
        "形状优化完成: "
        f"alpha_shrunk={optimize_stats['shrunk_alpha']}, "
        f"beta_shrunk={optimize_stats['shrunk_beta']}, "
        f"both={optimize_stats['both_shrunk']}"
    )

    encoder = TraversalOrderEncoder(
        quadtree,
        alpha=config.index.alpha,
        beta=config.index.beta,
    )
    xz_order = encoder.z_curve_order(include_muted=False)

    formatter = TrajectoryOrderFormatter(config=config)
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        formatter.output_dir = output_path.parent
        filename = output_path.name
    else:
        filename = (
            f"{config.reward.query_distribution_type}_"
            f"r{config.index.max_level}_"
            f"min{config.index.min_cell_trajs}_"
            f"a{config.index.alpha}_"
            f"b{config.index.beta}.json"
        )

    _, save_path = formatter.generate_config_file_from_order(
        order=xz_order,
        quadtree=quadtree,
        filename=filename,
        global_alpha=config.index.alpha,
        global_beta=config.index.beta,
        order_source="pruned_default_xz_order",
    )

    print(f"active_order_length: {len(xz_order)}")
    print(f"order_file: {save_path}")


if __name__ == "__main__":
    main()

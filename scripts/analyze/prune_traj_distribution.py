from collections import defaultdict
from pathlib import Path
import os

import matplotlib.pyplot as plt
import numpy as np

from src.config import TShapeConfig
from src.data.tdrive_loader import load_cleaned_dataset
from src.indexing import QuadTreeIndex
from src.utils.path_manager import get_path_manager


def collect_index_stats(index: QuadTreeIndex) -> dict:
    """
    采集四叉树当前状态的统计信息。
    仅统计未被标记为 muted 的活跃单元格。
    """
    # 获取所有非屏蔽单元格
    active_cells = [cell for cell in index.all_cells.values() if not getattr(cell, 'muted', False)]

    stats = {
        "total_active_cells": len(active_cells),
        "level_distribution": defaultdict(int),  # 每个层级有多少条轨迹
        "cell_count_per_level": defaultdict(int),  # 每个层级有多少个单元格
        "load_distribution": [],  # 负载列表（用于统计分布）
        "levels_list": []  # 轨迹层级列表（用于计算平均层级）
    }

    for cell in active_cells:
        traj_count = len(cell.trajectories)
        stats["level_distribution"][cell.level] += traj_count
        stats["cell_count_per_level"][cell.level] += 1
        stats["load_distribution"].append(traj_count)
        stats["levels_list"].extend([cell.level] * traj_count)

    return stats


def plot_pruning_comparison(before: dict, after: dict, min_trajs: int, output_path: Path):
    """绘制对比图"""
    plt.style.use('seaborn-v0_8-muted')
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f"QuadTree Pruning Effect Analysis (Min Trajs: {min_trajs})", fontsize=16, fontweight='bold')

    # 1. 轨迹层级分布对比
    ax1 = axes[0]
    all_levels = sorted(set(list(before["level_distribution"].keys()) + list(after["level_distribution"].keys())))

    val_before = [before["level_distribution"].get(lvl, 0) for lvl in all_levels]
    val_after = [after["level_distribution"].get(lvl, 0) for lvl in all_levels]

    x = np.arange(len(all_levels))
    width = 0.35
    ax1.bar(x - width / 2, val_before, width, label='Original', color='gray', alpha=0.6)
    ax1.bar(x + width / 2, val_after, width, label='After Pruning', color='teal', alpha=0.8)

    ax1.set_xticks(x)
    ax1.set_xticklabels(all_levels)
    ax1.set_title("Trajectory Count Distribution by Level", fontweight='bold')
    ax1.set_xlabel("Quadtree Level")
    ax1.set_ylabel("Num Trajectories")
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # 2. 活跃单元格数量对比 (展示索引压缩率)
    ax2 = axes[1]
    cell_before = [before["cell_count_per_level"].get(l, 0) for l in all_levels]
    cell_after = [after["cell_count_per_level"].get(l, 0) for l in all_levels]

    ax2.plot(all_levels, cell_before, marker='o', label='Original Cells', color='orange', linestyle='--')
    ax2.plot(all_levels, cell_after, marker='s', label='Pruned Cells (Active)', color='green', linewidth=2)

    ax2.set_title("Active Cell Count per Level (Compression)", fontweight='bold')
    ax2.set_xlabel("Quadtree Level")
    ax2.set_ylabel("Num Active Cells")
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, which="both", linestyle='--', alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=180)
    plt.close()


def run_pruning_analysis(data_path: str, max_level: int, alpha: int, beta: int, min_cell_trajs: int):
    """运行剪枝分析
    
    Args:
        data_path: 轨迹数据文件路径
        max_level: 四叉树最大层级
        alpha: 扩展元素X方向单元数
        beta: 扩展元素Y方向单元数
        min_cell_trajs: 剪枝阈值
    """
    # 1. 环境准备
    from src.config import IndexConfig
    config = TShapeConfig(
        index=IndexConfig(
            max_level=max_level,
            alpha=alpha,
            beta=beta,
            min_cell_trajs=min_cell_trajs
        )
    )
    bbox = config.get_original_bbox()
    index = QuadTreeIndex(bbox, max_level=max_level, alpha=alpha, beta=beta)

    # 2. 加载与分配
    print(f"🚀 加载轨迹数据: {data_path}")
    trajectories = load_cleaned_dataset(data_path)
    print(f"正在分配 {len(trajectories)} 条轨迹到四叉树...")
    for tid, points in trajectories:
        index.assign_trajectory(tid, points)

    # 3. 采集剪枝前状态
    stats_before = collect_index_stats(index)
    print(f"剪枝前活跃单元格数: {stats_before['total_active_cells']}")

    # 4. 执行剪枝
    print(f"🔥 执行剪枝 (阈值: {min_cell_trajs})...")
    prune_report = index.post_prune_tree(min_cell_trajs=min_cell_trajs, enable_optimize=True)

    # 5. 采集剪枝后状态
    stats_after = collect_index_stats(index)
    print(f"剪枝后活跃单元格数: {stats_after['total_active_cells']}")
    print(f"索引压缩率: {(1 - stats_after['total_active_cells'] / stats_before['total_active_cells']):.2%}")

    # 6. 绘图
    output_dir = Path("scripts/analyze/analysis_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    img_path = output_dir / f"pruning_effect_L{max_level}_M{min_cell_trajs}.png"

    plot_pruning_comparison(stats_before, stats_after, min_cell_trajs, img_path)
    print(f"✨ 分析报告图已保存至: {img_path}")


if __name__ == "__main__":
    # 使用环境变量或默认路径
    pm = get_path_manager()
    PATH = pm.tdrive_data_path or os.environ.get('TDRIVE_DATA_PATH', 'data/tdrive/tdrive_cleaned.txt')
    run_pruning_analysis(data_path=str(PATH), max_level=9, alpha=3, beta=3, min_cell_trajs=5)


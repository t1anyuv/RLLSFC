import argparse
import json
import math
from collections import defaultdict, Counter
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import matplotlib.pyplot as plt
import numpy as np

from src.config import TShapeConfig
from src.core.bounding_box import SpatialBoundingBox
from src.utils.trajectory_geometry import compute_trajectory_bounding_box
from src.data.tdrive_loader import load_cleaned_dataset


def calculate_trajectory_level_and_cell(
        points: List[Tuple[float, float]],
        max_level: int,
        alpha: int,
        beta: int,
        root_bbox: SpatialBoundingBox
) -> Tuple[Optional[int], Optional[Tuple[int, int]], Optional[str]]:
    """
    利用 QuadTreeIndex 现有接口计算轨迹分配情况。
    """
    if not points:
        return None, None, "empty_points"

    # 1. 计算轨迹 MBR
    traj_bbox = compute_trajectory_bounding_box(points)

    # 2. 计算对应层级
    x1, y1, x2, y2 = traj_bbox.min_x, traj_bbox.min_y, traj_bbox.max_x, traj_bbox.max_y
    eps = 1e-10
    root_w = root_bbox.max_x - root_bbox.min_x
    root_h = root_bbox.max_y - root_bbox.min_y

    # 计算相对比例下的理想层级
    max_dim_rel = max((x2 - x1) / (alpha * root_w),
                      (y2 - y1) / (beta * root_h))

    l_suggested = max_level if max_dim_rel <= 0 else int(math.floor(-math.log2(max_dim_rel)))
    l_suggested = max(0, min(l_suggested, max_level))

    def check_containment(lvl):
        # 计算该层级单元格尺寸
        cw = root_w * (0.5 ** lvl)
        ch = root_h * (0.5 ** lvl)
        # 计算所属单元格的左下角坐标 (Grid 坐标)
        gx = math.floor((x1 - root_bbox.min_x + eps) / cw) * cw + root_bbox.min_x
        gy = math.floor((y1 - root_bbox.min_y + eps) / ch) * ch + root_bbox.min_y
        # 验证扩大后的边界 (Enlarged BBox) 是否包含轨迹右上角
        return (gx + alpha * cw >= x2 - eps) and (gy + beta * ch >= y2 - eps)

    # 决定最终层级
    target_lvl = l_suggested if check_containment(l_suggested) else l_suggested - 1
    target_lvl = max(0, min(target_lvl, max_level))

    # 3. 计算最终的 bucket 坐标
    final_cw = root_w * (0.5 ** target_lvl)
    final_ch = root_h * (0.5 ** target_lvl)
    bucket_x = int(math.floor((x1 - root_bbox.min_x + eps) / final_cw))
    bucket_y = int(math.floor((y1 - root_bbox.min_y + eps) / final_ch))

    # 4. 验证 EE 包含性
    ee_min_x = root_bbox.min_x + bucket_x * final_cw
    ee_min_y = root_bbox.min_y + bucket_y * final_ch
    if not (ee_min_x <= x1 + eps and ee_min_x + alpha * final_cw >= x2 - eps and
            ee_min_y <= y1 + eps and ee_min_y + beta * final_ch >= y2 - eps):
        return None, None, "enlarged_bbox_not_contains"

    return target_lvl, (bucket_x, bucket_y), None


def analyze_trajectory_resolution_streaming(
        data_path: str,
        max_level: int = 8,
        alpha: int = 3,
        beta: int = 3,
        original_bbox: tuple = None,
        max_trajectories: int = None,
        analyze_cell_distribution: bool = False  # 是否分析单元格轨迹数量分布
) -> dict:
    """流式分析轨迹的层级分布。"""

    # 1. 初始化边界框
    if original_bbox:
        bbox = SpatialBoundingBox(*original_bbox)
    else:
        bbox = TShapeConfig().get_original_bbox()

    print(f"\n[开始分析] 参数: max_level={max_level}, alpha={alpha}, beta={beta}")
    print(f"设定边界: {bbox}")

    # 2. 加载轨迹数据
    trajectories = load_cleaned_dataset(data_path, max_trajectories=max_trajectories)

    # 3. 初始化统计容器
    stats_internal = {
        "total_trajectories": 0,
        "assigned_trajectories": 0,
        "level_distribution": defaultdict(int),
        "levels": [],
        "unassigned_reasons": defaultdict(int),
        "actual_range": {"min_x": float('inf'), "max_x": float('-inf'),
                         "min_y": float('inf'), "max_y": float('-inf')},
    }

    cell_trajectory_counts = defaultdict(int) if analyze_cell_distribution else None

    # 3. 处理数据
    for traj_id, points in trajectories:
        stats_internal["total_trajectories"] += 1

        # 更新实际物理边界统计
        traj_bbox = compute_trajectory_bounding_box(points)
        stats_internal["actual_range"]["min_x"] = min(stats_internal["actual_range"]["min_x"], traj_bbox.min_x)
        stats_internal["actual_range"]["max_x"] = max(stats_internal["actual_range"]["max_x"], traj_bbox.max_x)
        stats_internal["actual_range"]["min_y"] = min(stats_internal["actual_range"]["min_y"], traj_bbox.min_y)
        stats_internal["actual_range"]["max_y"] = max(stats_internal["actual_range"]["max_y"], traj_bbox.max_y)

        # 计算分配层级
        level, cell_bucket, reason = calculate_trajectory_level_and_cell(
            points, max_level, alpha, beta, bbox
        )

        if level is not None:
            stats_internal["assigned_trajectories"] += 1
            stats_internal["level_distribution"][level] += 1
            stats_internal["levels"].append(level)
            if analyze_cell_distribution:
                cell_trajectory_counts[(level, cell_bucket[0], cell_bucket[1])] += 1
        else:
            stats_internal["unassigned_reasons"][reason] += 1
            if reason == "out_of_bounds" and len(stats_internal["out_of_bounds_samples"]) < 10:
                stats_internal["out_of_bounds_samples"].append({"id": traj_id, "bbox": traj_bbox})

    # 4. 汇总详细的单元格分布
    cell_stats = _process_cell_distribution(cell_trajectory_counts) if analyze_cell_distribution else {}

    return {
        "total_trajectories": stats_internal["total_trajectories"],
        "assigned_trajectories": stats_internal["assigned_trajectories"],
        "unassigned_trajectories": stats_internal["total_trajectories"] - stats_internal["assigned_trajectories"],
        "levels": stats_internal["levels"],
        "level_distribution": dict(stats_internal["level_distribution"]),
        "level_stats": {
            "min": np.min(stats_internal["levels"]) if stats_internal["levels"] else 0,
            "max": np.max(stats_internal["levels"]) if stats_internal["levels"] else 0,
            "mean": np.mean(stats_internal["levels"]) if stats_internal["levels"] else 0,
            "median": np.median(stats_internal["levels"]) if stats_internal["levels"] else 0,
            "std": np.std(stats_internal["levels"]) if stats_internal["levels"] else 0
        },
        "cell_distribution_by_level": cell_stats
    }


def _process_cell_distribution(cell_counts: Optional[Dict[Tuple[int, int, int], int]]) -> Dict[int, dict]:
    """
    处理单元格分布数据，生成按层级划分的统计摘要。
    """
    if not cell_counts:
        return {}

    # 按层级分组统计
    level_to_cells = defaultdict(list)
    for (level, bx, by), count in cell_counts.items():
        level_to_cells[level].append(count)

    processed_stats = {}
    range_bins = {
        "1": lambda sc: sc == 1,
        "2-5": lambda sc: 2 <= sc <= 5,
        "6-10": lambda sc: 6 <= sc <= 10,
        "11-20": lambda sc: 11 <= sc <= 20,
        "21-50": lambda sc: 21 <= sc <= 50,
        "51-100": lambda sc: 51 <= sc <= 100,
        "100+": lambda sc: c > 100
    }

    for level, counts in level_to_cells.items():
        total_trajs = sum(counts)
        num_cells = len(counts)

        # 计算区间分布
        dist = {k: 0 for k in range_bins.keys()}
        for c in counts:
            for label, check in range_bins.items():
                if check(c):
                    dist[label] += 1
                    break

        processed_stats[level] = {
            "total_cells": num_cells,
            "total_trajectories": total_trajs,
            "avg_trajectories_per_cell": total_trajs / num_cells if num_cells > 0 else 0,
            "trajectory_count_distribution": dist,
            "trajectory_count_exact": dict(Counter(counts))
        }

    return processed_stats


def save_resolution_analysis_file(
        stats: dict,
        bbox: SpatialBoundingBox,
        max_level: int,
        alpha: int,
        beta: int,
        output_dir: Optional[str] = None
) -> str:
    output_path = Path(output_dir) if output_dir else Path("scripts/analyze/analysis_output")
    output_path.mkdir(parents=True, exist_ok=True)

    # 转换为 JSON 可序列化格式 (处理 numpy 对象)
    def json_serializable(obj):
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    analysis_data = {
        "metadata": {
            "total_trajectories": stats["total_trajectories"],
            "assigned_trajectories": stats["assigned_trajectories"],
            "unassigned_trajectories": stats["unassigned_trajectories"],
            "params": {"max_level": max_level, "alpha": alpha, "beta": beta},
            "bbox": {"min_x": bbox.min_x, "min_y": bbox.min_y, "max_x": bbox.max_x, "max_y": bbox.max_y}
        },
        "level_distribution": stats["level_distribution"],
        "level_stats": stats["level_stats"],
        "cell_distribution_by_level": stats.get("cell_distribution_by_level", {})
    }

    file_name = f"res_analysis_L{max_level}_A{alpha}_B{beta}.json"
    output_file = output_path / file_name

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_data, f, indent=2, ensure_ascii=False, default=json_serializable)

    print(f"📊 分析数据已导出: {output_file}")
    return str(output_file)


def plot_resolution_distribution(stats: dict, max_lvl: int, output_dir: Optional[str] = None) -> None:
    import seaborn as sns

    # 设置美化样式
    sns.set_theme(style="whitegrid")

    output_path = Path(output_dir) if output_dir else Path("scripts/analyze/analysis_output")
    output_path.mkdir(parents=True, exist_ok=True)

    if not stats.get("levels"):
        print("Warning: No trajectories assigned")
        return

    level_dist = stats["level_distribution"]
    cell_dist_by_level = stats.get("cell_distribution_by_level", {})
    has_cell_dist = bool(cell_dist_by_level)
    level_stats = stats["level_stats"]
    sorted_levels = sorted(level_dist.keys())

    # 创建画布
    fig, axes = plt.subplots(2, 2, figsize=(18, 11)) if has_cell_dist else plt.subplots(1, 1, figsize=(10, 6))
    fig.suptitle('Quadtree Trajectory Resolution Analysis', fontsize=18, fontweight='bold', y=0.98)

    ax1 = axes[0, 0] if has_cell_dist else axes

    # --- 1. 轨迹数量分布 (Bar + CDF Line) ---
    counts = [level_dist[lvl] for lvl in sorted_levels]
    bars = ax1.bar(sorted_levels, counts, color=sns.color_palette("husl", len(sorted_levels)), edgecolor='black',
                   alpha=0.7)
    ax1.set_title('Trajectory Count & Cumulative Distribution', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Number of Trajectories')

    # 添加 CDF 曲线
    ax1_twin = ax1.twinx()
    cdf = np.cumsum(counts) / sum(counts) * 100
    ax1_twin.plot(sorted_levels, cdf, color='darkred', marker='D', markersize=4, label='CDF %')
    ax1_twin.set_ylabel('Cumulative Percentage (%)')
    ax1_twin.set_ylim(0, 105)
    ax1_twin.grid(False)

    # 统计信息文本框
    stats_text = (f"Total: {stats['total_trajectories']}\nAssigned: {stats['assigned_trajectories']}\n"
                  f"Mean Level: {level_stats['mean']:.2f}\nStd Dev: {level_stats['std']:.2f}")
    ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), family='monospace', fontsize=9)

    if has_cell_dist:
        # --- 2. 单元格数量 & 相对填充率 ---
        ax2 = axes[0, 1]
        sorted_cell_lvls = sorted(cell_dist_by_level.keys())
        cell_counts = [cell_dist_by_level[lvl]['total_cells'] for lvl in sorted_cell_lvls]
        fill_rates = [(cell_counts[i] / (4 ** l) * 100) for i, l in enumerate(sorted_cell_lvls)]

        ax2.bar(sorted_cell_lvls, cell_counts, color='coral', alpha=0.6, edgecolor='black', label='Active Cells')
        ax2.set_title('Active Cell Count & Relative Fill Rate', fontsize=13, fontweight='bold')

        ax2_twin = ax2.twinx()
        ax2_twin.plot(sorted_cell_lvls, fill_rates, 'g--', marker='o', label='Fill Rate %')
        ax2_twin.set_ylabel('Fill Rate (Active/Theoretical %)')
        ax2_twin.set_yscale('log')  # 填充率通常随层级指数下降，对数显示更清晰

        # --- 3. 平均负载 ---
        ax3 = axes[1, 0]
        avgs = [cell_dist_by_level[lvl]['avg_trajectories_per_cell'] for lvl in sorted_cell_lvls]
        sns.lineplot(x=sorted_cell_lvls, y=avgs, ax=ax3, marker='s', color='teal', linewidth=2.5)
        ax3.fill_between(sorted_cell_lvls, avgs, alpha=0.2, color='teal')
        ax3.set_title('Avg Trajectories per Cell (Load Balance)', fontsize=13, fontweight='bold')
        ax3.set_ylabel('Avg Load')

        # --- 4. 区间热度分布 (Grouped Bar) ---
        ax4 = axes[1, 1]
        range_labels = ["1", "2-5", "6-10", "11-20", "21-50", "51-100", "100+"]
        x = np.arange(len(sorted_cell_lvls))
        width = 0.1

        for i, label in enumerate(range_labels):
            vals = [cell_dist_by_level[lvl]['trajectory_count_distribution'].get(label, 0) for lvl in sorted_cell_lvls]
            ax4.bar(x + (i - 3) * width, vals, width, label=label, alpha=0.8)

        ax4.set_xticks(x)
        ax4.set_xticklabels(sorted_cell_lvls)
        ax4.set_title('Cell Density Distribution by Range', fontsize=13, fontweight='bold')
        ax4.legend(title="Trajs/Cell", fontsize=8, ncol=2)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path / f"trajectory_resolution_analysis_L{max_lvl}.png", dpi=200)
    plt.close()
    print(f"Analysis plot saved to {output_path}")


def print_statistics(stats: dict) -> None:
    """以美化格式打印统计摘要。"""
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + " 轨迹分辨率分布统计报告 ".center(50) + "║")
    print("╚" + "═" * 58 + "╝")

    print(f"\n📊 [基础数据]")
    total = stats['total_trajectories']
    assigned = stats['assigned_trajectories']
    unassigned = stats['unassigned_trajectories']
    assign_rate = (assigned / total * 100) if total > 0 else 0
    print(f"  - 总轨迹数:   {total:<10}")
    print(f"  - 成功分配:   {assigned:<10} ({assign_rate:.2%})")
    print(f"  - 无法分配:   {unassigned:<10}")

    l_stats = stats["level_stats"]
    print(f"\n📏 [层级统计]")
    print(f"  - 层级范围:   {int(l_stats['min'])} ~ {int(l_stats['max'])}")
    print(f"  - 平均层级:   {l_stats['mean']:.2f} (标准差: {l_stats['std']:.2f})")

    print(f"\n📈 [各层级轨迹占比]")
    level_dist = stats["level_distribution"]
    max_count = max(level_dist.values()) if level_dist else 1
    for lvl in sorted(level_dist.keys()):
        count = level_dist[lvl]
        pct = (count / assigned * 100) if assigned > 0 else 0
        bar = "■" * int(count / max_count * 20)
        print(f"  Level {lvl:>2}: {count:>8} ({pct:>6.2f}%) {bar}")

    # 单元格负载分析
    cell_data = stats.get("cell_distribution_by_level")
    if cell_data:
        print("\n🏠 [单元格负载分布 (Trajs per Cell)]")
        print(f"{'层级':<6} | {'活跃单元格':<10} | {'平均负载':<10} | {'峰值负载'}")
        print("-" * 55)
        for lvl in sorted(cell_data.keys()):
            info = cell_data[lvl]
            exact = info.get('trajectory_count_exact', {0: 0})
            max_load = max(exact.keys()) if exact else 0
            print(f"L{lvl:<5} | {info['total_cells']:<10} | {info['avg_trajectories_per_cell']:<10.2f} | {max_load}")

    print("\n" + "=" * 60)


def main():
    """
    示例用法:
    python analyze_resolution.py --data-path data/cleaned.txt --max-level 12 --analyze-cell-distribution
    """
    parser = argparse.ArgumentParser(description="分析轨迹分辨率分布 (Memory-Efficient)")

    # 路径参数
    parser.add_argument("--data-path", type=str, required=True, help="清洗后的轨迹文件路径 (txt/csv/npy)")
    parser.add_argument("--output-dir", type=str, default=None, help="结果输出目录")

    # 四叉树核心参数
    parser.add_argument("--max-level", type=int, default=12, help="四叉树最大深度 (默认: 12)")
    parser.add_argument("--alpha", type=int, default=3, help="EE 扩展系数 Alpha (默认: 3)")
    parser.add_argument("--beta", type=int, default=3, help="EE 扩展系数 Beta (默认: 3)")

    # 空间边界参数
    parser.add_argument("--original-bbox", type=float, nargs=4, default=None,
                        metavar=("MIN_X", "MIN_Y", "MAX_X", "MAX_Y"),
                        help="手动指定边界框。若不指定，则自动从 TShapeConfig 读取")

    # 控制参数
    parser.add_argument("--max-trajectories", type=int, default=None, help="限制处理轨迹条数")
    parser.add_argument("--analyze-cell-distribution", action="store_true",
                        help="是否开启细粒度单元格负载分析 (开启后会计算每个 Cell 的轨迹计数)")

    args = parser.parse_args()

    # 1. 统一边界框获取逻辑
    if args.original_bbox:
        bbox = SpatialBoundingBox(*args.original_bbox)
    else:
        bbox = TShapeConfig().get_original_bbox()

    # 2. 执行核心分析
    stats = analyze_trajectory_resolution_streaming(
        data_path=args.data_path,
        max_level=args.max_level,
        alpha=args.alpha,
        beta=args.beta,
        original_bbox=(bbox.min_x, bbox.min_y, bbox.max_x, bbox.max_y),
        max_trajectories=args.max_trajectories,
        analyze_cell_distribution=args.analyze_cell_distribution
    )

    # 3. 结果产出
    print_statistics(stats)
    plot_resolution_distribution(stats, args.max_level, args.output_dir)
    save_resolution_analysis_file(stats, bbox, args.max_level, args.alpha, args.beta, args.output_dir)

    print(f"\n✅ 分析任务完成。")


if __name__ == "__main__":
    main()

"""
轨迹数据集分析脚本
分析轨迹数据集的时间、空间、边界、分布信息

支持数据格式: tid|oid|seg_id|经度,纬度;经度,纬度;...
"""
import argparse
import json
import math
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np


def parse_trajectory_line(line: str) -> Optional[Tuple[int, str, int, List[Tuple[float, float, Optional[str]]]]]:
    """
    解析轨迹行数据
    
    格式: tid|oid|seg_id|lon,lat[;timestamp];lon,lat[;timestamp];...
    返回: (tid, oid, seg_id, [(lon, lat, timestamp_or_None), ...])
    """
    line = line.strip()
    if not line or line.startswith('#'):
        return None
    
    try:
        parts = line.split('|')
        if len(parts) < 4:
            return None
        
        tid = int(parts[0])
        oid = parts[1]
        seg_id = int(parts[2])
        
        points = []
        point_strs = parts[3].split(';')
        for ps in point_strs:
            ps = ps.strip()
            if not ps:
                continue
            coords = ps.split(',')
            if len(coords) >= 2:
                lon = float(coords[0])
                lat = float(coords[1])
                timestamp = coords[2] if len(coords) > 2 else None
                points.append((lon, lat, timestamp))
        
        return tid, oid, seg_id, points
    except (ValueError, IndexError):
        return None


def compute_bounding_box(points: List[Tuple[float, float, Any]]) -> Tuple[float, float, float, float]:
    """计算点集的边界框"""
    if not points:
        return 0.0, 0.0, 0.0, 0.0
    
    lons = [p[0] for p in points]
    lats = [p[1] for p in points]
    return min(lons), min(lats), max(lons), max(lats)


def parse_timestamp(ts_str: str) -> Optional[datetime]:
    """尝试解析时间戳字符串"""
    if not ts_str or ts_str == "0":
        return None
    
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y/%m/%d %H:%M:%S",
        "%d/%m/%Y %H:%M:%S",
        "%H:%M:%S",
        "%Y%m%d%H%M%S",
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(ts_str, fmt)
        except ValueError:
            continue
    
    # 尝试解析Unix时间戳
    try:
        ts_num = float(ts_str)
        if ts_num > 1e9:  # 秒级时间戳
            return datetime.fromtimestamp(ts_num)
    except ValueError:
        pass
    
    return None


def analyze_trajectory_dataset(
    data_path: str,
    max_trajectories: Optional[int] = None
) -> Dict[str, Any]:
    """
    分析轨迹数据集
    
    返回包含以下信息的字典:
    - 基本信息: 轨迹数量、点数量
    - 时间信息: 时间范围、平均采样间隔、轨迹持续时间分布
    - 空间信息: 边界框、面积覆盖、各轨迹MBR统计
    - 分布信息: 轨迹长度分布、点密度分布
    """
    print(f"[开始分析] 数据文件: {data_path}")
    
    # 初始化统计容器
    stats = {
        "basic": {
            "total_trajectories": 0,
            "total_points": 0,
            "valid_timestamps": 0,
        },
        "spatial": {
            "global_bbox": {"min_lon": float('inf'), "min_lat": float('inf'), 
                           "max_lon": float('-inf'), "max_lat": float('-inf')},
            "trajectory_mbrs": [],  # 每条轨迹的MBR
        },
        "temporal": {
            "has_timestamp": False,
            "time_range": {"start": None, "end": None},
            "durations": [],  # 轨迹持续时间（秒）
            "intervals": [],  # 采样间隔（秒）
        },
        "distribution": {
            "traj_lengths": [],  # 轨迹点数量分布
            "traj_distances": [],  # 轨迹长度（欧氏距离）分布
            "traj_areas": [],  # MBR面积分布
        }
    }
    
    # 读取数据
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据文件不存在: {data_path}")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if max_trajectories and stats["basic"]["total_trajectories"] >= max_trajectories:
                break
            
            result = parse_trajectory_line(line)
            if result is None:
                continue
            
            tid, oid, seg_id, points = result
            if len(points) < 2:
                continue
            
            stats["basic"]["total_trajectories"] += 1
            stats["basic"]["total_points"] += len(points)
            stats["distribution"]["traj_lengths"].append(len(points))
            
            # 空间分析
            min_lon, min_lat, max_lon, max_lat = compute_bounding_box(points)
            stats["spatial"]["trajectory_mbrs"].append({
                "min_lon": min_lon, "min_lat": min_lat,
                "max_lon": max_lon, "max_lat": max_lat,
                "width": max_lon - min_lon,
                "height": max_lat - min_lat,
                "area": (max_lon - min_lon) * (max_lat - min_lat)
            })
            
            # 更新全局边界
            gb = stats["spatial"]["global_bbox"]
            gb["min_lon"] = min(gb["min_lon"], min_lon)
            gb["min_lat"] = min(gb["min_lat"], min_lat)
            gb["max_lon"] = max(gb["max_lon"], max_lon)
            gb["max_lat"] = max(gb["max_lat"], max_lat)
            
            # 时间分析
            timestamps = [parse_timestamp(p[2]) for p in points if p[2]]
            timestamps = [t for t in timestamps if t is not None]
            
            if timestamps:
                stats["temporal"]["has_timestamp"] = True
                stats["basic"]["valid_timestamps"] += len(timestamps)
                
                min_ts, max_ts = min(timestamps), max(timestamps)
                tr = stats["temporal"]["time_range"]
                if tr["start"] is None or min_ts < tr["start"]:
                    tr["start"] = min_ts
                if tr["end"] is None or max_ts > tr["end"]:
                    tr["end"] = max_ts
                
                # 轨迹持续时间
                duration = (max_ts - min_ts).total_seconds()
                stats["temporal"]["durations"].append(duration)
                
                # 采样间隔
                if len(timestamps) > 1:
                    intervals = [(timestamps[i+1] - timestamps[i]).total_seconds() 
                              for i in range(len(timestamps)-1)]
                    stats["temporal"]["intervals"].extend(intervals)
            
            # 计算轨迹欧氏距离
            distance = sum(
                math.sqrt((points[i+1][0] - points[i][0])**2 + 
                         (points[i+1][1] - points[i][1])**2)
                for i in range(len(points)-1)
            )
            stats["distribution"]["traj_distances"].append(distance)
            stats["distribution"]["traj_areas"].append(
                (max_lon - min_lon) * (max_lat - min_lat)
            )
    
    # 后处理统计计算
    _compute_derived_stats(stats)
    
    return stats


def _compute_derived_stats(stats: Dict[str, Any]):
    """计算派生统计量"""
    # 空间统计
    gb = stats["spatial"]["global_bbox"]
    if gb["min_lon"] != float('inf'):
        stats["spatial"]["global_width"] = gb["max_lon"] - gb["min_lon"]
        stats["spatial"]["global_height"] = gb["max_lat"] - gb["min_lat"]
        stats["spatial"]["global_area"] = stats["spatial"]["global_width"] * stats["spatial"]["global_height"]
    else:
        stats["spatial"]["global_width"] = 0
        stats["spatial"]["global_height"] = 0
        stats["spatial"]["global_area"] = 0
    
    # MBR统计
    if stats["spatial"]["trajectory_mbrs"]:
        widths = [m["width"] for m in stats["spatial"]["trajectory_mbrs"]]
        heights = [m["height"] for m in stats["spatial"]["trajectory_mbrs"]]
        areas = [m["area"] for m in stats["spatial"]["trajectory_mbrs"]]
        
        stats["spatial"]["avg_mbr_width"] = np.mean(widths)
        stats["spatial"]["avg_mbr_height"] = np.mean(heights)
        stats["spatial"]["avg_mbr_area"] = np.mean(areas)
        stats["spatial"]["median_mbr_area"] = np.median(areas)
    
    # 时间统计
    if stats["temporal"]["durations"]:
        stats["temporal"]["avg_duration"] = np.mean(stats["temporal"]["durations"])
        stats["temporal"]["median_duration"] = np.median(stats["temporal"]["durations"])
        stats["temporal"]["max_duration"] = max(stats["temporal"]["durations"])
    
    if stats["temporal"]["intervals"]:
        stats["temporal"]["avg_interval"] = np.mean(stats["temporal"]["intervals"])
        stats["temporal"]["median_interval"] = np.median(stats["temporal"]["intervals"])
    
    # 分布统计
    if stats["distribution"]["traj_lengths"]:
        lengths = stats["distribution"]["traj_lengths"]
        stats["distribution"]["avg_length"] = np.mean(lengths)
        stats["distribution"]["median_length"] = np.median(lengths)
        stats["distribution"]["min_length"] = min(lengths)
        stats["distribution"]["max_length"] = max(lengths)
        
        # 分段统计
        bins = [0, 10, 20, 50, 100, 200, 500, float('inf')]
        labels = ["0-10", "10-20", "20-50", "50-100", "100-200", "200-500", "500+"]
        length_dist = defaultdict(int)
        for l in lengths:
            for i, (b1, b2) in enumerate(zip(bins[:-1], bins[1:])):
                if b1 <= l < b2:
                    length_dist[labels[i]] += 1
                    break
        stats["distribution"]["length_distribution"] = dict(length_dist)
    
    if stats["distribution"]["traj_distances"]:
        dists = stats["distribution"]["traj_distances"]
        stats["distribution"]["avg_distance"] = np.mean(dists)
        stats["distribution"]["median_distance"] = np.median(dists)


def print_analysis_report(stats: Dict[str, Any]):
    """打印分析报告"""
    print("\n" + "=" * 70)
    print(" " * 20 + "轨迹数据集分析报告")
    print("=" * 70)
    
    # 基本信息
    print("\n📊 [基本信息]")
    basic = stats["basic"]
    print(f"  轨迹总数:     {basic['total_trajectories']:,}")
    print(f"  点总数:       {basic['total_points']:,}")
    print(f"  平均每轨迹点数: {basic['total_points']/basic['total_trajectories']:.2f}" if basic['total_trajectories'] > 0 else "  N/A")
    if basic['valid_timestamps'] > 0:
        print(f"  有效时间戳:   {basic['valid_timestamps']:,}")
    
    # 空间信息
    print("\n🗺️  [空间边界]")
    gb = stats["spatial"]["global_bbox"]
    if gb["min_lon"] != float('inf'):
        print(f"  经度范围:     [{gb['min_lon']:.6f}, {gb['max_lon']:.6f}]")
        print(f"  纬度范围:     [{gb['min_lat']:.6f}, {gb['max_lat']:.6f}]")
        print(f"  覆盖区域:     {stats['spatial']['global_width']:.4f}° × {stats['spatial']['global_height']:.4f}°")
        print(f"  覆盖面积:     {stats['spatial']['global_area']:.6f} (度²)")
    
    if "avg_mbr_width" in stats["spatial"]:
        print("\n📐 [轨迹MBR统计]")
        print(f"  平均MBR宽度:  {stats['spatial']['avg_mbr_width']:.6f}°")
        print(f"  平均MBR高度:  {stats['spatial']['avg_mbr_height']:.6f}°")
        print(f"  平均MBR面积:  {stats['spatial']['avg_mbr_area']:.8f} (度²)")
        print(f"  中位数MBR面积: {stats['spatial']['median_mbr_area']:.8f} (度²)")
    
    # 时间信息
    if stats["temporal"]["has_timestamp"]:
        print("\n⏱️  [时间信息]")
        tr = stats["temporal"]["time_range"]
        if tr["start"] and tr["end"]:
            print(f"  时间范围:     {tr['start']} 至 {tr['end']}")
            total_span = (tr["end"] - tr["start"]).total_seconds()
            print(f"  总时间跨度:   {total_span/3600:.2f} 小时 ({total_span/86400:.2f} 天)")
        
        if "avg_duration" in stats["temporal"]:
            avg_dur = stats["temporal"]["avg_duration"]
            med_dur = stats["temporal"]["median_duration"]
            max_dur = stats["temporal"]["max_duration"]
            print(f"  平均轨迹时长: {avg_dur/60:.2f} 分钟")
            print(f"  中位数时长:   {med_dur/60:.2f} 分钟")
            print(f"  最大时长:     {max_dur/3600:.2f} 小时")
        
        if "avg_interval" in stats["temporal"]:
            avg_int = stats["temporal"]["avg_interval"]
            med_int = stats["temporal"]["median_interval"]
            print(f"  平均采样间隔: {avg_int:.2f} 秒 ({avg_int/60:.2f} 分钟)")
            print(f"  中位数间隔:   {med_int:.2f} 秒 ({med_int/60:.2f} 分钟)")
    
    # 分布信息
    print("\n📈 [轨迹点数量分布]")
    dist = stats["distribution"]
    if "length_distribution" in dist:
        for label, count in dist["length_distribution"].items():
            pct = count / basic['total_trajectories'] * 100
            bar = "█" * int(pct / 2)
            print(f"  {label:>6}点: {count:>6,} ({pct:>5.1f}%) {bar}")
    
    if "avg_distance" in dist:
        print(f"\n📏 [轨迹距离统计]")
        print(f"  平均轨迹长度: {dist['avg_distance']:.4f} 度")
        print(f"  中位数长度:   {dist['median_distance']:.4f} 度")
    
    print("\n" + "=" * 70)


def save_analysis_json(stats: Dict[str, Any], output_path: str):
    """保存分析结果为JSON"""
    # 转换datetime对象为字符串
    def convert_datetime(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, dict):
            return {k: convert_datetime(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_datetime(i) for i in obj]
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return obj
    
    stats_serializable = convert_datetime(stats)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(stats_serializable, f, indent=2, ensure_ascii=False)
    print(f"\n💾 详细分析结果已保存: {output_path}")


def plot_analysis_charts(stats: Dict[str, Any], output_dir: str, dataset_name: str = "dataset"):
    """绘制分析图表"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 设置样式
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 创建多个子图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Trajectory Dataset Analysis: {dataset_name}', fontsize=16, fontweight='bold')
    
    # 1. 轨迹长度分布
    ax1 = axes[0, 0]
    lengths = stats["distribution"]["traj_lengths"]
    if lengths:
        ax1.hist(lengths, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Number of Points per Trajectory')
        ax1.set_ylabel('Count')
        ax1.set_title('Trajectory Length Distribution')
        ax1.axvline(np.mean(lengths), color='red', linestyle='--', label=f'Mean: {np.mean(lengths):.1f}')
        ax1.legend()
    
    # 2. MBR面积分布
    ax2 = axes[0, 1]
    areas = stats["distribution"]["traj_areas"]
    if areas and any(a > 0 for a in areas):
        # 过滤掉0值，使用对数刻度
        non_zero_areas = [a for a in areas if a > 0]
        ax2.hist(non_zero_areas, bins=50, color='coral', edgecolor='black', alpha=0.7)
        ax2.set_xlabel('MBR Area (degree²)')
        ax2.set_ylabel('Count')
        ax2.set_title('Trajectory MBR Area Distribution')
        ax2.set_yscale('log')
    
    # 3. 采样间隔分布（如果有时间数据）
    ax3 = axes[1, 0]
    intervals = stats["temporal"]["intervals"]
    if intervals:
        # 过滤异常值（超过1小时的间隔）
        filtered_intervals = [i for i in intervals if 0 < i < 3600]
        if filtered_intervals:
            ax3.hist(filtered_intervals, bins=50, color='green', edgecolor='black', alpha=0.7)
            ax3.set_xlabel('Sampling Interval (seconds)')
            ax3.set_ylabel('Count')
            ax3.set_title('Sampling Interval Distribution')
    else:
        ax3.text(0.5, 0.5, 'No Timestamp Data', ha='center', va='center', transform=ax3.transAxes)
    
    # 4. 轨迹持续时间分布
    ax4 = axes[1, 1]
    durations = stats["temporal"]["durations"]
    if durations:
        # 转换为分钟
        durations_min = [d/60 for d in durations if d > 0]
        if durations_min:
            ax4.hist(durations_min, bins=50, color='purple', edgecolor='black', alpha=0.7)
            ax4.set_xlabel('Trajectory Duration (minutes)')
            ax4.set_ylabel('Count')
            ax4.set_title('Trajectory Duration Distribution')
    else:
        ax4.text(0.5, 0.5, 'No Timestamp Data', ha='center', va='center', transform=ax4.transAxes)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    chart_path = output_path / f"analysis_{dataset_name}.png"
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 分析图表已保存: {chart_path}")


def main():
    parser = argparse.ArgumentParser(description="轨迹数据集分析工具")
    parser.add_argument("--data-path", type=str, required=True, 
                       help="轨迹数据文件路径 (格式: tid|oid|seg_id|lon,lat;...)")
    parser.add_argument("--output-dir", type=str, default="scripts/analyze/analysis_output",
                       help="分析结果输出目录")
    parser.add_argument("--dataset-name", type=str, default="dataset",
                       help="数据集名称（用于输出文件名）")
    parser.add_argument("--max-trajectories", type=int, default=None,
                       help="最大处理轨迹数（用于快速测试）")
    parser.add_argument("--save-json", action="store_true",
                       help="保存详细结果为JSON文件")
    parser.add_argument("--plot", action="store_true",
                       help="生成分析图表")
    
    args = parser.parse_args()
    
    # 执行分析
    stats = analyze_trajectory_dataset(args.data_path, max_trajectories=args.max_trajectories)
    
    # 打印报告
    print_analysis_report(stats)
    
    # 保存JSON
    if args.save_json:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / f"analysis_{args.dataset_name}.json"
        save_analysis_json(stats, str(json_path))
    
    # 绘制图表
    if args.plot:
        plot_analysis_charts(stats, args.output_dir, args.dataset_name)
    
    print("\n✅ 分析完成！")


if __name__ == "__main__":
    main()

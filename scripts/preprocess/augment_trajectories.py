"""
轨迹数据增强脚本

基于清洗后的轨迹数据(txt格式)生成增强轨迹，仅输出WKT格式
用于分布式环境测试性能

输入格式(清洗后的txt): tid|oid|seg_id|经度,纬度;经度,纬度;...
输出格式(wkt): tid,oid,seg_id,LINESTRING(lon lat, lon lat, ...)

增强方法:
1. 随机扰动: 对轨迹点添加随机偏移
2. 高斯扰动: 添加相关的高斯噪声保持轨迹形状
3. 插值细化: 在点之间插入新的插值点
4. 旋转: 围绕MBR中心旋转轨迹
5. 缩放: 围绕MBR中心缩放轨迹
6. 混合: 混合两条轨迹的特征
"""
import argparse
import copy
import math
import os
import random
from typing import List, Optional, Tuple

import numpy as np


def parse_trajectory_line(line: str) -> Optional[Tuple[int, str, int, List[Tuple[float, float]]]]:
    """
    解析清洗后的txt格式轨迹行数据
    
    输入格式: tid|oid|seg_id|lon,lat;lon,lat...
    返回: (tid, oid, seg_id, [(lon, lat), ...])
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
                points.append((lon, lat))
        
        return tid, oid, seg_id, points
    except (ValueError, IndexError):
        return None


def format_wkt_line(tid: int, oid: str, seg_id: int,
                    points: List[Tuple[float, float]]) -> str:
    """
    格式化为WKT输出格式
    
    格式: tid,oid,seg_id,LINESTRING(lon lat, lon lat, ...)
    """
    wkt_points = ", ".join([f"{p[0]} {p[1]}" for p in points])
    return f"{tid},{oid},{seg_id},LINESTRING({wkt_points})\n"


def add_random_perturbation(
    points: List[Tuple[float, float]],
    noise_scale: float = 0.0001
) -> List[Tuple[float, float]]:
    """
    对轨迹点添加随机扰动
    
    Args:
        points: 原始轨迹点 [(lon, lat), ...]
        noise_scale: 扰动幅度（度），默认约10米
    
    Returns:
        扰动后的轨迹点
    """
    perturbed = []
    for lon, lat in points:
        new_lon = lon + random.gauss(0, noise_scale)
        new_lat = lat + random.gauss(0, noise_scale)
        perturbed.append((new_lon, new_lat))
    
    return perturbed


def add_gaussian_perturbation(
    points: List[Tuple[float, float]],
    mbr_scale: float = 0.001,
    correlation: float = 0.8
) -> List[Tuple[float, float]]:
    """
    添加相关的高斯扰动（保持轨迹形状）
    
    Args:
        points: 原始轨迹点 [(lon, lat), ...]
        mbr_scale: 扰动相对于MBR的比例
        correlation: 相邻点间的扰动相关性
    
    Returns:
        扰动后的轨迹点
    """
    if not points:
        return points
    
    # 计算MBR
    lons = [p[0] for p in points]
    lats = [p[1] for p in points]
    width = max(lons) - min(lons)
    height = max(lats) - min(lats)
    
    noise_scale_lon = width * mbr_scale
    noise_scale_lat = height * mbr_scale
    
    # 生成相关噪声
    perturbed = []
    prev_noise_lon, prev_noise_lat = 0.0, 0.0
    
    for lon, lat in points:
        # 当前噪声是前一个噪声的相关版本 + 新噪声
        noise_lon = correlation * prev_noise_lon + (1 - correlation) * random.gauss(0, noise_scale_lon)
        noise_lat = correlation * prev_noise_lat + (1 - correlation) * random.gauss(0, noise_scale_lat)
        
        new_lon = lon + noise_lon
        new_lat = lat + noise_lat
        perturbed.append((new_lon, new_lat))
        
        prev_noise_lon, prev_noise_lat = noise_lon, noise_lat
    
    return perturbed


def interpolate_trajectory(
    points: List[Tuple[float, float]],
    factor: int = 2
) -> List[Tuple[float, float]]:
    """
    对轨迹进行插值细化
    
    Args:
        points: 原始轨迹点
        factor: 插值倍数，每段插入 factor-1 个点
    
    Returns:
        插值后的轨迹点
    """
    if len(points) < 2 or factor < 1:
        return points
    
    interpolated = [points[0]]
    
    for i in range(len(points) - 1):
        lon1, lat1 = points[i]
        lon2, lat2 = points[i + 1]
        
        for j in range(1, factor):
            t = j / factor
            # 线性插值
            new_lon = lon1 + t * (lon2 - lon1)
            new_lat = lat1 + t * (lat2 - lat1)
            interpolated.append((new_lon, new_lat))
        
        interpolated.append(points[i + 1])
    
    return interpolated


def rotate_trajectory(
    points: List[Tuple[float, float]],
    angle_deg: float = None
) -> List[Tuple[float, float]]:
    """
    旋转轨迹（围绕MBR中心）
    
    Args:
        points: 原始轨迹点
        angle_deg: 旋转角度（度），随机选择如果为None
    
    Returns:
        旋转后的轨迹点
    """
    if not points:
        return points
    
    if angle_deg is None:
        angle_deg = random.uniform(-30, 30)
    
    angle_rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    
    # 计算中心点
    lons = [p[0] for p in points]
    lats = [p[1] for p in points]
    cx = (min(lons) + max(lons)) / 2
    cy = (min(lats) + max(lats)) / 2
    
    rotated = []
    for lon, lat in points:
        # 平移到原点
        x, y = lon - cx, lat - cy
        # 旋转
        new_x = x * cos_a - y * sin_a
        new_y = x * sin_a + y * cos_a
        # 平移回去
        new_lon = new_x + cx
        new_lat = new_y + cy
        rotated.append((new_lon, new_lat))
    
    return rotated


def scale_trajectory(
    points: List[Tuple[float, float]],
    scale_range: Tuple[float, float] = (0.8, 1.2),
) -> List[Tuple[float, float]]:
    """
    缩放轨迹（围绕MBR中心），若缩放后点数不足则插值补充
    
    Args:
        points: 原始轨迹点
        scale_range: 缩放比例范围
    
    Returns:
        缩放后的轨迹点
    """
    if not points:
        return points
    
    scale = random.uniform(*scale_range)
    
    # 计算中心点
    lons = [p[0] for p in points]
    lats = [p[1] for p in points]
    cx = (min(lons) + max(lons)) / 2
    cy = (min(lats) + max(lats)) / 2
    
    scaled = []
    for lon, lat in points:
        new_lon = cx + (lon - cx) * scale
        new_lat = cy + (lat - cy) * scale
        scaled.append((new_lon, new_lat))
    
    # 缩放后若点数不足，进行插值补充
    if len(scaled) < 10:
        factor = (10 + len(scaled) - 1) // len(scaled)
        scaled = interpolate_trajectory(scaled, factor=factor)
    
    return scaled


def mix_trajectories(
    points1: List[Tuple[float, float]],
    points2: List[Tuple[float, float]],
    mix_ratio: float = None
) -> List[Tuple[float, float]]:
    """
    温和地混合两条轨迹的特征
    
    使用较小的混合比例(0.05-0.15)，保持主轨迹形状的同时
    引入次轨迹的局部特征变化
    
    Args:
        points1: 主轨迹(保持主体形状)
        points2: 次轨迹(提供局部扰动)
        mix_ratio: 混合比例，None时随机选择0.05-0.15
    
    Returns:
        混合后的轨迹点
    """
    if not points1 or not points2:
        return points1 if points1 else points2
    
    if mix_ratio is None:
        mix_ratio = random.uniform(0.05, 0.15)  # 更温和的比例
    
    # 归一化到相同长度（使用较短的，避免过度拉伸）
    min_len = min(len(points1), len(points2))
    
    mixed = []
    for i in range(min_len):
        lon1, lat1 = points1[i]
        lon2, lat2 = points2[i]
        # 主轨迹占主导，次轨迹提供小幅扰动
        new_lon = lon1 + mix_ratio * (lon2 - lon1)
        new_lat = lat1 + mix_ratio * (lat2 - lat1)
        mixed.append((new_lon, new_lat))
    
    # 保留主轨迹的剩余点（保持原始形状）
    if len(points1) > min_len:
        mixed.extend(points1[min_len:])
    
    return mixed


def generate_augmented_trajectory(
    source_trajectories: List[List[Tuple[float, float]]],
    augmentation_methods: List[str] = None,
    use_all_methods: bool = False,
    seed: Optional[int] = None
) -> List[Tuple[float, float]]:
    """
    生成增强后的轨迹
    
    Args:
        source_trajectories: 源轨迹列表，每项为[(lon, lat), ...]
        augmentation_methods: 增强方法列表
        use_all_methods: 是否为'all'模式（随机子集应用）
        seed: 随机种子
    
    Returns:
        增强后的轨迹点 [(lon, lat), ...]
    """
    if seed is not None:
        random.seed(seed)
    
    all_available_methods = ['perturb', 'gaussian', 'interpolate', 'rotate', 'scale', 'mix']
    
    if augmentation_methods is None:
        augmentation_methods = ['gaussian', 'rotate', 'scale']
    
    # 确定可用的方法集合
    if use_all_methods or set(augmentation_methods) == set(all_available_methods):
        methods_available = all_available_methods
    else:
        methods_available = augmentation_methods
    
    # 方法概率权重配置（perturb ≈ interpolate ≈ gaussian > rotate > mix > scale）
    method_weights = {
        'perturb': 0.20,
        'interpolate': 0.20,
        'gaussian': 0.20,
        'rotate': 0.18,
        'mix': 0.12,
        'scale': 0.10
    }
    
    # 计算可用方法的累积概率区间
    available_weights = []
    available_methods = []
    for m in methods_available:
        if m in method_weights:
            available_methods.append(m)
            available_weights.append(method_weights[m])
    
    # 归一化概率
    total_weight = sum(available_weights)
    if total_weight > 0:
        available_weights = [w / total_weight for w in available_weights]
    
    # 构建累积概率区间
    cumulative_probs = []
    cumsum = 0.0
    for w in available_weights:
        cumsum += w
        cumulative_probs.append(cumsum)
    
    # 随机选择一条源轨迹
    base_points = random.choice(source_trajectories)
    points = copy.deepcopy(base_points)
    
    # 生成随机数，根据区间选择一种方法
    r = random.random()
    selected_method = None
    for i, cp in enumerate(cumulative_probs):
        if r < cp:
            selected_method = available_methods[i]
            break
    
    if selected_method is None and available_methods:
        selected_method = available_methods[-1]
    
    # 应用选中的方法
    if selected_method == 'perturb':
        points = add_random_perturbation(points, noise_scale=0.0001)
    elif selected_method == 'gaussian':
        points = add_gaussian_perturbation(points, mbr_scale=0.02, correlation=0.9)
    elif selected_method == 'interpolate':
        points = interpolate_trajectory(points, factor=random.choice([2, 3]))
    elif selected_method == 'rotate':
        points = rotate_trajectory(points, angle_deg=random.uniform(-20, 20))
    elif selected_method == 'scale':
        points = scale_trajectory(points, scale_range=(0.85, 1.15))
    elif selected_method == 'mix' and len(source_trajectories) > 1:
        other_points = random.choice([p for p in source_trajectories if p != base_points])
        points = mix_trajectories(points, other_points)
    
    return points


def augment_dataset(
    input_file: str,
    output_file: str,
    multiplier: float,
    augmentation_methods: List[str] = None,
    use_all_methods: bool = False,
    seed: int = 42
):
    """
    增强数据集，输出WKT格式
    
    Args:
        input_file: 输入文件路径（清洗后的txt格式）
        output_file: 输出文件路径（wkt格式）
        multiplier: 增强倍数（支持小数）
        augmentation_methods: 增强方法列表
        use_all_methods: 是否为'all'模式（随机选择子集应用）
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    
    print(f"[开始数据增强]")
    print(f"  输入文件: {input_file}")
    print(f"  输出文件: {output_file}")
    print(f"  输出格式: WKT")
    print(f"  增强倍数: {multiplier}")
    if use_all_methods:
        print(f"  增强方法: all (按概率应用: perturb/interpolate/gaussian:80% > rotate:60% > mix:40% > scale:30%)")
    else:
        print(f"  增强方法: {augmentation_methods}")
    
    # 读取源轨迹（清洗后的txt格式）
    source_trajectories = []
    original_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            result = parse_trajectory_line(line)
            if result is None:
                continue
            
            tid, oid, seg_id, points = result
            source_trajectories.append({
                'tid': tid,
                'oid': oid,
                'seg_id': seg_id,
                'points': points
            })
            original_count += 1
    
    print(f"  加载源轨迹: {original_count} 条")
    
    if original_count == 0:
        print("错误: 未找到有效轨迹！")
        return
    
    # 计算需要生成的轨迹数量
    total_needed = int(original_count * multiplier)
    augmented_needed = total_needed - original_count
    
    print(f"  目标总数: {total_needed} 条")
    print(f"  需要生成: {max(0, augmented_needed)} 条")
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 写入增强后的数据（WKT格式）
    with open(output_file, 'w', encoding='utf-8') as f_out:
        # 首先写入所有原始轨迹（转换为WKT格式）
        for traj in source_trajectories:
            f_out.write(format_wkt_line(
                traj['tid'], traj['oid'], traj['seg_id'], traj['points']
            ))
        
        # 生成增强轨迹
        points_list = [t['points'] for t in source_trajectories]
        
        for i in range(augmented_needed):
            new_points = generate_augmented_trajectory(
                points_list,
                augmentation_methods=augmentation_methods,
                use_all_methods=use_all_methods,
                seed=seed + i
            )
            
            # 分配新的TID
            new_tid = original_count + i + 1
            # 使用随机源OID并标记为增强数据
            source_oid = source_trajectories[i % original_count]['oid']
            new_oid = f"{source_oid}_aug"
            
            f_out.write(format_wkt_line(new_tid, new_oid, 1, new_points))
            
            if (i + 1) % 1000 == 0:
                print(f"  已生成 {i + 1}/{augmented_needed} 条增强轨迹...")
    
    print(f"\n✅ 数据增强完成！")
    print(f"  原始轨迹: {original_count} 条")
    print(f"  增强轨迹: {max(0, augmented_needed)} 条")
    print(f"  总计: {total_needed} 条")
    print(f"  输出文件: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="轨迹数据增强工具 - 基于清洗后txt格式数据生成增强轨迹，输出WKT格式"
    )
    parser.add_argument("--input", type=str, required=True,
                       help="输入的轨迹数据文件路径（清洗后的txt格式: tid|oid|seg_id|lon,lat;...）")
    parser.add_argument("--output", type=str, required=True,
                       help="输出的增强数据文件路径（WKT格式）")
    parser.add_argument("--multiplier", type=float, required=True,
                       help="增强倍数（支持小数，如2.5表示生成2.5倍于原始数据的轨迹）")
    parser.add_argument("--methods", type=str, nargs='+',
                       default=['gaussian', 'rotate', 'scale'],
                       choices=['perturb', 'gaussian', 'interpolate', 'rotate', 'scale', 'mix', 'all'],
                       help="增强方法列表 (默认: gaussian rotate scale)")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子 (默认: 42)")
    
    args = parser.parse_args()
    
    # 验证参数
    if args.multiplier <= 1:
        print("错误: 增强倍数必须大于1")
        return 1
    
    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在: {args.input}")
        return 1
    
    # 处理 'all' 方法
    methods = args.methods
    use_all = False
    if 'all' in methods:
        use_all = True
        methods = ['perturb', 'gaussian', 'interpolate', 'rotate', 'scale', 'mix']
    
    # 执行增强
    augment_dataset(
        input_file=args.input,
        output_file=args.output,
        multiplier=args.multiplier,
        augmentation_methods=methods,
        use_all_methods=use_all,
        seed=args.seed
    )
    
    return 0


if __name__ == "__main__":
    exit(main())

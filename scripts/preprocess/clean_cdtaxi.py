"""
CD-Taxi 数据集清洗脚本

输入格式: [tid,[["时间",经度,纬度],...]]
示例: [7dceae818438b836e3d306296b4ccfbd,[["2018-09-30 19:15:38.0",104.04235,30.69204],["2018-09-30 19:18:23.0",104.04389,30.69443]]]

输出格式支持:
1. txt: tid|oid|seg_id|经度,纬度;经度,纬度;...
2. geojson: GeoJSON Feature 格式
3. wkt: tid,oid,seg_id,LINESTRING(lon lat, ...)

注意: 成都市经纬度范围（行政区划+误差容忍）
- 经度: 约 102.0°E - 105.5°E
- 纬度: 约 29.5°N - 32.0°N
"""
import json
import os
import re
from typing import List, Optional, Tuple

# 成都大致范围: 经度 102.5°E - 104.9°E, 纬度 30.0°N - 31.5°N
# 扩大范围以容忍边界误差和郊区数据
REASONABLE_MIN_LON, REASONABLE_MAX_LON = 102.0, 105.5  # 约±1°误差容忍
REASONABLE_MIN_LAT, REASONABLE_MAX_LAT = 29.5, 32.0    # 约±0.5°误差容忍
MIN_TRAJ_LENGTH = 10


def format_line(tid: int, obj_id: str, seg_idx: int, points: list, _fmt: str) -> str:
    """根据指定的格式对轨迹进行序列化（不保留时间戳）"""
    _fmt = _fmt.lower()

    if _fmt == 'txt':
        # 格式: 全局TID|对象ID|分段索引|lon,lat;lon,lat...
        points_str = ";".join([f"{p[0]},{p[1]}" for p in points])
        return f"{tid}|{obj_id}|{seg_idx}|{points_str}\n"

    elif _fmt == 'geojson':
        feature = {
            "type": "Feature",
            "properties": {
                "tid": tid,
                "oid": obj_id,
                "sid": seg_idx
            },
            "geometry": {
                "type": "LineString",
                "coordinates": [[p[0], p[1]] for p in points]
            }
        }
        return json.dumps(feature, ensure_ascii=False) + "\n"

    elif _fmt == 'wkt':
        wkt_points = ", ".join([f"{p[0]} {p[1]}" for p in points])
        return f"{tid},{obj_id},{seg_idx},LINESTRING({wkt_points})\n"

    raise ValueError(f"Unsupported format: {_fmt}")


def parse_cdtaxi_record(line: str) -> Optional[Tuple[str, List[Tuple[float, float]]]]:
    """
    解析 CD-Taxi 记录（不保留时间戳）
    
    输入格式: [tid,[["时间",经度,纬度],...]]
    返回: (tid, [(经度, 纬度), ...]) 或 None
    """
    line = line.strip()
    if not line:
        return None
    
    try:
        # 处理 JSON 数组格式
        # 格式: [tid, [[timestamp, lon, lat], ...]]
        
        # 首先尝试直接作为 JSON 解析
        try:
            data = json.loads(line)
            if isinstance(data, list) and len(data) >= 2:
                tid = str(data[0])
                points_data = data[1]
                
                points = []
                for p in points_data:
                    if isinstance(p, list) and len(p) >= 3:
                        # 只取经度和纬度，忽略时间戳
                        lon = float(p[1])
                        lat = float(p[2])
                        points.append((lon, lat))
                
                return tid, points
        except json.JSONDecodeError:
            pass
        
        # 如果 JSON 解析失败，尝试正则提取
        # 匹配 tid 部分（通常是 MD5 哈希）
        tid_match = re.match(r'^\s*\[\s*"([^"]+)"\s*,', line)
        if not tid_match:
            tid_match = re.match(r'^\s*\[\s*([^,\]]+)', line)
        
        if not tid_match:
            return None
        
        tid = tid_match.group(1).strip().strip('"')
        
        # 提取坐标数组部分
        coords_match = re.search(r'\[\s*\[.*?\]\s*\]', line)
        if not coords_match:
            return None
        
        coords_str = coords_match.group(0)
        
        # 解析坐标点（跳过时间戳）
        points = []
        # 匹配 ["timestamp", lon, lat] 或 [timestamp, lon, lat]
        point_pattern = r'\[\s*(?:"[^"]+"|[^,\]]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]'
        for match in re.finditer(point_pattern, coords_str):
            lon = float(match.group(1))
            lat = float(match.group(2))
            points.append((lon, lat))
        
        if len(points) < 2:
            return None
        
        return tid, points
    
    except (ValueError, IndexError, TypeError):
        return None


def format_output_line(tid: int, oid: str, seg_idx: int,
                     points: List[Tuple[float, float]], output_format: str) -> str:
    """根据指定格式输出轨迹"""
    return format_line(tid, oid, seg_idx, points, output_format)


def compute_global_mbr(input_file: str) -> Tuple[float, float, float, float]:
    """
    扫描文件，获取全局边界框（MBR）
    同时过滤掉在合理范围之外的漂移点
    """
    print("阶段 1: 正在扫描合法区域内的全局边界 (MBR)...")
    
    g_min_x, g_min_y = float('inf'), float('inf')
    g_max_x, g_max_y = float('-inf'), float('-inf')
    
    valid_points = 0
    filtered_points = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            result = parse_cdtaxi_record(line)
            if result is None:
                continue
            
            tid, points = result
            
            for lon, lat in points:
                # 检查是否在合理范围内
                if REASONABLE_MIN_LON <= lon <= REASONABLE_MAX_LON and \
                   REASONABLE_MIN_LAT <= lat <= REASONABLE_MAX_LAT:
                    g_min_x = min(g_min_x, lon)
                    g_min_y = min(g_min_y, lat)
                    g_max_x = max(g_max_x, lon)
                    g_max_y = max(g_max_y, lat)
                    valid_points += 1
                else:
                    filtered_points += 1
            
            if line_num % 10000 == 0:
                print(f"  已处理 {line_num} 行...")
    
    if g_min_x == float('inf'):
        print("警告: 未找到有效数据点，请检查坐标范围设置！")
        g_min_x = REASONABLE_MIN_LON
        g_min_y = REASONABLE_MIN_LAT
        g_max_x = REASONABLE_MAX_LON
        g_max_y = REASONABLE_MAX_LAT
    
    print(f"扫描完毕！")
    print(f"  有效点数: {valid_points}")
    print(f"  过滤点数: {filtered_points}")
    print(f"过滤漂移点后的精确 MBR 为:")
    print(f"  X (Lon): [{g_min_x:.6f}, {g_max_x:.6f}]")
    print(f"  Y (Lat): [{g_min_y:.6f}, {g_max_y:.6f}]")
    print("-" * 30)
    
    return g_min_x, g_min_y, g_max_x, g_max_y


def clean_and_save_dataset(
    input_file: str,
    output_file: str,
    g_mbr: Tuple[float, float, float, float],
    output_format: str = 'txt',
    min_traj_length: int = MIN_TRAJ_LENGTH
):
    """
    根据扫描到的 MBR 过滤并保存轨迹
    """
    g_min_x, g_min_y, g_max_x, g_max_y = g_mbr
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    total_count = 0
    valid_count = 0
    filtered_by_bounds = 0
    filtered_by_length = 0
    filtered_by_degenerate = 0
    
    global_tid = 1
    
    print(f"阶段 2: 正在写入清洗后的目标文件 {output_file} (格式: {output_format})...")
    
    with open(output_file, 'w', encoding='utf-8') as f_out:
        with open(input_file, 'r', encoding='utf-8') as f_in:
            for line_num, line in enumerate(f_in, 1):
                total_count += 1
                
                result = parse_cdtaxi_record(line)
                if result is None:
                    continue
                
                tid, points = result
                
                # 过滤包含漂移点的轨迹
                temp_points = []
                is_dirty = False
                
                for lon, lat in points:
                    if not (g_min_x <= lon <= g_max_x and g_min_y <= lat <= g_max_y):
                        is_dirty = True
                        break
                    temp_points.append((lon, lat))
                
                if is_dirty:
                    filtered_by_bounds += 1
                    continue
                
                # 检查轨迹长度
                if len(temp_points) < min_traj_length:
                    filtered_by_length += 1
                    continue
                
                # 检查轨迹是否为退化轨迹（一个点或一条直线）
                lons = [p[0] for p in temp_points]
                lats = [p[1] for p in temp_points]
                
                if max(lons) <= min(lons) or max(lats) <= min(lats):
                    filtered_by_degenerate += 1
                    continue
                
                # 输出清洗后的轨迹
                f_out.write(format_output_line(global_tid, tid, 1, temp_points, output_format))
                global_tid += 1
                valid_count += 1
                
                if line_num % 10000 == 0:
                    print(f"  已处理 {line_num} 行，有效轨迹 {valid_count} 条...")
    
    print(f"\n清洗完成！")
    print(f"  原始轨迹总数: {total_count}")
    print(f"  清洗后有效轨迹: {valid_count}")
    print(f"  边界过滤: {filtered_by_bounds}")
    print(f"  长度过滤: {filtered_by_length}")
    print(f"  退化轨迹过滤: {filtered_by_degenerate}")
    print(f"  剔除比例: {(total_count - valid_count) / total_count:.2%}" if total_count > 0 else "  N/A")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="CD-Taxi 数据集清洗工具")
    parser.add_argument("--input", type=str, required=True,
                       help="输入的 CD-Taxi 原始数据文件路径")
    parser.add_argument("--output", type=str, required=True,
                       help="输出的清洗后数据文件路径")
    parser.add_argument("--format", type=str, default='txt',
                       choices=['txt', 'geojson', 'wkt'],
                       help="输出格式 (默认: txt)")
    parser.add_argument("--min-lon", type=float, default=REASONABLE_MIN_LON,
                       help=f"最小经度边界（默认: {REASONABLE_MIN_LON}）")
    parser.add_argument("--max-lon", type=float, default=REASONABLE_MAX_LON,
                       help=f"最大经度边界（默认: {REASONABLE_MAX_LON}）")
    parser.add_argument("--min-lat", type=float, default=REASONABLE_MIN_LAT,
                       help=f"最小纬度边界（默认: {REASONABLE_MIN_LAT}）")
    parser.add_argument("--max-lat", type=float, default=REASONABLE_MAX_LAT,
                       help=f"最大纬度边界（默认: {REASONABLE_MAX_LAT}）")
    parser.add_argument("--min-traj-length", type=int, default=MIN_TRAJ_LENGTH,
                       help=f"轨迹最小长度（默认: {MIN_TRAJ_LENGTH}）")
    parser.add_argument("--skip-mbr-scan", action="store_true",
                       help="跳过 MBR 扫描，使用默认边界")
    
    args = parser.parse_args()
    
    # 更新局部边界变量
    min_lon = args.min_lon
    max_lon = args.max_lon
    min_lat = args.min_lat
    max_lat = args.max_lat
    
    print(f"CD-Taxi 数据清洗配置:")
    print(f"  输入文件: {args.input}")
    print(f"  输出文件: {args.output}")
    print(f"  输出格式: {args.format}")
    print(f"  经度范围: [{min_lon}, {max_lon}]")
    print(f"  纬度范围: [{min_lat}, {max_lat}]")
    print(f"  最小轨迹长度: {args.min_traj_length}")
    print("-" * 50)
    
    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在: {args.input}")
        return 1
    
    # 阶段 1: 计算全局 MBR
    if args.skip_mbr_scan:
        mbr = (min_lon, min_lat, max_lon, max_lat)
        print("跳过 MBR 扫描，使用默认边界")
    else:
        mbr = compute_global_mbr(args.input)
    
    # 阶段 2: 清洗并保存
    clean_and_save_dataset(
        args.input, args.output, mbr,
        output_format=args.format,
        min_traj_length=args.min_traj_length
    )
    
    print("\n✅ 数据清洗完成！")
    return 0


if __name__ == "__main__":
    exit(main())

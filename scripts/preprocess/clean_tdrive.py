import json
import os
import re

REASONABLE_MIN_LON, REASONABLE_MAX_LON = 115.0, 118.0
REASONABLE_MIN_LAT, REASONABLE_MAX_LAT = 39.0, 41.5
MIN_TRAJ_LENGTH = 10


def format_line(tid: int, obj_id: str, seg_idx: int, points: list, _fmt: str) -> str:
    """根据指定的格式对轨迹进行序列化"""
    _fmt = _fmt.lower()

    if _fmt == 'origin':
        # 格式: 8275-8275_1-MULTIPOINT Z((lon lat z), (lon lat z)...)
        point_strings = []
        for p in points:
            z_val = p[2] if len(p) > 2 else 0
            point_strings.append(f"({p[0]} {p[1]} {z_val})")
        multipoint_str = ", ".join(point_strings)
        return f"{obj_id}-{obj_id}_{seg_idx}-MULTIPOINT Z({multipoint_str})\n"

    elif _fmt == 'txt':
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

    # 抛出异常或返回空字符串，防止返回 None ---
    raise ValueError(f"Unsupported format: {_fmt}")


def parse_identifier(line: str) -> tuple:
    """
    从行首提取原始标识符信息
    输入示例: 3644-3644_3-MULTIPOINT...
    返回: (object_id, segment_index) -> ("3644", 3)
    """
    # 匹配 ID-ID_Index 格式
    id_match = re.match(r"^(\d+)-(\d+)_(\d+)", line)
    if id_match:
        obj_id = id_match.group(1)
        seg_idx = int(id_match.group(3))
        return obj_id, seg_idx
    return "unknown", 0


def compute_global_mbr(input_dir: str):
    """阶段 1: 扫描并过滤漂移点，获取精确的北京区域 MBR"""
    multipoint_re = re.compile(r"MULTIPOINT Z\((.*?)\)$")
    point_re = re.compile(r"\(([^()]+)\)")
    files = sorted(name for name in os.listdir(input_dir) if name.startswith("part-"))

    print("阶段 1: 正在扫描合法区域内的全局边界 (MBR)...")
    g_min_x, g_min_y = float('inf'), float('inf')
    g_max_x, g_max_y = float('-inf'), float('-inf')

    for filename in files:
        with open(os.path.join(input_dir, filename), "r", encoding="utf-8") as f_in:
            for line in f_in:
                match = multipoint_re.search(line)
                if not match:
                    continue
                raw_points = point_re.findall(match.group(1))

                for rp in raw_points:
                    parts = rp.strip().split()
                    if len(parts) < 2:
                        continue
                    try:
                        lon, lat = float(parts[0]), float(parts[1])
                        # --- 剔除漂移点 ---
                        if REASONABLE_MIN_LON <= lon <= REASONABLE_MAX_LON and \
                                REASONABLE_MIN_LAT <= lat <= REASONABLE_MAX_LAT:
                            if lon < g_min_x:
                                g_min_x = lon
                            if lon > g_max_x:
                                g_max_x = lon
                            if lat < g_min_y:
                                g_min_y = lat
                            if lat > g_max_y:
                                g_max_y = lat
                    except ValueError:
                        continue

    print(f"扫描完毕！过滤漂移点后的精确 MBR 为:")
    print(f"  X (Lon): [{g_min_x}, {g_max_x}]")
    print(f"  Y (Lat): [{g_min_y}, {g_max_y}]")
    print("-" * 30)
    return g_min_x, g_min_y, g_max_x, g_max_y


def clean_and_save_dataset(input_dir: str, output_file: str, g_mbr: tuple, output_format: str = 'txt'):
    """阶段 2: 根据扫描到的 MBR 过滤并保存轨迹"""
    g_min_x, g_min_y, g_max_x, g_max_y = g_mbr

    multipoint_re = re.compile(r"MULTIPOINT Z\((.*?)\)$")
    point_re = re.compile(r"\(([^()]+)\)")
    files = sorted(name for name in os.listdir(input_dir) if name.startswith("part-"))

    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))

    total_count = 0
    valid_count = 0
    global_tid = 1

    print(f"阶段 2: 正在写入清洗后的目标文件 {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f_out:
        for filename in files:
            with open(os.path.join(input_dir, filename), "r", encoding="utf-8") as f_in:
                for line in f_in:
                    total_count += 1
                    obj_id, seg_idx = parse_identifier(line)
                    match = multipoint_re.search(line)
                    if not match:
                        continue
                    raw_points = point_re.findall(match.group(1))

                    temp_points = []
                    is_dirty = False
                    for rp in raw_points:
                        parts = rp.strip().split()
                        if len(parts) < 2:
                            continue
                        try:
                            lon, lat = float(parts[0]), float(parts[1])
                            # 提取时间戳
                            timestamp = parts[2] if len(parts) > 2 else "0"
                            # 剔除包含任何漂移点的轨迹
                            if not (g_min_x <= lon <= g_max_x and g_min_y <= lat <= g_max_y):
                                is_dirty = True
                                break
                            temp_points.append((lon, lat, timestamp))
                        except ValueError:
                            continue

                    if not is_dirty and len(temp_points) > MIN_TRAJ_LENGTH:
                        lons = [p[0] for p in temp_points]
                        lats = [p[1] for p in temp_points]
                        # MBR 过滤：确保轨迹不是一个点或一条直线（必须有面积/范围）
                        if max(lons) > min(lons) and max(lats) > min(lats):
                            f_out.write(format_line(global_tid, obj_id, seg_idx, temp_points, output_format))
                            global_tid += 1
                            valid_count += 1

    print(f"清洗完成！")
    print(f"  原始轨迹总数: {total_count}")
    print(f"  清洗后有效轨迹: {valid_count}")
    print(f"  剔除比例: {(total_count - valid_count) / total_count:.2%}")


def clean_and_save_by_file(input_dir: str, output_dir: str, g_mbr: tuple, output_format: str = 'txt'):
    """
    阶段 2: 扫描每个输入文件，清洗后以原文件名保存到输出目录。
    """
    g_min_x, g_min_y, g_max_x, g_max_y = g_mbr
    multipoint_re = re.compile(r"MULTIPOINT Z\((.*?)\)$")
    point_re = re.compile(r"\(([^()]+)\)")

    # 扫描输入文件列表
    files = sorted(name for name in os.listdir(input_dir) if name.startswith("part-"))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    total_count = 0
    valid_count = 0
    global_tid = 1  # 保持全局 TID 递增

    print(f"阶段 2: 开始分文件清洗，输出目录: {output_dir}")

    for filename in files:
        input_path = os.path.join(input_dir, filename)
        # 保持输出文件名与输入文件名一致
        output_path = os.path.join(output_dir, filename)

        file_valid_count = 0

        with open(input_path, "r", encoding="utf-8") as f_in, \
                open(output_path, "w", encoding="utf-8") as f_out:

            for line in f_in:
                total_count += 1
                obj_id, seg_idx = parse_identifier(line)
                match = multipoint_re.search(line)
                if not match:
                    continue
                raw_points = point_re.findall(match.group(1))

                temp_points = []
                is_dirty = False
                for rp in raw_points:
                    parts = rp.strip().split()
                    if len(parts) < 2:
                        continue
                    try:
                        lon, lat = float(parts[0]), float(parts[1])
                        timestamp = parts[2] if len(parts) > 2 else "0"

                        # 漂移点过滤逻辑
                        if not (g_min_x <= lon <= g_max_x and g_min_y <= lat <= g_max_y):
                            is_dirty = True
                            break
                        temp_points.append((lon, lat, timestamp))
                    except ValueError:
                        continue

                # 轨迹有效性检查
                if not is_dirty and len(temp_points) > MIN_TRAJ_LENGTH:
                    lons = [p[0] for p in temp_points]
                    lats = [p[1] for p in temp_points]
                    if max(lons) > min(lons) and max(lats) > min(lats):
                        f_out.write(format_line(global_tid, obj_id, seg_idx, temp_points, output_format))
                        global_tid += 1
                        valid_count += 1
                        file_valid_count += 1

        print(f"  已处理: {filename} -> 有效轨迹: {file_valid_count}")

    print(f"\n清洗完成！")
    print(f"  总原始轨迹: {total_count}")
    print(f"  总有效轨迹: {valid_count}")
    print(f"  生成文件数: {len(files)}")


if __name__ == "__main__":
    # 配置区 - 使用环境变量或默认路径
    import os
    RAW_DATA_DIR = os.environ.get('RAW_TDRIVE_DIR', r"D:\dataset\Trajectory\TDrive\origin_test")
    OUTPUT_BASE_DIR = os.environ.get('OUTPUT_TDRIVE_DIR', r"D:\dataset\Trajectory\TDrive")

    # 生成三种格式的文件
    complete_dir = os.path.join(OUTPUT_BASE_DIR, "complete")
    formats = ['txt', 'geojson', 'wkt', 'origin']

    mbr = compute_global_mbr(RAW_DATA_DIR)

    for fmt in formats:
        # 自动生成对应的文件名
        ext = "origin" if fmt == "origin" else fmt
        target_path = os.path.join(complete_dir, f"tdrive_cleaned.{ext}")
        print(f"正在生成单文件格式: {fmt}")
        clean_and_save_dataset(RAW_DATA_DIR, target_path, mbr, output_format=fmt)

    by_file_target_path = os.path.join(OUTPUT_BASE_DIR + r"\new_test", f"tdrive_cleaned_by_file")
    clean_and_save_by_file(RAW_DATA_DIR, by_file_target_path, mbr, output_format='origin')

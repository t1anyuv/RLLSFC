"""
Utilities for loading and normalising trajectories from the TDrive dataset.
"""
import hashlib
import json
import os
import re
from typing import Iterable, List, Optional, Tuple

Trajectory = Tuple[int, List[Tuple[float, float]]]


def parse_tdrive_record(line: str) -> Optional[Trajectory]:
    """Parse a single order record from the TDrive dataset."""
    record = line.strip()
    if not record:
        return None

    identifier_match = re.match(r"^(\d+)-(\d+)_(\d+)-MULTIPOINT", record)
    if not identifier_match:
        return None

    identifier = f"{identifier_match.group(1)}-{identifier_match.group(2)}_{identifier_match.group(3)}"
    hash_object = hashlib.md5(identifier.encode('utf-8'))
    trajectory_id = int(hash_object.hexdigest(), 16) % (10 ** 9)

    multipoint_match = re.search(r"MULTIPOINT Z\((.*?)\)$", record)
    if not multipoint_match:
        return None

    points_blob = multipoint_match.group(1).strip()
    point_pattern = r"\(([^()]+)\)"
    raw_points = re.findall(point_pattern, points_blob)

    points: List[Tuple[float, float]] = []
    for raw_point in raw_points:
        parts = raw_point.strip().split()
        if len(parts) < 2:
            continue
        try:
            longitude = float(parts[0])
            latitude = float(parts[1])
        except ValueError:
            continue
        points.append((longitude, latitude))

    if len(points) < 2:
        return None

    return trajectory_id, points


def parse_cleaned_record(line: str, fmt: str = 'txt') -> Optional[Trajectory]:
    """
    解析清洗后的新格式。
    """
    line = line.strip()
    if not line:
        return None

    try:
        if fmt == 'txt':
            # 格式: 全局TID|对象ID|分段索引|lon,lat;lon,lat...
            parts = line.split('|')
            if len(parts) < 4:
                return None
            tid = int(parts[0])
            # 提取坐标点
            point_strs = parts[3].split(';')
            points = []
            for ps in point_strs:
                coords = ps.split(',')
                points.append((float(coords[0]), float(coords[1])))
            return tid, points

        elif fmt == 'geojson':
            # 每行是一个 GeoJSON Feature
            data = json.loads(line)
            tid = int(data["properties"]["tid"])

            # # GeoJSON 存储为 [lon, lat]，转回 (lon, lat)
            points: List[Tuple[float, float]] = [
                (float(c[0]), float(c[1])) for c in data["geometry"]["coordinates"]
            ]
            return tid, points

        elif fmt == 'wkt':
            # 格式: 全局TID,对象ID,分段索引,LINESTRING(lon lat, ...)
            parts = line.split(',', 3)
            if len(parts) < 4:
                return None
            tid = int(parts[0])
            wkt_content = parts[3]  # LINESTRING(...)
            # 提取括号内的内容
            coords_part = wkt_content[wkt_content.find("(") + 1: wkt_content.find(")")]
            points = []
            for pair in coords_part.split(','):
                lon_lat = pair.strip().split(' ')
                points.append((float(lon_lat[0]), float(lon_lat[1])))
            return tid, points

    except (ValueError, KeyError, IndexError):
        return None
    return None


def parse_path_postfix(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    extension_map = {
        '.txt': 'txt',
        '.csv': 'txt',
        '.geojson': 'geojson',
        '.geojsonl': 'geojson',
        '.json': 'geojson',
        '.wkt': 'wkt'
    }
    fmt = extension_map.get(ext)
    if fmt is None:
        print(f"Warning: 无法根据后缀 {ext} 自动识别格式，默认按 'txt' 处理。")
        fmt = 'txt'
    else:
        fmt = fmt.lower()
    return fmt


def load_cleaned_dataset(file_path: str, max_trajectories: Optional[int] = None) -> List[Trajectory]:
    """从单一清洗后的文件中加载。"""

    fmt = parse_path_postfix(file_path)

    trajectories: List[Trajectory] = []

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cleaned file not found: {file_path}")

    print(f"Loading cleaned dataset ({fmt}) from {file_path}...")

    with open(file_path, "r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            if max_trajectories and len(trajectories) >= max_trajectories:
                break

            trajectory = parse_cleaned_record(line, fmt=fmt)
            if trajectory:
                trajectories.append(trajectory)

            if (line_index + 1) % 10000 == 0:
                print(f"  Loaded {len(trajectories)} trajectories...")

    print(f"Successfully loaded {len(trajectories)} trajectories.")
    return trajectories


def load_tdrive_dataset(data_dir: str, max_trajectories: Optional[int] = None) -> List[Trajectory]:
    """Load a subset of the TDrive dataset from disk."""
    trajectories: List[Trajectory] = []
    files = sorted(name for name in os.listdir(data_dir) if name.startswith("part-"))

    if not files:
        raise ValueError(f"No part-* files found in {data_dir}")

    print(f"Discovered {len(files)} files in {data_dir}")

    for index, filename in enumerate(files):
        filepath = os.path.join(data_dir, filename)
        if not os.path.isfile(filepath):
            continue

        print(f"Loading file {index + 1}/{len(files)}: {filename}")
        with open(filepath, "r", encoding="utf-8") as handle:
            for line_index, line in enumerate(handle):
                if max_trajectories and len(trajectories) >= max_trajectories:
                    print(f"Reached order limit ({max_trajectories}).")
                    return trajectories

                trajectory = parse_tdrive_record(line)
                if trajectory:
                    trajectories.append(trajectory)

                if (line_index + 1) % 10000 == 0:
                    print(f"  processed {line_index + 1} lines, loaded {len(trajectories)} trajectories")

    print(f"Loaded {len(trajectories)} trajectories.")
    return trajectories


def _collect_dataset_bounds(trajectories: Iterable[Trajectory]) -> Tuple[float, float, float, float]:
    """Compute the bounding box for a collection of trajectories."""
    min_longitude = float("inf")
    min_latitude = float("inf")
    max_longitude = float("-inf")
    max_latitude = float("-inf")

    found_valid_point = False
    for _, points in trajectories:
        for longitude, latitude in points:
            found_valid_point = True
            min_longitude = min(min_longitude, longitude)
            min_latitude = min(min_latitude, latitude)
            max_longitude = max(max_longitude, longitude)
            max_latitude = max(max_latitude, latitude)

    if not found_valid_point:
        return 0.0, 0.0, 1.0, 1.0

    return min_longitude, min_latitude, max_longitude, max_latitude


def normalize_trajectories(
        trajectories: Iterable[Trajectory], target_bbox: Tuple[float, float, float, float]
) -> List[Trajectory]:
    """Normalise trajectories to the provided target bounding box."""
    trajectory_list = list(trajectories)
    if not trajectory_list:
        return []

    min_lon, min_lat, max_lon, max_lat = _collect_dataset_bounds(trajectory_list)

    longitude_range = max_lon - min_lon if max_lon != min_lon else 1.0
    latitude_range = max_lat - min_lat if max_lat != min_lat else 1.0

    target_min_x, target_min_y, target_max_x, target_max_y = target_bbox
    target_width = target_max_x - target_min_x
    target_height = target_max_y - target_min_y

    normalised: List[Trajectory] = []
    for trajectory_id, points in trajectory_list:
        transformed_points: List[Tuple[float, float]] = []
        for longitude, latitude in points:
            normalised_longitude = (longitude - min_lon) / longitude_range
            normalised_latitude = (latitude - min_lat) / latitude_range
            x_value = target_min_x + normalised_longitude * target_width
            y_value = target_min_y + normalised_latitude * target_height
            transformed_points.append((x_value, y_value))

        normalised.append((trajectory_id, transformed_points))

    return normalised

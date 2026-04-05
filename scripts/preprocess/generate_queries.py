r"""生成查询数据集脚本（通用版本）

支持多种数据集，通过参数指定配置。

生成三种分布类型的查询数据集：
1. uniform: 均匀分布（从轨迹点均匀采样）
2. skewed: 偏斜分布（80%集中在轨迹热点区域）
3. gaussian: 高斯分布（以轨迹点分布为中心）

每种类型包含5个范围: 100m, 500m, 1000m, 1500m, 2000m
每个范围生成100个查询, 每种类型共500个查询

使用示例:
    # TDrive (北京) - 使用预设配置
    python scripts/preprocess/generate_queries.py --dataset tdrive
    
    # 成都数据集 - 使用预设配置
    python scripts/preprocess/generate_queries.py --dataset chengdu
    
    # 自定义参数
    python scripts/preprocess/generate_queries.py \
        --min-lon 115.29 --min-lat 39.00 --max-lon 117.83 --max-lat 41.50 \
        --traj-path D:\dataset\Trajectory\TDrive\complete_clean\tdrive.txt \
        --output-dir resource/queries_tdrive
"""
import argparse
import random
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from collections import defaultdict
import json


# 预定义数据集配置
DATASET_CONFIGS = {
    'tdrive': {
        'min_lon': 115.29,
        'min_lat': 39.00,
        'max_lon': 117.83,
        'max_lat': 41.50,
        'traj_path': r'D:\dataset\Trajectory\TDrive\complete_clean\tdrive.txt',
        'output_dir': 'resource/queries',
    },
    'chengdu': {
        'min_lon': 104.04,
        'min_lat': 30.65,
        'max_lon': 104.13,
        'max_lat': 30.73,
        'traj_path': r'D:\dataset\Trajectory\Chengdu\cleaned_cd_taxi.txt',
        'output_dir': 'resource/queries_chengdu',
    },
}

# 查询范围（米）
QUERY_RANGES = [100, 500, 1000, 1500, 2000]
QUERIES_PER_RANGE = 100


class QueryGenerator:
    """查询生成器 - 从轨迹数据中采样生成查询"""
    
    def __init__(self, min_lon: float, min_lat: float, max_lon: float, max_lat: float,
                 output_dir: Path, traj_path: Optional[str] = None):
        self.min_lon = min_lon
        self.min_lat = min_lat
        self.max_lon = max_lon
        self.max_lat = max_lat
        self.output_dir = output_dir
        self.traj_path = traj_path
        
        # 加载轨迹点
        self.traj_points = self._load_trajectory_points()
        if not self.traj_points:
            raise ValueError(f"未能从轨迹文件加载任何点: {traj_path}")
        
        # 计算轨迹点分布参数（用于高斯分布）
        lons = [p[0] for p in self.traj_points]
        lats = [p[1] for p in self.traj_points]
        self.traj_center_lon = np.mean(lons)
        self.traj_center_lat = np.mean(lats)
        self.traj_std_lon = np.std(lons) if np.std(lons) > 0 else (max_lon - min_lon) / 4
        self.traj_std_lat = np.std(lats) if np.std(lats) > 0 else (max_lat - min_lat) / 4
        
        # 计算热点区域（用于偏斜分布）
        self.hotspot_points = self._compute_hotspot_points()
        
        print(f"已加载 {len(self.traj_points)} 个轨迹点")
        print(f"轨迹中心: ({self.traj_center_lon:.6f}, {self.traj_center_lat:.6f})")
        print(f"轨迹标准差: ({self.traj_std_lon:.6f}, {self.traj_std_lat:.6f})")
        print(f"热点区域包含 {len(self.hotspot_points)} 个点")
    
    def _load_trajectory_points(self) -> List[Tuple[float, float]]:
        """从轨迹文件加载所有轨迹点"""
        points = []
        if not self.traj_path or not Path(self.traj_path).exists():
            print(f"警告: 轨迹文件不存在: {self.traj_path}")
            return points
        
        print(f"正在加载轨迹数据: {self.traj_path}")
        try:
            with open(self.traj_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # 格式: TID|OID|SID|lon,lat;lon,lat;...
                    parts = line.split('|')
                    if len(parts) >= 4:
                        coords_str = parts[3]
                        coord_pairs = coords_str.split(';')
                        for cp in coord_pairs:
                            if ',' in cp:
                                try:
                                    lon_str, lat_str = cp.split(',')
                                    lon, lat = float(lon_str), float(lat_str)
                                    # 只保留在边界框内的点
                                    if self.min_lon <= lon <= self.max_lon and \
                                       self.min_lat <= lat <= self.max_lat:
                                        points.append((lon, lat))
                                except ValueError:
                                    continue
        except Exception as e:
            print(f"加载轨迹文件出错: {e}")
        
        return points
    
    def _compute_hotspot_points(self, grid_size: int = 10) -> List[Tuple[float, float]]:
        """计算热点区域的点（轨迹密度最高的20%区域）"""
        if not self.traj_points:
            return []
        
        # 使用网格统计密度
        lon_bins = np.linspace(self.min_lon, self.max_lon, grid_size + 1)
        lat_bins = np.linspace(self.min_lat, self.max_lat, grid_size + 1)
        
        grid_counts = defaultdict(list)
        for lon, lat in self.traj_points:
            lon_idx = np.searchsorted(lon_bins, lon, side='right') - 1
            lat_idx = np.searchsorted(lat_bins, lat, side='right') - 1
            lon_idx = max(0, min(lon_idx, grid_size - 1))
            lat_idx = max(0, min(lat_idx, grid_size - 1))
            grid_counts[(lon_idx, lat_idx)].append((lon, lat))
        
        # 找到密度最高的20%网格
        grid_densities = [(len(pts), pts) for pts in grid_counts.values()]
        grid_densities.sort(reverse=True)
        
        # 取前20%的网格
        num_hotspot_grids = max(1, int(len(grid_densities) * 0.2))
        hotspot_points = []
        for _, pts in grid_densities[:num_hotspot_grids]:
            hotspot_points.extend(pts)
        
        return hotspot_points if hotspot_points else self.traj_points
    
    def meters_to_degrees(self, meters: float, latitude: float) -> Tuple[float, float]:
        """将米转换为经纬度偏移量"""
        lat_offset = meters / 111000.0
        lon_offset = meters / (111000.0 * np.cos(np.radians(latitude)))
        return lon_offset, lat_offset

    def generate_uniform_queries(self, num_queries: int, range_meters: float) -> List[str]:
        """生成均匀分布的查询 - 从所有轨迹点均匀采样"""
        queries = []
        for _ in range(num_queries):
            center_lon, center_lat = random.choice(self.traj_points)
            lon_offset, lat_offset = self.meters_to_degrees(range_meters / 2, center_lat)
            min_lon = max(self.min_lon, center_lon - lon_offset)
            max_lon = min(self.max_lon, center_lon + lon_offset)
            min_lat = max(self.min_lat, center_lat - lat_offset)
            max_lat = min(self.max_lat, center_lat + lat_offset)
            queries.append(f"{min_lon:.6f}, {min_lat:.6f}, {max_lon:.6f}, {max_lat:.6f}")
        return queries

    def generate_skewed_queries(self, num_queries: int, range_meters: float) -> List[str]:
        """生成偏斜分布的查询（80%集中在轨迹热点区域）"""
        queries = []
        
        for _ in range(num_queries):
            if random.random() < 0.8 and self.hotspot_points:
                center_lon, center_lat = random.choice(self.hotspot_points)
            else:
                center_lon, center_lat = random.choice(self.traj_points)
            
            lon_offset, lat_offset = self.meters_to_degrees(range_meters / 2, center_lat)
            min_lon = max(self.min_lon, center_lon - lon_offset)
            max_lon = min(self.max_lon, center_lon + lon_offset)
            min_lat = max(self.min_lat, center_lat - lat_offset)
            max_lat = min(self.max_lat, center_lat + lat_offset)
            queries.append(f"{min_lon:.6f}, {min_lat:.6f}, {max_lon:.6f}, {max_lat:.6f}")
        return queries

    def generate_gaussian_queries(self, num_queries: int, range_meters: float) -> List[str]:
        """生成高斯分布的查询 - 以轨迹分布为中心"""
        queries = []
        
        for _ in range(num_queries):
            query_center_lon = np.random.normal(self.traj_center_lon, self.traj_std_lon)
            query_center_lat = np.random.normal(self.traj_center_lat, self.traj_std_lat)
            
            query_center_lon = np.clip(query_center_lon, self.min_lon, self.max_lon)
            query_center_lat = np.clip(query_center_lat, self.min_lat, self.max_lat)
            
            lon_offset, lat_offset = self.meters_to_degrees(range_meters / 2, query_center_lat)
            min_lon = max(self.min_lon, query_center_lon - lon_offset)
            max_lon = min(self.max_lon, query_center_lon + lon_offset)
            min_lat = max(self.min_lat, query_center_lat - lat_offset)
            max_lat = min(self.max_lat, query_center_lat + lat_offset)
            queries.append(f"{min_lon:.6f}, {min_lat:.6f}, {max_lon:.6f}, {max_lat:.6f}")
        return queries

    def run(self):
        """运行生成"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        distributions = {
            'gaussian': self.generate_gaussian_queries,
            'skewed': self.generate_skewed_queries,
            'uniform': self.generate_uniform_queries
        }
        
        for dist_name, generator_func in distributions.items():
            print(f"\n生成 {dist_name} 分布查询...")
            all_queries = []
            
            for range_meters in QUERY_RANGES:
                print(f"  范围: {range_meters}m")
                queries = generator_func(QUERIES_PER_RANGE, range_meters)
                for q in queries:
                    all_queries.append({'query': q, 'range_meters': range_meters})
                
                range_dist_dir = self.output_dir / "range" / dist_name
                range_dist_dir.mkdir(parents=True, exist_ok=True)
                filepath = range_dist_dir / f"{dist_name}_{range_meters}m.txt"
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(queries))
                print(f"    生成 {len(queries)} 个查询 -> range/{dist_name}/{filepath.name}")
            
            random.shuffle(all_queries)
            total = len(all_queries)
            train_size = int(total * 0.7)
            val_size = int(total * 0.15)
            
            splits = {
                'train': all_queries[:train_size],
                'val': all_queries[train_size:train_size + val_size],
                'test': all_queries[train_size + val_size:]
            }
            
            dist_dir = self.output_dir / dist_name
            dist_dir.mkdir(parents=True, exist_ok=True)
            
            for split_name, split_data in splits.items():
                filepath = dist_dir / f"queries_{split_name}.json"
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(split_data, f, indent=2, ensure_ascii=False)
                print(f"    已保存 {split_name}: {len(split_data)} 个查询")
            
            print(f"  ✓ {dist_name} 完成: {total} 个查询")
        
        print(f"\n✓ 查询数据集生成完成！")
        print(f"  边界框: [{self.min_lon}, {self.min_lat}, {self.max_lon}, {self.max_lat}]")
        print(f"  输出目录: {self.output_dir}")
        print(f"  文件结构:")
        print(f"    range/[distribution]/[distribution]_[range]m.txt - 分类型分范围文件（各100条）")
        print(f"    gaussian/ | skewed/ | uniform/")
        print(f"      - queries_train.json (350 条)")
        print(f"      - queries_val.json (75 条)")
        print(f"      - queries_test.json (75 条)")


def main():
    parser = argparse.ArgumentParser(description='生成查询数据集（基于轨迹数据采样）')
    parser.add_argument('--dataset', type=str, choices=['tdrive', 'chengdu'],
                        help='使用预定义数据集配置 (tdrive/chengdu)')
    parser.add_argument('--min-lon', type=float, help='最小经度')
    parser.add_argument('--min-lat', type=float, help='最小纬度')
    parser.add_argument('--max-lon', type=float, help='最大经度')
    parser.add_argument('--max-lat', type=float, help='最大纬度')
    parser.add_argument('--traj-path', type=str, help='轨迹数据文件路径')
    parser.add_argument('--output-dir', type=str, help='输出目录')
    
    args = parser.parse_args()
    
    # 如果使用预定义数据集
    if args.dataset:
        config = DATASET_CONFIGS[args.dataset]
        min_lon = config['min_lon']
        min_lat = config['min_lat']
        max_lon = config['max_lon']
        max_lat = config['max_lat']
        traj_path = config['traj_path']
        output_dir = Path(config['output_dir'])
        print(f"使用预定义配置: {args.dataset}")
    else:
        # 使用命令行参数或默认值
        min_lon = args.min_lon or 115.29
        min_lat = args.min_lat or 39.00
        max_lon = args.max_lon or 117.83
        max_lat = args.max_lat or 41.50
        traj_path = args.traj_path or r'D:\dataset\Trajectory\TDrive\complete_clean\tdrive.txt'
        output_dir = Path(args.output_dir or 'resource/queries')
    
    generator = QueryGenerator(
        min_lon=min_lon,
        min_lat=min_lat,
        max_lon=max_lon,
        max_lat=max_lat,
        output_dir=output_dir,
        traj_path=traj_path
    )
    generator.run()


if __name__ == "__main__":
    main()

"""生成查询数据集脚本

生成三种分布类型的查询数据集：
1. uniform: 均匀分布
2. skewed: 偏斜分布
3. gaussian: 高斯分布

每种类型包含5个范围: 100m, 500m, 1000m, 1500m, 2000m
每个范围生成100个查询, 每种类型共500个查询
"""
import random
import numpy as np
from pathlib import Path
from typing import List, Tuple


# TDrive数据集边界框（北京）
MIN_LON = 115.29
MIN_LAT = 39.00
MAX_LON = 117.83
MAX_LAT = 41.50

# 查询范围（米）
QUERY_RANGES = [100, 500, 1000, 1500, 2000]
QUERIES_PER_RANGE = 100

# 输出目录
OUTPUT_DIR = Path("resource/queries")


def meters_to_degrees(meters: float, latitude: float) -> Tuple[float, float]:
    """将米转换为经纬度偏移量
    
    Args:
        meters: 距离（米）
        latitude: 纬度（用于计算经度偏移）
    
    Returns:
        (lon_offset, lat_offset) 经纬度偏移量
    """
    # 1度纬度约等于111km
    lat_offset = meters / 111000.0
    
    # 1度经度 = 111km * cos(latitude)
    lon_offset = meters / (111000.0 * np.cos(np.radians(latitude)))
    
    return lon_offset, lat_offset


def generate_uniform_queries(num_queries: int, range_meters: float) -> List[str]:
    """生成均匀分布的查询
    
    Args:
        num_queries: 查询数量
        range_meters: 查询范围（米）
    
    Returns:
        查询字符串列表
    """
    queries = []
    
    for _ in range(num_queries):
        # 随机选择中心点
        center_lon = random.uniform(MIN_LON, MAX_LON)
        center_lat = random.uniform(MIN_LAT, MAX_LAT)
        
        # 计算偏移量
        lon_offset, lat_offset = meters_to_degrees(range_meters / 2, center_lat)
        
        # 计算边界框
        min_lon = max(MIN_LON, center_lon - lon_offset)
        max_lon = min(MAX_LON, center_lon + lon_offset)
        min_lat = max(MIN_LAT, center_lat - lat_offset)
        max_lat = min(MAX_LAT, center_lat + lat_offset)
        
        queries.append(f"{min_lon:.6f}, {min_lat:.6f}, {max_lon:.6f}, {max_lat:.6f}")
    
    return queries


def generate_skewed_queries(num_queries: int, range_meters: float) -> List[str]:
    """生成偏斜分布的查询（80%集中在20%的热点区域）
    
    Args:
        num_queries: 查询数量
        range_meters: 查询范围（米）
    
    Returns:
        查询字符串列表
    """
    queries = []
    
    # 定义热点区域（北京市中心附近）
    hotspot_center_lon = (MIN_LON + MAX_LON) / 2
    hotspot_center_lat = (MIN_LAT + MAX_LAT) / 2
    hotspot_radius_lon = (MAX_LON - MIN_LON) * 0.2
    hotspot_radius_lat = (MAX_LAT - MIN_LAT) * 0.2
    
    for _ in range(num_queries):
        # 80%的查询在热点区域
        if random.random() < 0.8:
            center_lon = random.uniform(
                hotspot_center_lon - hotspot_radius_lon,
                hotspot_center_lon + hotspot_radius_lon
            )
            center_lat = random.uniform(
                hotspot_center_lat - hotspot_radius_lat,
                hotspot_center_lat + hotspot_radius_lat
            )
        else:
            center_lon = random.uniform(MIN_LON, MAX_LON)
            center_lat = random.uniform(MIN_LAT, MAX_LAT)
        
        # 计算偏移量
        lon_offset, lat_offset = meters_to_degrees(range_meters / 2, center_lat)
        
        # 计算边界框
        min_lon = max(MIN_LON, center_lon - lon_offset)
        max_lon = min(MAX_LON, center_lon + lon_offset)
        min_lat = max(MIN_LAT, center_lat - lat_offset)
        max_lat = min(MAX_LAT, center_lat + lat_offset)
        
        queries.append(f"{min_lon:.6f}, {min_lat:.6f}, {max_lon:.6f}, {max_lat:.6f}")
    
    return queries


def generate_gaussian_queries(num_queries: int, range_meters: float) -> List[str]:
    """生成高斯分布的查询
    
    Args:
        num_queries: 查询数量
        range_meters: 查询范围（米）
    
    Returns:
        查询字符串列表
    """
    queries = []
    
    # 高斯分布中心（北京市中心附近）
    center_lon = (MIN_LON + MAX_LON) / 2
    center_lat = (MIN_LAT + MAX_LAT) / 2
    
    # 标准差（覆盖约95%的区域在边界内）
    std_lon = (MAX_LON - MIN_LON) / 4
    std_lat = (MAX_LAT - MIN_LAT) / 4
    
    for _ in range(num_queries):
        # 使用高斯分布生成中心点
        query_center_lon = np.random.normal(center_lon, std_lon)
        query_center_lat = np.random.normal(center_lat, std_lat)
        
        # 确保在边界内
        query_center_lon = np.clip(query_center_lon, MIN_LON, MAX_LON)
        query_center_lat = np.clip(query_center_lat, MIN_LAT, MAX_LAT)
        
        # 计算偏移量
        lon_offset, lat_offset = meters_to_degrees(range_meters / 2, query_center_lat)
        
        # 计算边界框
        min_lon = max(MIN_LON, query_center_lon - lon_offset)
        max_lon = min(MAX_LON, query_center_lon + lon_offset)
        min_lat = max(MIN_LAT, query_center_lat - lat_offset)
        max_lat = min(MAX_LAT, query_center_lat + lat_offset)
        
        queries.append(f"{min_lon:.6f}, {min_lat:.6f}, {max_lon:.6f}, {max_lat:.6f}")
    
    return queries


def main():
    """主函数"""
    import json
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    distributions = {
        'gaussian': generate_gaussian_queries,
        'skewed': generate_skewed_queries,
        'uniform': generate_uniform_queries
    }
    
    for dist_name, generator_func in distributions.items():
        print(f"\n生成 {dist_name} 分布查询...")
        
        all_queries = []
        
        for range_meters in QUERY_RANGES:
            print(f"  范围: {range_meters}m")
            
            # 生成查询
            queries = generator_func(QUERIES_PER_RANGE, range_meters)
            
            # 添加到总列表，带上范围标签
            for q in queries:
                all_queries.append({
                    'query': q,
                    'range_meters': range_meters
                })
            
            # 保存分类型分范围的独立文件
            filepath = OUTPUT_DIR / f"{dist_name}_{range_meters}m.txt"
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write('\n'.join(queries))
            
            print(f"    生成 {len(queries)} 个查询 -> {filepath.name}")
        
        # 对该分布类型的查询随机打乱
        random.shuffle(all_queries)
        
        # 切分数据集: 70% 训练, 15% 验证, 15% 测试
        total = len(all_queries)
        train_size = int(total * 0.7)
        val_size = int(total * 0.15)
        
        train_queries = all_queries[:train_size]
        val_queries = all_queries[train_size:train_size + val_size]
        test_queries = all_queries[train_size + val_size:]
        
        # 保存该分布类型的切分数据集
        dist_dir = OUTPUT_DIR / dist_name
        dist_dir.mkdir(parents=True, exist_ok=True)
        
        splits = {
            'train': train_queries,
            'val': val_queries,
            'test': test_queries
        }
        
        for split_name, split_data in splits.items():
            filepath = dist_dir / f"queries_{split_name}.json"
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(split_data, f, indent=2, ensure_ascii=False)
            print(f"    已保存 {split_name}: {len(split_data)} 个查询")
        
        print(f"  ✓ {dist_name} 完成: {total} 个查询")
    
    print(f"\n✓ 所有查询数据集生成完成！")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"  文件结构:")
    print(f"    [distribution]_[range]m.txt - 分类型分范围文件（各100条）")
    print(f"    gaussian/ | skewed/ | uniform/")
    print(f"      - queries_train.json (350 条)")
    print(f"      - queries_val.json (75 条)")
    print(f"      - queries_test.json (75 条)")


if __name__ == "__main__":
    main()

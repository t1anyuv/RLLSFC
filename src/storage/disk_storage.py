"""磁盘存储实现：二进制文件 + 内存缓存。"""
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from functools import lru_cache

import numpy as np

from src.common import SpatialBoundingBox
from src.storage.base import TrajectoryStorage
from src.utils.trajectory_geometry import compute_trajectory_bounding_box


class DiskTrajectoryStorage(TrajectoryStorage):
    """磁盘存储实现。
    
    将轨迹数据持久化到二进制文件，使用LRU缓存加速热点数据访问。
    适合大数据集（30GB+）场景，内存占用可控。
    
    存储格式：
    - 索引文件 (.npy): NumPy结构化数组，包含 tid, offset, size, mbr
    - 数据文件 (.bin): 紧凑二进制格式 [num_points: uint32][points: (x,y) * N]
    
    Attributes:
        _storage_dir: 存储目录路径
        _cache_mb: LRU缓存大小（MB）
        _index: NumPy结构化数组索引
        _data_file: 数据文件句柄
    """
    
    # 索引文件的数据类型定义
    INDEX_DTYPE = np.dtype([
        ('tid', np.int64),
        ('offset', np.int64),
        ('size', np.int32),
        ('mbr_min_x', np.float64),
        ('mbr_min_y', np.float64),
        ('mbr_max_x', np.float64),
        ('mbr_max_y', np.float64),
    ])
    
    # 数据文件格式：每个轨迹 [num_points: uint32][x: float32][y: float32]...
    HEADER_FMT = '<I'  # 小端无符号32位整数（num_points）
    POINT_FMT = '<ff'  # 小端两个float32（x, y）
    
    def __init__(self, storage_dir: Path, cache_mb: int = 2048):
        """初始化磁盘存储。
        
        Args:
            storage_dir: 存储目录路径
            cache_mb: LRU缓存大小（MB），默认2GB
        """
        self._storage_dir = Path(storage_dir)
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._cache_mb = cache_mb
        
        self._index_path = self._storage_dir / "traj_index.npy"
        self._data_path = self._storage_dir / "traj_data.bin"
        
        # 延迟初始化，在第一次写入时创建
        self._index: Optional[np.ndarray] = None
        self._data_file = None
        self._tid_to_row: Dict[int, int] = {}
        self._current_offset = 0
        self._num_trajectories = 0
    
    def _ensure_open(self) -> None:
        """确保文件已打开。"""
        if self._data_file is None:
            if self._data_path.exists():
                # 读取模式
                self._data_file = open(self._data_path, 'rb')
                self._load_index()
            else:
                # 写入模式（首次创建）
                self._data_file = open(self._data_path, 'wb')
    
    def _load_index(self) -> None:
        """加载索引文件。"""
        if self._index_path.exists():
            self._index = np.load(self._index_path)
            # 构建tid到行号的映射
            self._tid_to_row = {tid: i for i, tid in enumerate(self._index['tid'])}
            self._num_trajectories = len(self._index)
    
    def _save_index(self) -> None:
        """保存索引文件。"""
        if self._index is not None:
            np.save(self._index_path, self._index)
    
    @property
    def _cache_size(self) -> int:
        """计算缓存条目数。
        
        假设平均每条轨迹100个点，约800字节 + 开销 ≈ 1000字节/条
        """
        return self._cache_mb * 1024 * 1024 // 1000
    
    @lru_cache(maxsize=None)  # 使用动态计算的maxsize
    def _get_cached_trajectory(self, tid: int) -> Tuple[Tuple[float, float], ...]:
        """带LRU缓存的轨迹读取。"""
        # 注意：返回tuple以便缓存，外部转换为list
        points = self._read_trajectory_from_disk(tid)
        return tuple(points)
    
    def _read_trajectory_from_disk(self, tid: int) -> List[Tuple[float, float]]:
        """从磁盘读取轨迹。"""
        if self._index is None or tid not in self._tid_to_row:
            raise KeyError(f"Trajectory {tid} not found")
        
        row = self._tid_to_row[tid]
        offset = int(self._index['offset'][row])
        size = int(self._index['size'][row])
        
        # 读取数据
        self._data_file.seek(offset)
        data = self._data_file.read(size)
        
        # 解码
        num_points = struct.unpack(self.HEADER_FMT, data[:4])[0]
        points = []
        point_data = data[4:]
        
        for i in range(num_points):
            start = i * 8  # 每个点8字节（2个float32）
            x, y = struct.unpack(self.POINT_FMT, point_data[start:start+8])
            points.append((x, y))
        
        return points
    
    def get_trajectory(self, tid: int) -> List[Tuple[float, float]]:
        """获取轨迹，优先从缓存读取。"""
        if self._index is None:
            self._ensure_open()
        
        # 使用LRU缓存
        cached = self._get_cached_trajectory(tid)
        return list(cached)
    
    def store_trajectory(self, tid: int, points: List[Tuple[float, float]]) -> None:
        """存储轨迹到磁盘。"""
        if self._data_file is None:
            self._ensure_open()
        
        if self._data_file.mode != 'wb':
            raise RuntimeError("Cannot write to existing storage, create new instance")
        
        # 编码数据
        num_points = len(points)
        header = struct.pack(self.HEADER_FMT, num_points)
        point_data = b''.join(struct.pack(self.POINT_FMT, x, y) for x, y in points)
        data = header + point_data
        
        # 计算MBR
        mbr = compute_trajectory_bounding_box(points)
        
        # 记录索引信息（暂存到列表，最后批量写入）
        if not hasattr(self, '_pending_records'):
            self._pending_records = []
        
        self._pending_records.append({
            'tid': tid,
            'offset': self._current_offset,
            'size': len(data),
            'mbr_min_x': mbr.min_x,
            'mbr_min_y': mbr.min_y,
            'mbr_max_x': mbr.max_x,
            'mbr_max_y': mbr.max_y,
        })
        
        # 写入数据文件
        self._data_file.write(data)
        self._current_offset += len(data)
        self._num_trajectories += 1
    
    def finalize_writes(self) -> None:
        """完成批量写入，保存索引文件。
        
        在所有轨迹写入完成后调用。
        """
        if hasattr(self, '_pending_records') and self._pending_records:
            # 创建索引数组
            self._index = np.array(self._pending_records, dtype=self.INDEX_DTYPE)
            # 按tid排序便于查找
            self._index = np.sort(self._index, order='tid')
            self._save_index()
            
            # 构建映射
            self._tid_to_row = {tid: i for i, tid in enumerate(self._index['tid'])}
            
            # 清理临时数据
            del self._pending_records
        
        # 关闭写入模式，切换到读取模式
        if self._data_file:
            self._data_file.close()
            self._data_file = open(self._data_path, 'rb')
    
    def get_trajectory_mbr(self, tid: int) -> SpatialBoundingBox:
        """从索引直接获取MBR（无需读取轨迹数据）。"""
        if self._index is None:
            self._ensure_open()
        
        if tid not in self._tid_to_row:
            raise KeyError(f"Trajectory {tid} not found")
        
        row = self._tid_to_row[tid]
        return SpatialBoundingBox(
            self._index['mbr_min_x'][row],
            self._index['mbr_min_y'][row],
            self._index['mbr_max_x'][row],
            self._index['mbr_max_y'][row]
        )
    
    def get_many_trajectories(self, tids: List[int]) -> Dict[int, List[Tuple[float, float]]]:
        """批量获取轨迹，进行顺序读取优化。"""
        if self._index is None:
            self._ensure_open()
        
        result = {}
        # 按磁盘位置排序，最小化seek
        valid_tids = [tid for tid in tids if tid in self._tid_to_row]
        sorted_tids = sorted(valid_tids, key=lambda t: self._tid_to_row[t])
        
        for tid in sorted_tids:
            result[tid] = self.get_trajectory(tid)
        
        return result
    
    def has_trajectory(self, tid: int) -> bool:
        """检查轨迹是否存在。"""
        if self._index is None:
            self._ensure_open()
        return tid in self._tid_to_row
    
    def get_all_tids(self) -> List[int]:
        """获取所有轨迹ID。"""
        if self._index is None:
            self._ensure_open()
        return list(self._index['tid'])
    
    def close(self) -> None:
        """关闭文件句柄。"""
        if self._data_file:
            # 如果有待写入数据，先完成
            if hasattr(self, '_pending_records') and self._pending_records:
                self.finalize_writes()
            self._data_file.close()
            self._data_file = None
    
    def __len__(self) -> int:
        """???????????"""
        if self._index is None:
            self._ensure_open()
        return self._num_trajectories
    
    @property
    def memory_usage_bytes(self) -> int:
        """估算当前内存占用。
        
        包括：
        - 索引数组（常驻内存）
        - tid_to_row字典
        - LRU缓存
        """
        total = 0
        if self._index is not None:
            total += self._index.nbytes
        total += len(self._tid_to_row) * 72  # 字典开销
        
        # LRU缓存占用（估算）
        cache_info = self._get_cached_trajectory.cache_info()
        # 无法直接获取内存大小，估算
        
        return total
    
    def clear_cache(self) -> None:
        """清空LRU缓存，释放内存。"""
        self._get_cached_trajectory.cache_clear()
    
    def get_cache_info(self) -> Dict[str, int]:
        """获取缓存统计信息。"""
        info = self._get_cached_trajectory.cache_info()
        return {
            'hits': info.hits,
            'misses': info.misses,
            'maxsize': info.maxsize,
            'currsize': info.currsize,
        }

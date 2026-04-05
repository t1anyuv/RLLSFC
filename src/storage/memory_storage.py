"""内存存储实现：全量加载轨迹到内存。"""
import sys
from typing import Dict, List, Tuple

from src.core.bounding_box import SpatialBoundingBox
from src.storage.base import TrajectoryStorage
from src.utils.trajectory_geometry import compute_trajectory_bounding_box


class InMemoryTrajectoryStorage(TrajectoryStorage):
    """内存存储实现。
    
    将所有轨迹数据存储在内存字典中，提供最快的访问速度。
    适合小数据集（< 5GB内存）场景。
    
    Attributes:
        _trajectory_points: 轨迹ID到坐标点的映射
        _trajectory_mbrs: 轨迹ID到MBR的映射
    """
    
    def __init__(self):
        """初始化空存储。"""
        self._trajectory_points: Dict[int, List[Tuple[float, float]]] = {}
        self._trajectory_mbrs: Dict[int, SpatialBoundingBox] = {}
    
    def get_trajectory(self, tid: int) -> List[Tuple[float, float]]:
        """获取指定轨迹的坐标点序列。
        
        Args:
            tid: 轨迹ID
            
        Returns:
            轨迹坐标点列表
            
        Raises:
            KeyError: 轨迹ID不存在
        """
        if tid not in self._trajectory_points:
            raise KeyError(f"Trajectory {tid} not found")
        return self._trajectory_points[tid]
    
    def store_trajectory(self, tid: int, points: List[Tuple[float, float]]) -> None:
        """存储轨迹数据。
        
        Args:
            tid: 轨迹ID
            points: 轨迹坐标点列表
        """
        self._trajectory_points[tid] = points
        # 预计算MBR加速后续查询
        self._trajectory_mbrs[tid] = compute_trajectory_bounding_box(points)
    
    def get_trajectory_mbr(self, tid: int) -> SpatialBoundingBox:
        """获取轨迹MBR。
        
        Args:
            tid: 轨迹ID
            
        Returns:
            轨迹MBR
            
        Raises:
            KeyError: 轨迹ID不存在
        """
        if tid not in self._trajectory_mbrs:
            raise KeyError(f"Trajectory {tid} not found")
        return self._trajectory_mbrs[tid]
    
    def get_many_trajectories(self, tids: List[int]) -> Dict[int, List[Tuple[float, float]]]:
        """批量获取轨迹。
        
        内存模式下直接字典查询。
        
        Args:
            tids: 轨迹ID列表
            
        Returns:
            轨迹ID到坐标点的映射
        """
        return {tid: self._trajectory_points[tid] for tid in tids if tid in self._trajectory_points}
    
    def has_trajectory(self, tid: int) -> bool:
        """检查轨迹是否存在。"""
        return tid in self._trajectory_points
    
    def get_all_tids(self) -> List[int]:
        """获取所有轨迹ID。"""
        return list(self._trajectory_points.keys())
    
    def close(self) -> None:
        """清空存储释放内存。"""
        self._trajectory_points.clear()
        self._trajectory_mbrs.clear()
    
    def __len__(self) -> int:
        """返回轨迹总数。"""
        return len(self._trajectory_points)
    
    @property
    def memory_usage_bytes(self) -> int:
        """估算当前内存占用（字节）。
        
        计算方法：
        - 字典开销：每个条目约72字节（Python dict overhead）
        - 坐标点：每个点2个float64，16字节
        - MBR：每个4个float64，32字节
        """
        total = sys.getsizeof(self._trajectory_points)
        total += sys.getsizeof(self._trajectory_mbrs)
        
        # 轨迹点数据
        for points in self._trajectory_points.values():
            # 列表开销 + 元组开销 + float开销
            total += sys.getsizeof(points)
            total += len(points) * 16  # 每个点约16字节
        
        # MBR数据
        for mbr in self._trajectory_mbrs.values():
            total += sys.getsizeof(mbr)
        
        return total

    @property
    def trajectory_points(self):
        return self._trajectory_points

    @property
    def trajectory_mbrs(self):
        return self._trajectory_mbrs

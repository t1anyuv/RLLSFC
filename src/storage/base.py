"""轨迹存储抽象基类，定义统一访问接口。"""
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional

from src.common import SpatialBoundingBox


Trajectory = Tuple[int, List[Tuple[float, float]]]


class TrajectoryStorage(ABC):
    """轨迹存储抽象基类。
    
    提供统一的轨迹数据访问接口，支持内存模式和磁盘模式两种实现。
    所有轨迹数据操作应通过此类接口进行，避免直接访问底层存储结构。
    """
    
    @abstractmethod
    def get_trajectory(self, tid: int) -> List[Tuple[float, float]]:
        """获取指定轨迹的坐标点序列。
        
        Args:
            tid: 轨迹ID
            
        Returns:
            轨迹坐标点列表，每个点为 (x, y) 元组
            
        Raises:
            KeyError: 轨迹ID不存在
        """
        pass
    
    @abstractmethod
    def store_trajectory(self, tid: int, points: List[Tuple[float, float]]) -> None:
        """存储轨迹数据。
        
        此方法在数据加载阶段批量调用，将轨迹持久化到存储中。
        
        Args:
            tid: 轨迹ID
            points: 轨迹坐标点列表
        """
        pass
    
    @abstractmethod
    def get_trajectory_mbr(self, tid: int) -> SpatialBoundingBox:
        """获取轨迹的最小边界矩形 (MBR)。
        
        Args:
            tid: 轨迹ID
            
        Returns:
            轨迹的MBR边界框
            
        Raises:
            KeyError: 轨迹ID不存在
        """
        pass
    
    @abstractmethod
    def get_many_trajectories(self, tids: List[int]) -> Dict[int, List[Tuple[float, float]]]:
        """批量获取多条轨迹。
        
        用于相似度计算等需要同时访问多条轨迹的场景。
        磁盘存储实现可进行批量预读取优化。
        
        Args:
            tids: 轨迹ID列表
            
        Returns:
            轨迹ID到坐标点的映射字典
        """
        pass
    
    @abstractmethod
    def has_trajectory(self, tid: int) -> bool:
        """检查轨迹是否存在。
        
        Args:
            tid: 轨迹ID
            
        Returns:
            是否存在该轨迹
        """
        pass
    
    @abstractmethod
    def get_all_tids(self) -> List[int]:
        """获取所有轨迹ID列表。
        
        Returns:
            所有存储的轨迹ID列表
        """
        pass
    
    @abstractmethod
    def close(self) -> None:
        """关闭存储，释放资源。
        
        磁盘存储实现需要关闭文件句柄。
        """
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        """返回存储的轨迹总数。"""
        pass
    
    @property
    @abstractmethod
    def memory_usage_bytes(self) -> int:
        """返回当前内存占用（字节）。
        
        用于监控和自动模式切换决策。
        """
        pass
    
    def __enter__(self):
        """支持上下文管理器。"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出时自动关闭资源。"""
        self.close()

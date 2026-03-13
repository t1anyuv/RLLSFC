"""轨迹存储模块：支持内存和磁盘两种存储模式。

此模块提供统一的轨迹数据存储接口，支持：
- InMemoryTrajectoryStorage: 全内存存储，适合小数据集
- DiskTrajectoryStorage: 磁盘存储+LRU缓存，适合大数据集

示例用法:
    >>> from src.storage import create_storage
    >>> storage = create_storage("memory")
    >>> storage.store_trajectory(1, [(0.1, 0.2), (0.3, 0.4)])
    >>> points = storage.get_trajectory(1)
"""
from pathlib import Path
from typing import Literal, Optional, Union

from src.storage.base import TrajectoryStorage, Trajectory
from src.storage.memory_storage import InMemoryTrajectoryStorage
from src.storage.disk_storage import DiskTrajectoryStorage


def create_storage(
    mode: Literal["memory", "disk", "auto"] = "auto",
    storage_dir: Optional[Union[str, Path]] = None,
    cache_mb: int = 2048,
    estimated_data_gb: Optional[float] = None,
) -> TrajectoryStorage:
    """创建轨迹存储实例。
    
    Args:
        mode: 存储模式
            - "memory": 使用内存存储
            - "disk": 使用磁盘存储
            - "auto": 根据预估数据量自动选择（默认）
        storage_dir: 磁盘存储目录（磁盘模式必需）
        cache_mb: 磁盘模式LRU缓存大小（MB），默认2GB
        estimated_data_gb: 预估数据大小（GB），用于auto模式决策
        
    Returns:
        配置好的存储实例
        
    Raises:
        ValueError: 参数配置错误
        
    Examples:
        >>> # 内存模式
        >>> storage = create_storage("memory")
        >>> 
        >>> # 磁盘模式
        >>> storage = create_storage("disk", storage_dir="./storage", cache_mb=4096)
        >>> 
        >>> # 自动模式（预估>10GB自动选择磁盘）
        >>> storage = create_storage("auto", estimated_data_gb=30.0)
    """
    # Auto模式决策
    if mode == "auto":
        threshold_gb = 8.0  # 默认8GB阈值
        if estimated_data_gb and estimated_data_gb > threshold_gb:
            mode = "disk"
        else:
            mode = "memory"
    
    # 创建对应实现
    if mode == "memory":
        return InMemoryTrajectoryStorage()
    
    elif mode == "disk":
        if storage_dir is None:
            # 使用默认路径
            from src.utils.path_manager import get_path_manager
            pm = get_path_manager()
            storage_dir = pm.get_project_root() / "resource" / "storage"
        
        return DiskTrajectoryStorage(Path(storage_dir), cache_mb=cache_mb)
    
    else:
        raise ValueError(f"Unknown storage mode: {mode}")


__all__ = [
    "TrajectoryStorage",
    "InMemoryTrajectoryStorage", 
    "DiskTrajectoryStorage",
    "create_storage",
    "Trajectory",
]

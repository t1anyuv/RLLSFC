"""轨迹统计特征计算。"""
from typing import Dict, List, Tuple

import numpy as np

from src.indexing.quadtree_cell import QuadTreeCell


class TrajectoryStatistics:
    """轨迹统计特征计算器。
    
    负责计算单元格的轨迹相关统计信息：
    - N_ee: 被扩大元素覆盖的轨迹数量
    - N_int: 与轨迹 MBR 的交集数量
    - N_cov: 完整覆盖轨迹 MBR 的数量
    - density_center: 轨迹密度中心
    """

    def __init__(self, target_cells: List[QuadTreeCell]):
        self.target_cells = target_cells
        self.ee_counts: Dict[QuadTreeCell, int] = {}
        self.intersect_counts: Dict[QuadTreeCell, int] = {}
        self.cover_counts: Dict[QuadTreeCell, int] = {}
        self.density_centres: Dict[QuadTreeCell, Tuple[float, float]] = {}
        self.max_ee_count = 1

    def compute(self) -> None:
        """计算所有轨迹统计特征。"""
        # 1. N_ee 计算
        self.ee_counts = {cell: len(cell.trajectories) for cell in self.target_cells}
        self.intersect_counts = {cell: 0 for cell in self.target_cells}
        self.cover_counts = {cell: 0 for cell in self.target_cells}

        # 缓存所有轨迹 MBR
        trajectory_mbrs: Dict[int, Tuple[float, float, float, float]] = {}
        for cell in self.target_cells:
            for tid, bbox in cell.trajectory_mbrs.items():
                trajectory_mbrs[tid] = (bbox.min_x, bbox.min_y, bbox.max_x, bbox.max_y)

        # 2. N_int, N_cov 计算
        for cell in self.target_cells:
            cb = cell.bbox
            cminx, cminy, cmaxx, cmaxy = cb.min_x, cb.min_y, cb.max_x, cb.max_y
            
            for tid, (tminx, tminy, tmaxx, tmaxy) in trajectory_mbrs.items():
                # 检查相交
                if cmaxx < tminx or cminx > tmaxx or cmaxy < tminy or cminy > tmaxy:
                    continue
                self.intersect_counts[cell] += 1
                
                # 检查完全覆盖
                if cminx <= tminx and cmaxx >= tmaxx and cminy <= tminy and cmaxy >= tmaxy:
            
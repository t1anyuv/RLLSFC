"""轨迹统计特征计算。"""
from typing import Dict, List, Tuple

import numpy as np

from src.indexing.quadtree_cell import QuadTreeCell


class TrajectoryStatistics:
    """计算目标单元格上的轨迹统计信息。"""

    def __init__(self, target_cells: List[QuadTreeCell]):
        self.target_cells = target_cells
        self.ee_counts: Dict[QuadTreeCell, int] = {}
        self.intersect_counts: Dict[QuadTreeCell, int] = {}
        self.cover_counts: Dict[QuadTreeCell, int] = {}
        self.density_centres: Dict[QuadTreeCell, Tuple[float, float]] = {}
        self.max_ee_count = 1

    def compute(self) -> None:
        """
        计算轨迹统计特征。

        计算内容：
        1. N_ee: 被扩大元素覆盖的轨迹数量
        2. N_int, N_cov: 与轨迹 MBR 的交集、覆盖数量
        3. density_center: 轨迹密度中心
        """
        self.ee_counts = {cell: len(cell.trajectories) for cell in self.target_cells}
        self.intersect_counts = {cell: 0 for cell in self.target_cells}
        self.cover_counts = {cell: 0 for cell in self.target_cells}
        self.density_centres = {}

        trajectory_mbrs = self._cache_trajectory_mbrs()
        self._compute_intersections_and_covers(trajectory_mbrs)
        self._compute_density_centers(trajectory_mbrs)
        self.max_ee_count = max(max(self.ee_counts.values(), default=0), 1)

    def _cache_trajectory_mbrs(self) -> Dict[int, Tuple[float, float, float, float]]:
        """缓存所有轨迹的 MBR。"""
        trajectory_mbrs: Dict[int, Tuple[float, float, float, float]] = {}
        for cell in self.target_cells:
            for tid, bbox in cell.trajectory_mbrs.items():
                trajectory_mbrs[tid] = (bbox.min_x, bbox.min_y, bbox.max_x, bbox.max_y)
        return trajectory_mbrs

    def _compute_intersections_and_covers(
        self,
        trajectory_mbrs: Dict[int, Tuple[float, float, float, float]],
    ) -> None:
        """计算每个单元格与轨迹 MBR 的相交数和覆盖数。"""
        for cell in self.target_cells:
            cb = cell.bbox
            cminx, cminy, cmaxx, cmaxy = cb.min_x, cb.min_y, cb.max_x, cb.max_y

            for tminx, tminy, tmaxx, tmaxy in trajectory_mbrs.values():
                if cmaxx < tminx or cminx > tmaxx or cmaxy < tminy or cminy > tmaxy:
                    continue

                self.intersect_counts[cell] += 1

                if cminx <= tminx and cmaxx >= tmaxx and cminy <= tminy and cmaxy >= tmaxy:
                    self.cover_counts[cell] += 1

    def _compute_density_centers(
        self,
        trajectory_mbrs: Dict[int, Tuple[float, float, float, float]],
    ) -> None:
        """计算每个单元格的轨迹密度中心。"""
        for cell in self.target_cells:
            tids = cell.trajectories
            if not tids:
                self.density_centres[cell] = cell.get_center()
                continue

            centers = []
            for tid in tids:
                if tid not in trajectory_mbrs:
                    continue
                tminx, tminy, tmaxx, tmaxy = trajectory_mbrs[tid]
                centers.append(((tminx + tmaxx) / 2, (tminy + tmaxy) / 2))

            if centers:
                arr = np.array(centers, dtype=float)
                self.density_centres[cell] = (float(arr[:, 0].mean()), float(arr[:, 1].mean()))
            else:
                self.density_centres[cell] = cell.get_center()

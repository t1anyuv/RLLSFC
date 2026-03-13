"""
Feature engineering for the reinforcement learning state representation.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.indexing.quadtree_cell import QuadTreeCell
from src.indexing.quadtree_index import QuadTreeIndex


class StateFeatureBuilder:
    """
    为所有目标 QuadTreeCell 预计算特征向量并动态构建状态。

    特征包括：
    1. 轨迹统计特征 (4维)：
       - N_ee_log: 被扩大元素覆盖的轨迹数量(对数)
       - N_ee / N_int: 覆盖与交集比例
       - N_cov / N_int: 完整覆盖比例
       - d_center: 密度中心距离(对数)
    2. 空间特征 (4维)：
       - Δx, Δy: cell 中心相对于整体空间中心的偏移
       - Δdx, Δdy: 轨迹密度中心相对于扩大元素中心的偏移
    3. 结构特征 (2维)：
       - S: log 面积
       - qcode: 四叉树编码值
    4. 遍历特征 (4维)：
       - visited_ratio: 访问比例
       - Δx_rel, Δy_rel: 相对于上一个访问节点的位移
       - n_heat: 邻居中未访问节点的轨迹密度
    
    总维度: 10 (静态) + 4 (动态) = 14
    """
    
    # 常量定义
    EPSILON = 1e-6
    LOG_EPSILON = 1e-8
    DISTANCE_SCALE = 10.0
    MAX_NEIGHBORS = 8

    def __init__(
            self,
            quadtree: QuadTreeIndex,
            alpha: int,
            beta: int,
            target_cells: List[QuadTreeCell],
            cache_dir: Optional[str] = None,
    ):
        self.quadtree = quadtree
        self.alpha = alpha
        self.beta = beta

        # static(10) + dynamic(4)
        self.state_dimensions = 14

        self.cache_dir = cache_dir
        self.target_cells = target_cells

        self.ee_counts: Dict[QuadTreeCell, int] = {}
        self.intersect_counts: Dict[QuadTreeCell, int] = {}
        self.cover_counts: Dict[QuadTreeCell, int] = {}
        self.density_centres: Dict[QuadTreeCell, Tuple[float, float]] = {}
        self.spatial_features: Dict[QuadTreeCell, Tuple[float, float, float, float]] = {}
        self.structure_features: Dict[QuadTreeCell, Tuple[float, float]] = {}
        self.feature_vectors: Dict[QuadTreeCell, np.ndarray] = {}

        self._precompute()

    def _precompute(self) -> None:
        """
        执行特征预计算的完整流程。
        1. 统计轨迹相关信息（数量、交集、覆盖等）；
        2. 计算空间特征（中心位置、密度中心偏移）；
        3. 计算结构特征（层级、编码）；
        4. 拼接成最终的特征向量。
        """
        self._compute_trajectory_statistics()
        self._compute_spatial_features()
        self._compute_structure_features()
        self._assemble_feature_vectors()

    def _compute_trajectory_statistics(self) -> None:
        """
        计算轨迹统计特征。
        
        计算内容：
        1. N_ee: 被扩大元素覆盖的轨迹数量
        2. N_int, N_cov: 与轨迹 MBR 的交集、覆盖数量
        3. density_center: 轨迹密度中心
        """
        # 初始化计数器
        self.ee_counts = {cell: len(cell.trajectories) for cell in self.target_cells}
        self.intersect_counts = {cell: 0 for cell in self.target_cells}
        self.cover_counts = {cell: 0 for cell in self.target_cells}
        self.density_centres = {}

        # 缓存所有轨迹 MBR
        trajectory_mbrs = self._cache_trajectory_mbrs()

        # 计算交集和覆盖
        self._compute_intersections_and_covers(trajectory_mbrs)

        # 计算密度中心
        self._compute_density_centers(trajectory_mbrs)

        # 最大 ee_count，用于归一化
        self.max_ee_count = max(max(self.ee_counts.values()), 1)

    def _cache_trajectory_mbrs(self) -> Dict[int, Tuple[float, float, float, float]]:
        """缓存所有轨迹的 MBR。"""
        trajectory_mbrs = {}
        for cell in self.target_cells:
            for tid, bbox in cell.trajectory_mbrs.items():
                trajectory_mbrs[tid] = (bbox.min_x, bbox.min_y, bbox.max_x, bbox.max_y)
        return trajectory_mbrs

    def _compute_intersections_and_covers(
        self,
        trajectory_mbrs: Dict[int, Tuple[float, float, float, float]]
    ) -> None:
        """计算单元格与轨迹的交集和覆盖关系。"""
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
                    self.cover_counts[cell] += 1

    def _compute_density_centers(
        self,
        trajectory_mbrs: Dict[int, Tuple[float, float, float, float]]
    ) -> None:
        """计算每个单元格的轨迹密度中心。"""
        for cell in self.target_cells:
            tids = cell.trajectories

            if not tids:
                # 无轨迹使用 cell 几何中心
                self.density_centres[cell] = cell.get_center()
                continue

            centers = []
            for tid in tids:
                if tid not in trajectory_mbrs:
                    continue
                tminx, tminy, tmaxx, tmaxy = trajectory_mbrs[tid]
                centers.append(((tminx + tmaxx) / 2, (tminy + tmaxy) / 2))

            if centers:
                arr = np.array(centers)
                self.density_centres[cell] = (float(arr[:, 0].mean()), float(arr[:, 1].mean()))
            else:
                self.density_centres[cell] = cell.get_center()

    def _compute_spatial_features(self) -> None:
        """
        计算空间特征。
        
        特征包括：
        1. Δx, Δy: 单元格中心相对于整个空间中心的偏移
        2. Δdx, Δdy: 轨迹密度中心相对扩大元素中心的归一化偏移
        """
        qbox = self.quadtree.bbox

        # 整个空间中心和尺寸
        space_cx = (qbox.min_x + qbox.max_x) / 2
        space_cy = (qbox.min_y + qbox.max_y) / 2
        width = qbox.max_x - qbox.min_x
        height = qbox.max_y - qbox.min_y

        temp_den_offsets = []
        
        for cell in self.target_cells:
            # 计算单元格中心相对于空间中心的偏移
            cx, cy = cell.get_center()
            dx_pos = (cx - space_cx) / (width + self.EPSILON)
            dy_pos = (cy - space_cy) / (height + self.EPSILON)

            # 计算扩大元素中心
            eb = cell.get_enlarged_element_bbox(self.alpha, self.beta)
            e_cx = (eb.min_x + eb.max_x) / 2
            e_cy = (eb.min_y + eb.max_y) / 2
            e_w = eb.max_x - eb.min_x
            e_h = eb.max_y - eb.min_y

            # 计算密度中心相对于扩大元素中心的偏移
            dcenter = self.density_centres[cell]
            dx_den = (dcenter[0] - e_cx) / (e_w + self.EPSILON)
            dy_den = (dcenter[1] - e_cy) / (e_h + self.EPSILON)

            self.spatial_features[cell] = (dx_pos, dy_pos, dx_den, dy_den)
            temp_den_offsets.append([dx_den, dy_den])

        # Z-Score 归一化参数
        den_arr = np.array(temp_den_offsets)
        self.den_mean = np.mean(den_arr, axis=0)
        self.den_std = np.std(den_arr, axis=0) + self.EPSILON

    def _compute_structure_features(self) -> None:
        """
        计算单元格的结构特征。
        1. S: 单元面积的对数尺度；
        2. q_code: 最大层级下的编码归一化值；
        """
        max_code_value = (4 ** (self.quadtree.max_level + 1) - 1) // 3

        all_log_sizes = []
        for cell in self.target_cells:
            w, h = cell.bbox.max_x - cell.bbox.min_x, cell.bbox.max_y - cell.bbox.min_y
            all_log_sizes.append(np.log(w * h + 1e-8))

        self.ls_min, self.ls_max = min(all_log_sizes), max(all_log_sizes)

        for i, cell in enumerate(self.target_cells):
            log_size = all_log_sizes[i]
            # Min-Max 归一化
            size_norm = (log_size - self.ls_min) / (self.ls_max - self.ls_min + 1e-8)
            code_norm = cell.get_quadrant_code(self.quadtree.max_level) / (max_code_value + 1e-8)
            self.structure_features[cell] = (size_norm, code_norm)

    def _assemble_feature_vectors(self) -> None:
        """
        将统计特征、空间特征和结构特征拼接为完整特征向量。
        
        特征顺序：
        1. 密度统计 (3): N_ee_log, N_ee/N_int, N_cov/N_int
        2. 距离 (1): d_center
        3. 空间位置 (4): Δx, Δy, Δdx_norm, Δdy_norm
        4. 结构 (2): log_size, quad_code
        """
        for cell in self.target_cells:
            ee = self.ee_counts[cell]
            inter = self.intersect_counts[cell]
            cov = self.cover_counts[cell]

            # 1. 密度统计特征
            f_ee_log = np.log1p(ee) / np.log1p(self.max_ee_count)
            f_rel_ee = np.sqrt(ee / (inter + self.EPSILON))
            f_rel_cov = np.sqrt(cov / (inter + self.EPSILON))

            # 2. 距离特征
            cx, cy = cell.get_center()
            dcx, dcy = self.density_centres[cell]
            dist = np.sqrt((dcx - cx) ** 2 + (dcy - cy) ** 2)
            f_dist_log = np.log1p(dist * self.DISTANCE_SCALE)

            # 3. 空间特征
            dx_p, dy_p, dx_d, dy_d = self.spatial_features[cell]
            f_dx_den = (dx_d - self.den_mean[0]) / self.den_std[0]
            f_dy_den = (dy_d - self.den_mean[1]) / self.den_std[1]

            # 4. 结构特征
            f_size, f_code = self.structure_features[cell]

            features = [
                f_ee_log, f_rel_ee, f_rel_cov,      # 密度 (3)
                f_dist_log,                         # 距离 (1)
                dx_p, dy_p, f_dx_den, f_dy_den,     # 空间 (4)
                f_size, f_code,                     # 结构 (2)
            ]

            self.feature_vectors[cell] = np.array(features, dtype=np.float32)

    def get_features(
        self,
        cell: QuadTreeCell,
        prev_cell: Optional[QuadTreeCell],
        visited_ratio: float,
        visited_cells: set
    ) -> np.ndarray:
        """
        动态构建包含遍历特征的完整向量。
        
        参数:
            cell: 当前单元格
            prev_cell: 上一个访问的单元格
            visited_ratio: 已访问比例
            visited_cells: 已访问单元格集合
            
        返回:
            完整特征向量 (14维)
        """
        base = self.feature_vectors.get(cell)
        if base is None:
            return np.zeros(self.state_dimensions, dtype=np.float32)

        # 计算动态特征
        dx_rel, dy_rel = self._compute_relative_position(cell, prev_cell)
        f_nb_heat = self._compute_neighbor_heat(cell, visited_cells)

        # 拼接: static(10) + dynamic(4)
        dynamic_features = [visited_ratio, dx_rel, dy_rel, f_nb_heat]
        result = np.concatenate([base, dynamic_features]).astype(np.float32)

        return result

    def _compute_relative_position(
        self,
        cell: QuadTreeCell,
        prev_cell: Optional[QuadTreeCell]
    ) -> Tuple[float, float]:
        """计算相对于上一个访问节点的位移。"""
        if not prev_cell:
            return 0.0, 0.0

        c1x, c1y = cell.get_center()
        c2x, c2y = prev_cell.get_center()
        qbox = self.quadtree.bbox
        
        dx_rel = (c1x - c2x) / (qbox.max_x - qbox.min_x)
        dy_rel = (c1y - c2y) / (qbox.max_y - qbox.min_y)
        
        return dx_rel, dy_rel

    def _compute_neighbor_heat(
        self,
        cell: QuadTreeCell,
        visited_cells: set
    ) -> float:
        """计算邻居未访问轨迹数量（热度）。"""
        raw_neighbors = self.quadtree.get_eight_neighbor_cells(cell)
        total_nb_trajs = 0
        
        for nb in raw_neighbors:
            if not nb or nb.muted or nb in visited_cells:
                continue
            total_nb_trajs += self.ee_counts.get(nb, 0)
        
        # 归一化
        max_possible = self.max_ee_count * self.MAX_NEIGHBORS
        f_nb_heat = np.log1p(total_nb_trajs) / np.log1p(max_possible)
        
        return float(f_nb_heat)

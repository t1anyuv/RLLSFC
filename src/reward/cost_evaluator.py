"""
训练与评估阶段共享的奖励 / 成本评估工具。
"""
from typing import Iterable, Tuple

import numpy as np

from src.indexing.quadtree_cell import QuadTreeCell
from src.indexing.quadtree_index import QuadTreeIndex


class TraversalCostEvaluator:
    """遍历成本评估器。
    
    结合几何距离和轨迹相似度计算强化学习奖励。
    """
    
    # 常量定义
    DEFAULT_TAU_LOC = 1.0
    DEFAULT_TAU_SCAN = 0.1
    GLOBAL_REWARD_SCALE = 10.0

    def __init__(
        self,
        quadtree: QuadTreeIndex,
        tau_loc: float = DEFAULT_TAU_LOC,
        tau_scan: float = DEFAULT_TAU_SCAN
    ):
        """初始化成本评估器。
        
        参数:
            quadtree: 四叉树索引
            tau_loc: 定位成本权重
            tau_scan: 扫描成本权重
        """
        self.quadtree = quadtree
        self.tau_loc = tau_loc
        self.tau_scan = tau_scan

    @staticmethod
    def _distance(cell_a: QuadTreeCell, cell_b: QuadTreeCell) -> float:
        centroid_a = cell_a.get_center()
        centroid_b = cell_b.get_center()
        return float(np.linalg.norm(np.array(centroid_a) - np.array(centroid_b)))

    def proximity_reward(
        self,
        cell_a: QuadTreeCell,
        cell_b: QuadTreeCell,
        normaliser: float
    ) -> float:
        """计算邻近性奖励。
        
        依据单元格中心距离计算邻近性奖励，并归一化到 [0,1]。
        
        参数:
            cell_a: 第一个单元格
            cell_b: 第二个单元格
            normaliser: 归一化因子（通常为最大距离）
            
        返回:
            归一化的邻近性奖励 [0, 1]
        """
        distance = self._distance(cell_a, cell_b)
        normalised = min(distance / normaliser, 1.0)
        return 1.0 - normalised

    def jaccard_similarity(self, cell_a: QuadTreeCell, cell_b: QuadTreeCell) -> float:
        """
        计算两个单元格的 Jaccard 相似度。
        
        基于轨迹集合的交集和并集计算相似度。
        使用轨迹到单元格的索引映射优化性能。
        
        参数:
            cell_a: 第一个单元格
            cell_b: 第二个单元格
            
        返回:
            Jaccard 相似度 [0, 1]
        """
        # 尝试使用索引映射优化
        trajectory_to_cells = getattr(self.quadtree, 'trajectory_to_cells', None)
        trajectory_mbrs = getattr(self.quadtree, 'trajectory_mbrs', None)

        if trajectory_to_cells is not None and trajectory_mbrs is not None:
            return self._jaccard_with_index(cell_a, cell_b, trajectory_mbrs)
        else:
            return self._jaccard_fallback(cell_a, cell_b)

    def _jaccard_with_index(
        self,
        cell_a: QuadTreeCell,
        cell_b: QuadTreeCell,
        trajectory_mbrs: dict
    ) -> float:
        """使用索引映射计算 Jaccard 相似度。"""
        trajectories_a = set()
        trajectories_b = set()

        # 收集 cell_a 的轨迹及其与 cell_b 的相交轨迹
        for trajectory_id in cell_a.trajectories:
            trajectories_a.add(trajectory_id)
            trajectory_bbox = trajectory_mbrs.get(trajectory_id)
            if trajectory_bbox and cell_b.bbox.intersects(trajectory_bbox):
                trajectories_b.add(trajectory_id)

        # 收集 cell_b 的轨迹及其与 cell_a 的相交轨迹
        for trajectory_id in cell_b.trajectories:
            trajectories_b.add(trajectory_id)
            trajectory_bbox = trajectory_mbrs.get(trajectory_id)
            if trajectory_bbox and cell_a.bbox.intersects(trajectory_bbox):
                trajectories_a.add(trajectory_id)

        return self._compute_jaccard(trajectories_a, trajectories_b)

    def _jaccard_fallback(self, cell_a: QuadTreeCell, cell_b: QuadTreeCell) -> float:
        """回退方法：不使用索引映射计算 Jaccard 相似度。"""
        trajectories_a = set()
        trajectories_b = set()

        for trajectory_id in cell_a.trajectories:
            trajectories_a.add(trajectory_id)
            if self.quadtree.trajectory_intersects_cell(trajectory_id, cell_b):
                trajectories_b.add(trajectory_id)

        for trajectory_id in cell_b.trajectories:
            trajectories_b.add(trajectory_id)
            if self.quadtree.trajectory_intersects_cell(trajectory_id, cell_a):
                trajectories_a.add(trajectory_id)

        return self._compute_jaccard(trajectories_a, trajectories_b)

    @staticmethod
    def _compute_jaccard(set_a: set, set_b: set) -> float:
        """计算两个集合的 Jaccard 系数。"""
        if not set_a and not set_b:
            return 1.0

        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        
        if union == 0:
            return 0.0
        
        return intersection / union

    def step_reward(
        self,
        current_cell: QuadTreeCell,
        next_cell: QuadTreeCell,
        normaliser: float,
        proximity_weight: float = 0.5,
        similarity_weight: float = 0.5,
    ) -> float:
        """计算单步奖励。
        
        结合空间邻近性和轨迹相似度计算奖励。
        
        参数:
            current_cell: 当前单元格
            next_cell: 下一个单元格
            normaliser: 距离归一化因子
            proximity_weight: 邻近性权重
            similarity_weight: 相似度权重
            
        返回:
            加权奖励值
        """
        proximity_value = self.proximity_reward(current_cell, next_cell, normaliser)
        similarity_value = self.jaccard_similarity(current_cell, next_cell)
        return proximity_weight * proximity_value + similarity_weight * similarity_value

    def query_cost(self, intervals: Iterable[Tuple[int, int]]) -> float:
        """计算查询成本。
        
        基于区间数量与覆盖长度计算成本。
        
        参数:
            intervals: 查询区间列表 [(start, end), ...]
            
        返回:
            查询成本
        """
        intervals_list = list(intervals)
        m_value = len(intervals_list)
        length_value = sum(end - start for start, end in intervals_list)
        return self.tau_loc * m_value + self.tau_scan * length_value

    def global_reward(
        self,
        baseline_intervals: Iterable[Tuple[int, int]],
        learned_intervals: Iterable[Tuple[int, int]]
    ) -> float:
        """计算全局奖励。
        
        根据学习顺序相较于基线顺序的成本改进量生成全局奖励。
        
        参数:
            baseline_intervals: 基线顺序的查询区间
            learned_intervals: 学习顺序的查询区间
            
        返回:
            全局奖励（放大后）
        """
        baseline_cost = self.query_cost(baseline_intervals)
        learned_cost = self.query_cost(learned_intervals)
        improvement = baseline_cost - learned_cost
        return max(0.0, improvement) * self.GLOBAL_REWARD_SCALE

from typing import List, Optional, Tuple

import numpy as np

from src.core.bounding_box import SpatialBoundingBox
from src.features.state_feature_builder import StateFeatureBuilder
from src.indexing.quadtree_cell import QuadTreeCell
from src.indexing.quadtree_index import QuadTreeIndex
from src.reward.cost_evaluator import TraversalCostEvaluator


class TraversalEnvironment:
    """
    强化学习环境：
    学习 QuadTree 单元格的访问顺序，以最小化 TShape 索引的查询代价。
    """

    def __init__(
            self,
            quadtree: QuadTreeIndex,
            cost_evaluator: TraversalCostEvaluator,
            reference_queries: Optional[List[SpatialBoundingBox]] = None,
            alpha: int = 2,
            beta: int = 2,
            exclude_muted_cells: bool = True,
            baseline_include_muted: bool = True,
            local_reward_weight: float = 0.3,
            global_reward_weight: float = 0.7,
            global_reward_num_evals: int = 1,
    ):
        """初始化强化学习环境。"""

        self.quadtree = quadtree
        """表示空间划分的四叉树索引结构。"""

        self.cost_evaluator = cost_evaluator
        """ 用于计算轨迹遍历顺序对应查询成本或奖励的评估器。"""

        if reference_queries is None:
            raise ValueError("必须提供 reference_queries，请使用预处理好的查询数据集")
        self.reference_queries = reference_queries
        """一组用于离线评估和计算全局奖励的参考查询区域。"""

        self.alpha = alpha
        self.beta = beta

        self.exclude_muted_cells = exclude_muted_cells
        """是否在学习过程中排除哑节点。"""

        self.baseline_include_muted = baseline_include_muted
        """在计算基线顺序时是否包含哑节点。"""

        self.local_reward_weight = local_reward_weight
        """局部奖励权重。"""

        self.global_reward_weight = global_reward_weight
        """全局奖励权重。"""

        self.global_reward_scale = 10.0
        """全局奖励缩放倍数，用于增大全局奖励的数量级（内部默认值）。"""

        self.global_reward_clip_max = 100.0
        """全局奖励裁剪上界"""

        self.global_reward_clip_min = -100.0
        """全局奖励裁剪下界"""

        self.global_reward_num_evals = global_reward_num_evals
        """全局奖励计算次数"""

        self._global_eval_checkpoints = set()
        """全局奖励评估检查点（步数集合）"""

        self.all_cells: List[QuadTreeCell] = self.quadtree.get_active_cells() if exclude_muted_cells \
            else list(self.quadtree.all_cells.values())
        """可访问的单元格"""

        self.cell_to_index = {cell: index for index, cell in enumerate(self.all_cells)}
        """单元格到索引值的映射"""

        self.num_cells = len(self.all_cells)
        """单元格数量"""

        self.max_distance = self._estimate_max_distance()
        """最大距离，用于归一化"""

        self.feature_builder = StateFeatureBuilder(quadtree, alpha, beta, self.all_cells)
        """特征构造器"""

        self.current_cell: Optional[QuadTreeCell] = None
        """当前访问的单元格"""

        self.prev_cell: Optional[QuadTreeCell] = None
        """上一访问的单元格"""

        self.visited_cells: set = set()
        """已访问的单元格"""

        self.visited_order: List[QuadTreeCell] = []
        """已访问的顺序"""

        self.available_actions: List[int] = []
        """可用动作集合"""

        self.current_episode = 0
        """当前的 episode"""

        # 初始化全局奖励计算的缓存组件
        self._init_global_reward_cache()

    def _init_global_reward_cache(self):
        """构建全局奖励评估缓存，避免重复创建 encoder / evaluator。"""
        from src.evaluation.traversal_evaluator import TraversalPerformanceEvaluator
        from src.indexing.traversal_encoder import TraversalOrderEncoder

        # 创建编码器并预计算基线顺序
        self._cached_encoder = TraversalOrderEncoder(self.quadtree, self.alpha, self.beta)
        self._cached_baseline_order = self._cached_encoder.z_curve_order(include_muted=self.baseline_include_muted)

        # 性能评估器
        self._cached_evaluator = TraversalPerformanceEvaluator(
            self.quadtree,
            self._cached_encoder,
            self.cost_evaluator,
            reference_queries=self.reference_queries,
            baseline_include_muted=self.baseline_include_muted,
        )

        # baseline 成本预计算
        self._precompute_baseline_results()

    def _precompute_baseline_results(self):
        """预计算 baseline 在所有参考查询上的成本，用于归一化全局奖励。"""

        self._cached_baseline_costs = []
        self._cached_baseline_interval_counts_raw = []
        self._cached_baseline_interval_counts_merged = []

        for query in self.reference_queries:
            nodes = self._cached_evaluator.tshape_search(query, self._cached_baseline_order,
                                                         skip_muted=not self.baseline_include_muted)[0]
            intervals = self._cached_evaluator.compute_query_intervals(self._cached_baseline_order, nodes)
            cost = self.cost_evaluator.query_cost(intervals)

            self._cached_baseline_costs.append(cost)
            self._cached_baseline_interval_counts_raw.append(len(nodes))
            self._cached_baseline_interval_counts_merged.append(len(intervals))

    @property
    def state_dimension(self) -> int:
        """状态向量维度（由 StateFeatureBuilder 决定）。"""
        return self.feature_builder.state_dimensions

    @property
    def action_dimension(self) -> int:
        """动作空间维度（对应单元格数量）。"""
        return self.num_cells

    def _estimate_max_distance(self) -> float:
        """计算四叉树空间边界对角线长度，用于距离归一化。"""
        width = self.quadtree.bbox.max_x - self.quadtree.bbox.min_x
        height = self.quadtree.bbox.max_y - self.quadtree.bbox.min_y
        return float(np.sqrt(width ** 2 + height ** 2))

    def reset(self, start_cell: Optional[QuadTreeCell] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        重置环境状态。

        参数：
            start_cell : Optional[QuadTreeCell]
                初始遍历单元格，默认从根单元格开始。

        返回：
            Tuple[np.ndarray, np.ndarray]
                - 状态特征向量；
                - 动作掩码（表示可行动作）。
        """
        self.current_cell = start_cell or self.quadtree.root
        self.visited_cells = {self.current_cell}
        self.visited_order = [self.current_cell]

        self.available_actions = self._compute_available_actions()
        self.current_episode += 1
        
        # 计算全局奖励评估检查点
        self._global_eval_checkpoints = set()
        if self.global_reward_num_evals > 0:
            for i in range(1, self.global_reward_num_evals + 1):
                checkpoint_step = int(self.num_cells * i / self.global_reward_num_evals)
                self._global_eval_checkpoints.add(checkpoint_step)

        return self._build_state()

    def _compute_available_actions(self) -> List[int]:
        """
        计算当前单元格下的可用动作（即下一步可访问的单元格）。

        约束: 所有动作必须是非哑节点。

        1. 八方向邻居:
            - 直接加入所有非哑、未访问的邻居节点。

        2. 父节点(向上逐层搜索):
            - 若当前节点以及邻居节点的父节点全部为哑节点，则继续向上寻找更高层祖先；
            - 一旦某一层中出现非哑且未访问的父节点，则加入并停止向上搜索；
            - 若某一层出现非哑但已访问的父节点，则此次向上搜索终止，不再继续寻找祖先。

        3. 子节点(向下逐层搜索):
            - 若当前节点及邻居节点的所有子节点全部为哑节点，则继续向下扩展；
            - 一旦某一层中出现非哑且未访问的子节点，则加入并停止向下搜索；
            - 若某一层出现非哑但已访问的子节点，则此次向下搜索终止，不再继续寻找后代。
        """

        if not self.current_cell:
            return []

        candidate_cells = set()

        quadtree = self.quadtree
        visited_cells = self.visited_cells
        cur = self.current_cell

        # ----------------------------------------------------------------------
        # Step 1: 八方向邻居 —— 直接加入非哑、未访问邻居
        # ----------------------------------------------------------------------
        raw_neighbors = quadtree.get_eight_neighbor_cells(cur)
        valid_neighbors = []

        for nb in raw_neighbors:
            if not nb or nb.muted or nb in visited_cells:
                continue
            candidate_cells.add(nb)
            valid_neighbors.append(nb)

        # 基于当前节点 + 有效邻居节点进行父子搜索
        nodes = [cur] + valid_neighbors

        # ----------------------------------------------------------------------
        # Step 2: 逐层向上找父节点
        # ----------------------------------------------------------------------
        parent_layer = {node.parent for node in nodes if node.parent is not None}

        while parent_layer:
            # 当前层可用（非哑且未访问）的父节点
            active_parents = [
                p for p in parent_layer
                if p and (not p.muted) and (p not in visited_cells)
            ]

            # 若当前层存在可用父节点，则加入并终止向上搜索
            if active_parents:
                candidate_cells.update(active_parents)
                break

            # ---------------------
            # 若本层没有可用父节点，继续向上寻找
            # ---------------------
            next_layer = {p.parent for p in parent_layer if p}
            parent_layer = next_layer

        # ----------------------------------------------------------------------
        # Step 3: 逐层向下找子节点
        # ----------------------------------------------------------------------
        child_layer = {child for node in nodes for child in node.children if child}

        while child_layer:
            # 当前层可用（非哑且未访问）的子节点
            active_children = [
                c for c in child_layer
                if c and (not c.muted) and (c not in visited_cells)
            ]

            # 若当前层存在可用子节点，则加入并终止向下搜索
            if active_children:
                candidate_cells.update(active_children)
                break

            # ---------------------
            # 若本层没有可用子节点，继续向下寻找
            # ---------------------
            next_layer = {gc for c in child_layer for gc in c.children if gc}
            child_layer = next_layer

        # ----------------------------------------------------------------------
        # Step 4: fallback，全局寻找最近的有效节点
        # ----------------------------------------------------------------------
        if not candidate_cells:
            remaining = [cell for cell in quadtree.get_all_cells() if (not cell.muted) and (cell not in visited_cells)]
            if remaining:
                # 按空间距离最近
                cx, cy = self.current_cell.get_center()
                fallback_cell = min(
                    remaining,
                    key=lambda c: (c.get_center()[0] - cx) ** 2 + (c.get_center()[1] - cy) ** 2,
                )
                if fallback_cell:
                    candidate_cells.add(fallback_cell)

        # ----------------------------------------------------------------------
        # 映射成动作索引
        # ----------------------------------------------------------------------
        action_indices = [self.cell_to_index[cell] for cell in candidate_cells if cell in self.cell_to_index]

        return action_indices

    def _build_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        构建状态特征向量和动作掩码。

        返回：
            Tuple[np.ndarray, np.ndarray]
                - feature_vector: 当前状态的数值特征；
                - action_mask: 合法动作的掩码（1 表示可选，0 表示不可选）。
        """
        # 计算已访问比例，并生成特征向量与动作掩码。
        visited_ratio = len(self.visited_cells) / max(1, self.num_cells)
        feature_vector = self.feature_builder.get_features(self.current_cell,
                                                           self.prev_cell,
                                                           visited_ratio,
                                                           self.visited_cells)

        action_mask = np.zeros(self.num_cells, dtype=np.float32)
        for action_index in self.available_actions:
            action_mask[action_index] = 1.0

        return feature_vector, action_mask

    def _compute_global_reward(self, learned_order: List[QuadTreeCell]) -> Tuple[float, float]:
        """
        计算完整序列的全局奖励。
        
        返回归一化后的全局奖励以及原始成本差值。
        """
        learned_costs = []

        for query_bbox in self.reference_queries:
            learned_nodes = self._cached_evaluator.tshape_search(query_bbox, learned_order, skip_muted=True)[0]
            learned_intervals = self._cached_evaluator.compute_query_intervals(learned_order, learned_nodes)
            learned_cost = self.cost_evaluator.query_cost(learned_intervals)
            learned_costs.append(learned_cost)

        # 使用缓存的 baseline 结果计算比较指标
        baseline_avg_cost = float(np.mean(self._cached_baseline_costs))
        learned_avg_cost = float(np.mean(learned_costs))

        if baseline_avg_cost > 0:
            improvement_percent = ((baseline_avg_cost - learned_avg_cost) / baseline_avg_cost) * 100.0
        else:
            improvement_percent = 0.0

        raw_cost_diff = baseline_avg_cost - learned_avg_cost
        global_reward = improvement_percent * self.global_reward_scale

        return global_reward, raw_cost_diff
    

    def step(self, action: int) -> Tuple[np.ndarray, np.ndarray, float, bool, dict]:
        """
        执行一步环境更新。

        参数：
            action : int
                当前动作（单元格索引）。

        返回：
            Tuple[np.ndarray, np.ndarray, float, bool, dict]
                - 新状态特征；
                - 动作掩码；
                - 奖励值；
                - 是否终止；
                - 额外信息（如访问节点数，全局奖励等）。
        """

        # --------- 若所有 cell 已访问 ---------
        if len(self.visited_cells) >= self.num_cells:
            s, m = self._build_state()
            return s, m, 0.0, True, {
                "visited_count": self.num_cells,
                "total_cells": self.num_cells,
            }

        # --------- 非法动作 ---------
        if action not in self.available_actions:
            s, m = self._build_state()
            return s, m, -0.1, False, {"error": "invalid_action"}

        # --------- 移动 & 局部奖励 ---------
        next_cell = self.all_cells[action]
        local_reward = self.cost_evaluator.step_reward(
            self.current_cell,
            next_cell,
            self.max_distance,
            proximity_weight=0.5,
            similarity_weight=0.5,
        )

        self.prev_cell = self.current_cell
        self.current_cell = next_cell
        self.visited_cells.add(next_cell)
        self.visited_order.append(next_cell)
        self.available_actions = self._compute_available_actions()

        done = len(self.visited_cells) >= self.num_cells
        current_step = len(self.visited_order)

        # --------- 全局奖励---------
        raw_cost_diff = 0.0
        global_trigger = False

        # 检查是否到达全局奖励评估检查点
        if current_step in self._global_eval_checkpoints:
            global_reward, raw_cost_diff = self._compute_global_reward(self.visited_order)
            global_reward_cliped = float(np.clip(global_reward,
                                                 self.global_reward_clip_min,
                                                 self.global_reward_clip_max))
            global_trigger = True
        else:
            global_reward_cliped = 0.0

        reward = self.local_reward_weight * local_reward + self.global_reward_weight * global_reward_cliped

        info = {
            "visited_count": len(self.visited_cells),
            "total_cells": self.num_cells,
            "local_reward": local_reward,
            "global_reward": global_reward_cliped,
            "raw_cost_diff": raw_cost_diff,
            "global_reward_triggered": global_trigger,
        }

        next_state, next_action_mask = self._build_state()
        return next_state, next_action_mask, reward, done, info

    def learned_order(self) -> List[QuadTreeCell]:
        """返回当前 episode 的访问顺序。"""
        return self.visited_order.copy()

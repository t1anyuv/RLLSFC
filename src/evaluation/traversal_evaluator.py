"""
用于基于查询负载对比 RL 学到的遍历顺序与 Z 曲线基线。
"""
from typing import List, Optional, Tuple, Set, Dict, Any

import numpy as np

from src.core.bounding_box import SpatialBoundingBox
from src.indexing.quadtree_cell import QuadTreeCell
from src.indexing.quadtree_index import QuadTreeIndex
from src.indexing.traversal_encoder import TraversalOrderEncoder
from src.reward.cost_evaluator import TraversalCostEvaluator
from src.utils.signature import compute_query_signature


class TraversalPerformanceEvaluator:

    def __init__(
            self,
            quadtree: QuadTreeIndex,
            encoder: TraversalOrderEncoder,
            reward_calculator: TraversalCostEvaluator,
            reference_queries: Optional[List[SpatialBoundingBox]] = None,
            baseline_include_muted: bool = False,
    ):
        self.quadtree = quadtree
        self.encoder = encoder
        self.cost_evaluator = reward_calculator
        self.alpha = encoder.alpha
        self.beta = encoder.beta
        self.reference_queries = reference_queries
        self.baseline_include_muted = baseline_include_muted

    def tshape_search(
            self,
            query_bbox: SpatialBoundingBox,
            traversal_order: List[QuadTreeCell],
            skip_muted: bool = True
    ) -> Tuple[List[QuadTreeCell], Set[int]]:
        """
        基于 TShape 搜索算法生成与查询相关的节点集合。
        
        按层搜索逻辑：
        1. 从第一层开始，逐层搜索
        2. 如果某个节点被覆盖，则加入该节点的所有子节点到结果中
        3. 如果某个节点相交，则：
           - 加入这个节点到结果中（如果未被屏蔽）
           - 将该节点的四个子节点入队列（用于下一层处理）
        4. 如果节点被屏蔽，则不加入结果，但其他操作不变（继续处理子节点）
        
        参数:
            query_bbox: 查询边界框
            traversal_order: 遍历顺序列表
            skip_muted: 是否跳过被屏蔽的节点（不加入结果）
        
        返回:
            Tuple[List[QuadTreeCell], Set[int]]: (命中的单元格列表, 经过签名过滤后的候选轨迹ID集合)
        """

        cell_to_code = {cell: idx for idx, cell in enumerate(traversal_order)}
        results_cells = set()
        candidate_traj_ids = set()

        quadtree = self.quadtree
        global_alpha, global_beta = self.alpha, self.beta

        current_level_cells = {quadtree.root}
        next_level_cells = set()

        current_level = 0

        while current_level <= quadtree.max_level and current_level_cells:
            # 处理当前层的所有节点
            for cell in current_level_cells:
                # 处理无效节点：只添加子节点
                if skip_muted and cell.muted:
                    self._add_children(cell, next_level_cells)
                    continue

                # 计算扩大元素边界框
                enlarged_bbox = cell.get_enlarged_element_bbox(global_alpha, global_beta)

                # 判断查询区域与扩大元素的关系
                if query_bbox.contains(enlarged_bbox):
                    # 被覆盖：加入该节点的所有子节点到结果中
                    self._collect_all_descendants(cell, cell_to_code, results_cells, candidate_traj_ids, skip_muted)
                elif query_bbox.intersects(enlarged_bbox):
                    # 相交：加入这个节点，然后将四个子节点加入下一层队列
                    query_sig = compute_query_signature(global_alpha, global_beta, cell, query_bbox)
                    for tid in cell.trajectories:
                        traj_sig = cell.signatures.get(tid, 0)
                        if (traj_sig & query_sig) != 0:
                            candidate_traj_ids.add(tid)
                        else:
                            pass
                    if cell in cell_to_code:
                        results_cells.add(cell)

                    # 加入子节点
                    self._add_children(cell, next_level_cells)

            # 移动到下一层
            current_level += 1
            current_level_cells = next_level_cells
            next_level_cells = set()

        return list(results_cells), candidate_traj_ids

    def tshape_search_debug(
            self,
            query_bbox: SpatialBoundingBox,
            traversal_order: List[QuadTreeCell],
            target_tid: int,
            skip_muted: bool = True
    ) -> Tuple[List[QuadTreeCell], Set[int]]:

        cell_to_code = {cell: idx for idx, cell in enumerate(traversal_order)}
        results_cells = set()
        candidate_traj_ids = set()

        quadtree = self.quadtree
        global_alpha, global_beta = self.alpha, self.beta

        current_level_cells = {quadtree.root}
        next_level_cells = set()

        current_level = 0

        while current_level <= quadtree.max_level and current_level_cells:
            # 处理当前层的所有节点
            for cell in current_level_cells:
                if target_tid in cell.trajectories and target_tid != -1:
                    print(f"[DEBUG_SEARCH] TID: {target_tid} check: Muted={cell.muted}, skip_muted={skip_muted}")
                # 处理无效节点：只添加子节点
                if skip_muted and cell.muted:
                    self._add_children(cell, next_level_cells)
                    continue

                # 计算扩大元素边界框
                enlarged_bbox = cell.get_enlarged_element_bbox(global_alpha, global_beta)

                # 判断查询区域与扩大元素的关系
                if query_bbox.contains(enlarged_bbox):
                    if target_tid in cell.trajectories and target_tid != -1:
                        print(f"[DEBUG_SEARCH] TID: {target_tid} is in a FULLY COVERED Cell (Level {cell.level})")
                        print(f"  - Query BBox: {query_bbox}")
                        print(f"  - Enlarged BBox: {enlarged_bbox}")
                    # 被覆盖：加入该节点的所有子节点到结果中
                    self._collect_all_descendants(cell, cell_to_code, results_cells, candidate_traj_ids, skip_muted)
                elif query_bbox.intersects(enlarged_bbox):
                    # 相交：加入这个节点，然后将四个子节点加入下一层队列
                    query_sig = compute_query_signature(global_alpha, global_beta, cell, query_bbox)
                    for tid in cell.trajectories:
                        if tid == target_tid and target_tid != -1:
                            query_sig = compute_query_signature(global_alpha, global_beta, cell, query_bbox)
                            traj_sig = cell.signatures.get(tid, 0)
                            match = (traj_sig & query_sig) != 0
                            print(f"[DEBUG_SEARCH] TID: {tid} Found in Cell: {cell.code} (Level {cell.level})")
                            print(f"  - Traj Sig:  {bin(traj_sig)}")
                            print(f"  - Query Sig: {bin(query_sig)}")
                            print(f"  - Match:     {match}")
                            print(f"  - Cell Alpha/Beta: {cell.alpha}/{cell.beta}")
                        traj_sig = cell.signatures.get(tid, 0)
                        if (traj_sig & query_sig) != 0:
                            candidate_traj_ids.add(tid)
                        else:
                            pass
                    if cell in cell_to_code:
                        results_cells.add(cell)

                    # 加入子节点
                    self._add_children(cell, next_level_cells)

            # 移动到下一层
            current_level += 1
            current_level_cells = next_level_cells
            next_level_cells = set()

        if target_tid not in candidate_traj_ids and target_tid != -1:
            print(f"[DEBUG_SEARCH] CRITICAL: TID {target_tid} never added to candidate_ids!")

        return list(results_cells), candidate_traj_ids

    def _add_children(self, cell, next_level_cells):
        if cell.level < self.quadtree.max_level:
            for child in cell.children:
                if child:
                    next_level_cells.add(child)

    def _collect_all_descendants(self, root_cell, cell_to_code, res_cells, res_trajs, skip_muted):
        """收集被覆盖分支下的所有 Cell 和轨迹 ID。"""
        stack = [root_cell]
        while stack:
            cell = stack.pop()

            # 记录有效数据
            if not (skip_muted and cell.muted):
                if cell in cell_to_code:
                    res_cells.add(cell)
                res_trajs.update(cell.trajectories)

            # 继续向下
            if cell.level < self.quadtree.max_level:
                for child in cell.children:
                    if child:
                        stack.append(child)

    def compute_hgs_score(
            self,
            learned_order: List[QuadTreeCell],
            test_queries: List[SpatialBoundingBox]
    ) -> Dict[str, Any]:
        """
        计算双重改进率及 HGS 综合得分。
        """
        # 1. 评估参考集
        ref_metrics = self.evaluate_final_order(learned_order)
        i_ref = ref_metrics['improvement_percent']

        # 2. 评估测试集
        original_queries = self.reference_queries
        self.reference_queries = test_queries

        test_metrics = self.evaluate_final_order(learned_order)
        i_test = test_metrics['improvement_percent']

        # 恢复原始参考集
        self.reference_queries = original_queries

        # 3. 计算 HGS 得分: I_test * (1 - Gap/Max)
        gap = abs(i_ref - i_test)
        max_val = max(i_ref, i_test, 1e-6)
        score = i_test * (1 - (gap / max_val))

        return {
            "i_ref": i_ref,
            "i_test": i_test,
            "hgs_score": score,
            "test_metrics": test_metrics
        }

    @staticmethod
    def compute_query_intervals(order: List[QuadTreeCell], nodes: List[QuadTreeCell]) -> List[Tuple[int, int]]:
        """计算查询对应的编码区间，将查询涉及的单元格合并为连续区间"""

        cell_to_code = {cell: idx for idx, cell in enumerate(order)}
        codes = sorted(cell_to_code[cell] for cell in nodes if cell in cell_to_code)
        if not codes:
            return []

        intervals: List[Tuple[int, int]] = []
        start = codes[0]
        end = codes[0]

        for code in codes[1:]:
            if code == end + 1:
                end = code
            else:
                # 当前区间中断，写入结果后开启新的区间。
                intervals.append((start, end + 1))
                start = code
                end = code
        intervals.append((start, end + 1))
        return intervals

    def compare_orders(self, baseline_order: List[QuadTreeCell],
                       learned_order: List[QuadTreeCell], num_queries: int = 100) -> dict:
        """在一批查询上复用搜索、区间合并和成本计算流程，评估两个顺序的差异。"""
        if self.reference_queries is None:
            raise ValueError("未提供 reference_queries，无法执行对比评估")
        queries = self.reference_queries

        baseline_costs = []
        learned_costs = []

        baseline_nodes_hit = []
        baseline_scan_counts = []

        learned_nodes_hit = []
        learned_scan_counts = []

        for query_bbox in queries:
            # baseline 如果包括了哑节点，就不应该搜索时跳过
            baseline_nodes = self.tshape_search(query_bbox, baseline_order,
                                                skip_muted=not self.baseline_include_muted)[0]

            learned_nodes = self.tshape_search(query_bbox, learned_order, skip_muted=True)[0]

            baseline_nodes_hit.append(len(baseline_nodes))
            learned_nodes_hit.append(len(learned_nodes))

            baseline_intervals = self.compute_query_intervals(baseline_order, baseline_nodes)
            learned_intervals = self.compute_query_intervals(learned_order, learned_nodes)

            baseline_scan_counts.append(len(baseline_intervals))
            learned_scan_counts.append(len(learned_intervals))

            baseline_cost = self.cost_evaluator.query_cost(baseline_intervals)
            learned_cost = self.cost_evaluator.query_cost(learned_intervals)

            baseline_costs.append(baseline_cost)
            learned_costs.append(learned_cost)

        avg_baseline_cost = float(np.mean(baseline_costs)) if baseline_costs else 0.0
        avg_learned_cost = float(np.mean(learned_costs)) if learned_costs else 0.0

        improvement = (
            (avg_baseline_cost - avg_learned_cost) / avg_baseline_cost * 100 if avg_baseline_cost > 0 else 0.0
        )

        avg_baseline_nodes_hit = np.mean(baseline_nodes_hit)
        avg_learned_nodes_hit = np.mean(learned_nodes_hit)

        avg_baseline_scan_counts = np.mean(baseline_scan_counts)
        avg_learned_scan_counts = np.mean(learned_scan_counts)

        total_baseline_cost = float(np.sum(baseline_costs)) if baseline_costs else 0.0
        total_learned_cost = float(np.sum(learned_costs)) if learned_costs else 0.0

        # Using the overall improvement keeps the reward signal consistent with evaluation metrics.
        global_reward = max(0.0, total_baseline_cost - total_learned_cost)

        return {
            "baseline_avg_cost": avg_baseline_cost,
            "learned_avg_cost": avg_learned_cost,
            "improvement_percent": improvement,
            "baseline_costs": baseline_costs,
            "learned_costs": learned_costs,
            "global_reward": global_reward,
            "baseline_nodes_hit": avg_baseline_nodes_hit,
            "baseline_scan_counts": avg_baseline_scan_counts,
            "learned_nodes_hit": avg_learned_nodes_hit,
            "learned_scan_counts": avg_learned_scan_counts,
        }

    def evaluate_final_order(self, learned_order: List[QuadTreeCell]) -> dict:
        """
        评估最终学习到的顺序与Z曲线baseline比较
        """
        baseline_order = self.encoder.z_curve_order(include_muted=self.baseline_include_muted)
        return self.compare_orders(baseline_order, learned_order)

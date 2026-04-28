from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from src.common import SpatialBoundingBox
from src.indexing.quadtree_cell import QuadTreeCell
from src.indexing.quadtree_index import QuadTreeIndex
from src.indexing.traversal_encoder import TraversalOrderEncoder
from src.common import TraversalCostEvaluator
from src.utils.signature import compute_query_signature

TraversalOrderLookup = Dict[QuadTreeCell, int]


class TraversalPerformanceEvaluator:
    def __init__(
        self,
        quadtree: QuadTreeIndex,
        encoder: TraversalOrderEncoder,
        reward_calculator: TraversalCostEvaluator,
        reference_queries: Optional[List[SpatialBoundingBox]] = None,
        quadcode_include_muted: bool = False,
        default_order_tail: Optional[List[QuadTreeCell]] = None,
    ):
        self.quadtree = quadtree
        self.encoder = encoder
        self.cost_evaluator = reward_calculator
        self.alpha = encoder.alpha
        self.beta = encoder.beta
        self.reference_queries = reference_queries
        self.quadcode_include_muted = quadcode_include_muted
        if default_order_tail is not None:
            self._default_order_tail: Optional[List[QuadTreeCell]] = list(default_order_tail)
        else:
            self._default_order_tail = self.encoder.z_curve_order(
                include_muted=self.quadcode_include_muted,
            )

    @staticmethod
    def _add_children(quadtree: QuadTreeIndex, cell: QuadTreeCell, next_level_cells: Set[QuadTreeCell]) -> None:
        if cell.level < quadtree.max_level:
            for child in cell.children:
                if child is not None:
                    next_level_cells.add(child)

    @staticmethod
    def _append_matching_rowkeys(
        *,
        cell: QuadTreeCell,
        row_key: int,
        query_sig: int,
        candidate_traj_ids: Set[int],
        row_intervals: List[Tuple[int, int]],
    ) -> None:
        matched = False
        for traj_id in cell.trajectories:
            traj_sig = cell.signatures.get(traj_id, 0)
            if traj_sig == 0 or (traj_sig & query_sig) == 0:
                continue
            candidate_traj_ids.add(traj_id)
            matched = True

        if matched:
            row_intervals.append((int(row_key), int(row_key) + 1))

    @staticmethod
    def _collect_subtree_trajectories(
        *,
        quadtree: QuadTreeIndex,
        cell: QuadTreeCell,
        candidate_traj_ids: Set[int],
        skip_muted: bool,
    ) -> None:
        stack = [cell]
        while stack:
            current = stack.pop()
            if skip_muted and current.muted:
                for child in current.children:
                    if child is not None:
                        stack.append(child)
                continue

            candidate_traj_ids.update(current.trajectories)
            if current.level < quadtree.max_level:
                for child in current.children:
                    if child is not None:
                        stack.append(child)

    def _build_order_lookup(
        self,
        traversal_order: Iterable[QuadTreeCell],
    ) -> TraversalOrderLookup:
        lookup: TraversalOrderLookup = {}
        next_order = 0

        for cell in traversal_order:
            if cell in lookup:
                continue
            lookup[cell] = next_order
            next_order += 1

        if self._default_order_tail is not None:
            for cell in self._default_order_tail:
                if cell in lookup:
                    continue
                lookup[cell] = next_order
                next_order += 1

        return lookup

    def _can_use_xz_coverage_intervals(self) -> bool:
        loader = self.encoder.get_order_loader()
        if loader is None or not loader.is_loaded():
            return False
        order_source = loader.get_order_source()
        contiguous = loader.get_effective_subtree_contiguous()
        return order_source is not None and "xz" in order_source.lower() and contiguous is True

    def search_quadcode_intervals(
        self,
        query_bbox: SpatialBoundingBox,
        traversal_order: List[QuadTreeCell],
        skip_muted: bool = True,
    ) -> Tuple[List[Tuple[int, int]], Set[int], int]:
        _ = traversal_order
        max_level = self.quadtree.max_level
        row_intervals: List[Tuple[int, int]] = []
        candidate_traj_ids: Set[int] = set()
        current_level_cells = {self.quadtree.root}
        next_level_cells: Set[QuadTreeCell] = set()

        while current_level_cells:
            for cell in current_level_cells:
                if skip_muted and cell.muted:
                    self._add_children(self.quadtree, cell, next_level_cells)
                    continue

                enlarged_bbox = cell.get_enlarged_element_bbox(self.alpha, self.beta)

                if query_bbox.contains(enlarged_bbox):
                    quad_start, quad_end = QuadTreeCell.subtree_leaf_quad_codes(
                        cell.quadrant_sequence,
                        max_level,
                    )
                    row_intervals.append((quad_start, quad_end))
                    self._collect_subtree_trajectories(
                        quadtree=self.quadtree,
                        cell=cell,
                        candidate_traj_ids=candidate_traj_ids,
                        skip_muted=skip_muted,
                    )
                elif query_bbox.intersects(enlarged_bbox):
                    query_sig = compute_query_signature(self.alpha, self.beta, cell, query_bbox)
                    if not cell.signatures:
                        self._add_children(self.quadtree, cell, next_level_cells)
                        continue

                    quad_code = cell.get_quadrant_code(max_level)
                    self._append_matching_rowkeys(
                        cell=cell,
                        row_key=quad_code,
                        query_sig=query_sig,
                        candidate_traj_ids=candidate_traj_ids,
                        row_intervals=row_intervals,
                    )
                    self._add_children(self.quadtree, cell, next_level_cells)

            current_level_cells = next_level_cells
            next_level_cells = set()

        merged = self.cost_evaluator.merge_intervals(row_intervals)
        return [(int(start), int(end)) for start, end in merged], candidate_traj_ids, 0

    def search_quadorder_intervals(
        self,
        query_bbox: SpatialBoundingBox,
        traversal_order: List[QuadTreeCell],
        skip_muted: bool = True,
        aligned: Optional[TraversalOrderLookup] = None,
    ) -> Tuple[List[Tuple[int, int]], Set[int], int]:
        order_lookup = aligned or self._build_order_lookup(traversal_order)
        can_use_xz_coverage = self._can_use_xz_coverage_intervals()
        order_loader = self.encoder.get_order_loader() if can_use_xz_coverage else None
        row_intervals: List[Tuple[int, int]] = []
        candidate_traj_ids: Set[int] = set()
        current_level_cells = {self.quadtree.root}
        next_level_cells: Set[QuadTreeCell] = set()
        contained_orders: List[int] = []

        while current_level_cells:
            for cell in current_level_cells:
                if skip_muted and cell.muted:
                    self._add_children(self.quadtree, cell, next_level_cells)
                    continue

                enlarged_bbox = cell.get_enlarged_element_bbox(self.alpha, self.beta)

                if query_bbox.contains(enlarged_bbox):
                    used_fast_interval = False
                    if can_use_xz_coverage and order_loader is not None:
                        order_value = order_lookup.get(cell)
                        coverage = order_loader.get_coverage_by_cell(cell)
                        subtree_count = None if coverage is None else coverage.get("effective_subtree_count")
                        if order_value is not None and isinstance(subtree_count, int):
                            row_intervals.append((int(order_value), int(order_value) + int(subtree_count) + 1))
                            self._collect_subtree_trajectories(
                                quadtree=self.quadtree,
                                cell=cell,
                                candidate_traj_ids=candidate_traj_ids,
                                skip_muted=skip_muted,
                            )
                            used_fast_interval = True

                    if not used_fast_interval:
                        stack = [cell]
                        while stack:
                            sub_cell = stack.pop()
                            if not (skip_muted and sub_cell.muted):
                                order_value = order_lookup.get(sub_cell)
                                if order_value is not None:
                                    contained_orders.append(int(order_value))
                                candidate_traj_ids.update(sub_cell.trajectories)
                            if sub_cell.level < self.quadtree.max_level:
                                for child in sub_cell.children:
                                    if child is not None:
                                        stack.append(child)
                elif query_bbox.intersects(enlarged_bbox):
                    query_sig = compute_query_signature(self.alpha, self.beta, cell, query_bbox)
                    if not cell.signatures:
                        self._add_children(self.quadtree, cell, next_level_cells)
                        continue

                    quad_order = order_lookup.get(cell)
                    if quad_order is None:
                        self._add_children(self.quadtree, cell, next_level_cells)
                        continue

                    self._append_matching_rowkeys(
                        cell=cell,
                        row_key=quad_order,
                        query_sig=query_sig,
                        candidate_traj_ids=candidate_traj_ids,
                        row_intervals=row_intervals,
                    )
                    self._add_children(self.quadtree, cell, next_level_cells)

            current_level_cells = next_level_cells
            next_level_cells = set()

        for order_value in sorted(set(contained_orders)):
            row_intervals.append((order_value, order_value + 1))

        merged = self.cost_evaluator.merge_intervals(row_intervals)
        return [(int(start), int(end)) for start, end in merged], candidate_traj_ids, 0

    def compute_hgs_score(
        self,
        quadorder: List[QuadTreeCell],
        test_queries: List[SpatialBoundingBox],
    ) -> Dict[str, Any]:
        ref_metrics = self.evaluate_final_order(quadorder)
        i_ref = ref_metrics["improvement_percent"]

        original_queries = self.reference_queries
        self.reference_queries = test_queries

        test_metrics = self.evaluate_final_order(quadorder)
        i_test = test_metrics["improvement_percent"]

        self.reference_queries = original_queries

        gap = abs(i_ref - i_test)
        max_val = max(i_ref, i_test, 1e-6)
        score = i_test * (1 - (gap / max_val))

        return {
            "i_ref": i_ref,
            "i_test": i_test,
            "hgs_score": score,
            "test_metrics": test_metrics,
        }

    def compare_orders(self, quadcode_order: List[QuadTreeCell], quadorder: List[QuadTreeCell]) -> dict:
        if self.reference_queries is None:
            raise ValueError("reference_queries are required")
        queries = self.reference_queries

        quadcode_costs = []
        quadorder_costs = []
        quadorder_lookup = self._build_order_lookup(quadorder)

        quadcode_nodes_hit = []
        quadcode_scan_counts = []
        quadorder_nodes_hit = []
        quadorder_scan_counts = []

        for query_bbox in queries:
            quadcode_intervals, quadcode_candidates, quadcode_gap_move_bits = self.search_quadcode_intervals(
                query_bbox,
                quadcode_order,
                skip_muted=not self.quadcode_include_muted,
            )
            quadorder_intervals, quadorder_candidates, quadorder_gap_move_bits = self.search_quadorder_intervals(
                query_bbox,
                quadorder,
                skip_muted=True,
                aligned=quadorder_lookup,
            )

            quadcode_nodes_hit.append(len(quadcode_candidates))
            quadorder_nodes_hit.append(len(quadorder_candidates))
            quadcode_scan_counts.append(len(quadcode_intervals))
            quadorder_scan_counts.append(len(quadorder_intervals))

            quadcode_cost = self.cost_evaluator.query_cost(
                quadcode_intervals,
                gap_move_bits=quadcode_gap_move_bits,
            )
            quadorder_cost = self.cost_evaluator.query_cost(
                quadorder_intervals,
                gap_move_bits=quadorder_gap_move_bits,
            )
            quadcode_costs.append(quadcode_cost)
            quadorder_costs.append(quadorder_cost)

        avg_quadcode_cost = float(np.mean(quadcode_costs)) if quadcode_costs else 0.0
        avg_quadorder_cost = float(np.mean(quadorder_costs)) if quadorder_costs else 0.0
        improvement = (
            (avg_quadcode_cost - avg_quadorder_cost) / avg_quadcode_cost * 100 if avg_quadcode_cost > 0 else 0.0
        )

        avg_quadcode_nodes_hit = np.mean(quadcode_nodes_hit)
        avg_quadorder_nodes_hit = np.mean(quadorder_nodes_hit)
        avg_quadcode_scan_counts = np.mean(quadcode_scan_counts)
        avg_quadorder_scan_counts = np.mean(quadorder_scan_counts)
        total_quadcode_cost = float(np.sum(quadcode_costs)) if quadcode_costs else 0.0
        total_quadorder_cost = float(np.sum(quadorder_costs)) if quadorder_costs else 0.0
        global_reward = total_quadcode_cost - total_quadorder_cost

        return {
            "quadcode_avg_cost": avg_quadcode_cost,
            "quadorder_avg_cost": avg_quadorder_cost,
            "improvement_percent": improvement,
            "quadcode_costs": quadcode_costs,
            "quadorder_costs": quadorder_costs,
            "global_reward": global_reward,
            "quadcode_nodes_hit": avg_quadcode_nodes_hit,
            "quadcode_scan_counts": avg_quadcode_scan_counts,
            "quadorder_nodes_hit": avg_quadorder_nodes_hit,
            "quadorder_scan_counts": avg_quadorder_scan_counts,
        }

    def evaluate_final_order(self, quadorder: List[QuadTreeCell]) -> dict:
        quadcode_order = self.encoder.z_curve_order(include_muted=self.quadcode_include_muted)
        return self.compare_orders(quadcode_order, quadorder)

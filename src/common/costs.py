from math import log
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from src.indexing.quadtree_cell import QuadTreeCell
from src.indexing.quadtree_index import QuadTreeIndex


class TraversalCostEvaluator:
    """Shared local-reward and rowKey cost evaluator."""

    def __init__(
        self,
        quadtree: QuadTreeIndex,
    ):
        self.quadtree = quadtree
        self._jaccard_cache: Dict[Tuple[int, int], float] = {}

    @staticmethod
    def merge_intervals(parts: Iterable[Tuple[float, float]]) -> List[Tuple[float, float]]:
        parts_list = [(float(start), float(end)) for start, end in parts if end > start]
        if not parts_list:
            return []

        sorted_parts = sorted(parts_list, key=lambda item: (item[0], item[1]))
        merged: List[List[float]] = [list(sorted_parts[0])]
        for start, end in sorted_parts[1:]:
            if start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])

        return [(start, end) for start, end in merged]

    @staticmethod
    def _distance(cell_a: QuadTreeCell, cell_b: QuadTreeCell) -> float:
        centroid_a = cell_a.get_center()
        centroid_b = cell_b.get_center()
        return float(np.linalg.norm(np.array(centroid_a) - np.array(centroid_b)))

    def proximity_reward(
        self,
        cell_a: QuadTreeCell,
        cell_b: QuadTreeCell,
        normaliser: float,
    ) -> float:
        distance = self._distance(cell_a, cell_b)
        normalised = min(distance / normaliser, 1.0)
        return 1.0 - normalised

    def jaccard_similarity(
        self,
        cell_a: QuadTreeCell,
        cell_b: QuadTreeCell,
        use_cache: bool = True,
    ) -> float:
        cache_key = None
        if use_cache:
            cache_key = self._jaccard_cache_key(cell_a, cell_b)
            cached = self._jaccard_cache.get(cache_key)
            if cached is not None:
                return cached

        trajectory_to_cell = self.quadtree.trajectory_to_cells
        trajectory_mbrs = self.quadtree.trajectory_mbrs

        if trajectory_to_cell is not None and trajectory_mbrs is not None:
            similarity = self._jaccard_with_index(cell_a, cell_b, trajectory_mbrs)
        else:
            similarity = self._jaccard_fallback(cell_a, cell_b)

        if use_cache and cache_key is not None:
            self._jaccard_cache[cache_key] = similarity
        return similarity

    @staticmethod
    def _jaccard_cache_key(cell_a: QuadTreeCell, cell_b: QuadTreeCell) -> Tuple[int, int]:
        node_a = id(cell_a)
        node_b = id(cell_b)
        return (node_a, node_b) if node_a <= node_b else (node_b, node_a)

    @staticmethod
    def _collect_intersecting_trajectories_with_bbox(
        source_cell: QuadTreeCell,
        target_cell: QuadTreeCell,
        trajectory_mbrs: dict,
        source_bucket: set,
        target_bucket: set,
    ) -> None:
        for trajectory_id in source_cell.trajectories:
            source_bucket.add(trajectory_id)
            trajectory_bbox = trajectory_mbrs.get(trajectory_id)
            if trajectory_bbox and target_cell.bbox.intersects(trajectory_bbox):
                target_bucket.add(trajectory_id)

    def _collect_intersecting_trajectories_with_search(
        self,
        source_cell: QuadTreeCell,
        target_cell: QuadTreeCell,
        source_bucket: set,
        target_bucket: set,
    ) -> None:
        for trajectory_id in source_cell.trajectories:
            source_bucket.add(trajectory_id)
            if self.quadtree.trajectory_intersects_cell(trajectory_id, target_cell):
                target_bucket.add(trajectory_id)

    def _jaccard_with_index(
        self,
        cell_a: QuadTreeCell,
        cell_b: QuadTreeCell,
        trajectory_mbrs: dict,
    ) -> float:
        trajectories_a = set()
        trajectories_b = set()

        self._collect_intersecting_trajectories_with_bbox(
            cell_a, cell_b, trajectory_mbrs, trajectories_a, trajectories_b
        )
        self._collect_intersecting_trajectories_with_bbox(
            cell_b, cell_a, trajectory_mbrs, trajectories_b, trajectories_a
        )

        return self._compute_jaccard(trajectories_a, trajectories_b)

    def _jaccard_fallback(self, cell_a: QuadTreeCell, cell_b: QuadTreeCell) -> float:
        trajectories_a = set()
        trajectories_b = set()

        self._collect_intersecting_trajectories_with_search(
            cell_a, cell_b, trajectories_a, trajectories_b
        )
        self._collect_intersecting_trajectories_with_search(
            cell_b, cell_a, trajectories_b, trajectories_a
        )

        return self._compute_jaccard(trajectories_a, trajectories_b)

    @staticmethod
    def _compute_jaccard(set_a: set, set_b: set) -> float:
        if not set_a and not set_b:
            return 1.0

        union = len(set_a | set_b)
        if union == 0:
            return 0.0
        return len(set_a & set_b) / union

    def step_reward(
        self,
        current_cell: QuadTreeCell,
        next_cell: QuadTreeCell,
        normaliser: float,
        proximity_weight: float = 0.5,
        similarity_weight: float = 0.5,
    ) -> float:
        proximity_value = self.proximity_reward(current_cell, next_cell, normaliser)
        similarity_value = self.jaccard_similarity(current_cell, next_cell)
        return proximity_weight * proximity_value + similarity_weight * similarity_value

    @staticmethod
    def _normalize_rowkey_intervals(
        intervals: Iterable[Tuple[int, int]],
        gap_move_bits: int,
    ) -> List[Tuple[float, float]]:
        scale = float(1 << int(gap_move_bits)) if gap_move_bits > 0 else 1.0
        return [
            (float(start) / scale, float(end) / scale)
            for start, end in intervals
            if end > start
        ]

    @staticmethod
    def _compute_gap_lengths(intervals: Sequence[Tuple[float, float]]) -> List[float]:
        if len(intervals) <= 1:
            return []

        sorted_intervals = sorted(intervals, key=lambda x: (x[0], x[1]))
        gaps: List[float] = []
        for idx in range(len(sorted_intervals) - 1):
            gap = float(sorted_intervals[idx + 1][0] - sorted_intervals[idx][1])
            if gap > 0:
                gaps.append(gap)
        return gaps

    @staticmethod
    def _normalized_gap_entropy(gaps: Sequence[float]) -> float:
        if len(gaps) <= 1:
            return 0.0

        total_gap = float(sum(gaps))
        if total_gap <= 0:
            return 0.0

        probabilities = np.array([gap / total_gap for gap in gaps if gap > 0], dtype=float)
        if len(probabilities) <= 1:
            return 0.0

        entropy = float(-np.sum(probabilities * np.log(probabilities)))
        max_entropy = log(len(probabilities))
        if max_entropy <= 0:
            return 0.0
        return float(np.clip(entropy / max_entropy, 0.0, 1.0))

    def query_cost(self, intervals: Iterable[Tuple[int, int]], gap_move_bits: int = 0) -> float:
        intervals_list = [(int(start), int(end)) for start, end in intervals if end > start]
        interval_count = len(intervals_list)

        if interval_count == 0:
            return 0.0

        if interval_count <= 1:
            return float(interval_count)

        normalized_intervals = self._normalize_rowkey_intervals(intervals_list, gap_move_bits)
        merged_intervals = self.merge_intervals(normalized_intervals)
        gap_entropy = self._normalized_gap_entropy(self._compute_gap_lengths(merged_intervals))

        return float(interval_count + gap_entropy)

    def global_reward(
        self,
        quadorder_intervals: Iterable[Tuple[int, int]],
        quadcode_cost: float,
        scale: float = 1.0,
        quadorder_gap_move_bits: int = 0,
    ) -> Tuple[float, float]:
        quadorder_cost = self.query_cost(quadorder_intervals, gap_move_bits=quadorder_gap_move_bits)
        improvement = quadcode_cost - quadorder_cost
        return improvement * scale, improvement

from dataclasses import dataclass
from itertools import combinations
from typing import List, Tuple

import numpy as np

from src.common import SpatialBoundingBox
from src.indexing.quadtree_cell import QuadTreeCell
from src.utils.signature import compute_signature_vectorized


@dataclass(frozen=True)
class PartitionMetrics:
    alpha: int
    beta: int
    occupancy_ratio: float
    discrimination: float
    score: float


class SignatureOptimizer:
    """Adaptive partition optimizer for a single enlarged element."""

    def __init__(
            self,
            alpha_range: Tuple[int, int] = (2, 8),
            beta_range: Tuple[int, int] = (2, 8),
            occupancy_weight: float = 0.3,
            discrimination_weight: float = 0.7,
            max_sample_trajs: int = 64,
    ):
        self.min_alpha, self.max_alpha = alpha_range
        self.min_beta, self.max_beta = beta_range
        self.occupancy_weight = occupancy_weight
        self.discrimination_weight = discrimination_weight
        self.max_sample_trajs = max_sample_trajs

    def _sample_trajectories(
            self,
            traj_points_list: List[np.ndarray],
            seed: int,
    ) -> List[np.ndarray]:
        if len(traj_points_list) <= self.max_sample_trajs:
            return traj_points_list

        rng = np.random.default_rng(seed)
        sample_indices = rng.choice(
            len(traj_points_list),
            size=self.max_sample_trajs,
            replace=False,
        )
        return [traj_points_list[int(idx)] for idx in sample_indices]

    @staticmethod
    def _bit_count(value: int) -> int:
        return bin(int(value)).count("1")

    def evaluate_partition(
            self,
            alpha: int,
            beta: int,
            traj_points_list: List[np.ndarray],
            ee_bbox: SpatialBoundingBox,
            seed: int = 0,
    ) -> PartitionMetrics:
        """Score a partition by occupancy ratio and normalized Hamming discrimination."""
        if not traj_points_list:
            return PartitionMetrics(alpha, beta, 0.0, 0.0, 0.0)

        sampled_trajs = self._sample_trajectories(traj_points_list, seed)
        signatures = [
            compute_signature_vectorized(alpha, beta, points, ee_bbox)
            for points in sampled_trajs
        ]

        total_cells = alpha * beta
        occupied_mask = 0
        for signature in signatures:
            occupied_mask |= int(signature)
        occupancy_ratio = self._bit_count(occupied_mask) / total_cells if total_cells else 0.0

        if len(signatures) < 2 or total_cells == 0:
            discrimination = 0.0
        else:
            pair_scores = [
                self._bit_count(int(signatures[i]) ^ int(signatures[j])) / total_cells
                for i, j in combinations(range(len(signatures)), 2)
            ]
            discrimination = float(np.mean(pair_scores)) if pair_scores else 0.0

        score = (
                self.occupancy_weight * occupancy_ratio
                + self.discrimination_weight * discrimination
        )
        return PartitionMetrics(alpha, beta, occupancy_ratio, discrimination, score)

    def find_best_config(
            self,
            global_alpha: int,
            global_beta: int,
            traj_points_list: List[np.ndarray],
            cell: QuadTreeCell,
    ) -> Tuple[int, int]:
        """Find the best adaptive partition by grid search from global alpha/beta up to the configured max."""
        ee_bbox = cell.get_enlarged_element_bbox(global_alpha, global_beta)
        base_alpha = min(max(2, global_alpha), self.max_alpha)
        base_alpha = max(self.min_alpha, base_alpha)
        base_beta = min(max(2, global_beta), self.max_beta)
        base_beta = max(self.min_beta, base_beta)
        seed = int(cell.code * 1315423911) & 0xFFFFFFFF

        best = self.evaluate_partition(base_alpha, base_beta, traj_points_list, ee_bbox, seed)

        for alpha in range(base_alpha, self.max_alpha + 1):
            for beta in range(base_beta, self.max_beta + 1):
                if alpha == base_alpha and beta == base_beta:
                    continue

                candidate = self.evaluate_partition(alpha, beta, traj_points_list, ee_bbox, seed)
                if candidate.score > best.score * 1.1:
                    best = candidate

        return best.alpha, best.beta

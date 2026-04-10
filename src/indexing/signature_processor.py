"""QuadTree signature processor."""
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, Union

import numpy as np

from src.indexing.quadtree_cell import QuadTreeCell
from src.training.signature_optimizer import SignatureOptimizer
from src.utils.signature import compute_signature_vectorized


TrajectoryPointsAccessor = Union[Dict[int, List], Callable[[int], List]]


class SignatureProcessor:
    """Compute cell signatures and optionally optimize local partitions."""

    def __init__(self, global_alpha: int, global_beta: int, max_workers: Optional[int] = None):
        self.global_alpha = global_alpha
        self.global_beta = global_beta
        self.optimizer = SignatureOptimizer(alpha_range=(2, 8), beta_range=(2, 8))
        self.max_workers = max_workers
        self.logger = logging.getLogger(self.__class__.__name__)

    def process_cell_signatures(
        self,
        cells_by_level: List[List[QuadTreeCell]],
        trajectory_points: TrajectoryPointsAccessor,
        max_level: int,
        enable_optimize: bool = True,
        parallel: bool = False,
    ) -> Dict[str, int]:
        def get_points(tid: int):
            if callable(trajectory_points):
                return trajectory_points(tid)
            return trajectory_points.get(tid)

        if parallel and self.max_workers and self.max_workers > 1:
            return self._process_parallel(cells_by_level, get_points, max_level, enable_optimize)

        return self._process_serial(cells_by_level, get_points, max_level, enable_optimize)

    def _process_serial(
        self,
        cells_by_level: List[List[QuadTreeCell]],
        get_points: Callable[[int], List],
        max_level: int,
        enable_optimize: bool,
    ) -> Dict[str, int]:
        stats = {"shrunk_alpha": 0, "shrunk_beta": 0, "both_shrunk": 0}

        for level in range(0, max_level + 1):
            for cell in cells_by_level[level]:
                result = self._process_single_cell(cell, get_points, enable_optimize)
                stats["shrunk_alpha"] += result["shrunk_alpha"]
                stats["shrunk_beta"] += result["shrunk_beta"]
                stats["both_shrunk"] += result["both_shrunk"]

        return stats

    def _process_parallel(
        self,
        cells_by_level: List[List[QuadTreeCell]],
        get_points: Callable[[int], List],
        max_level: int,
        enable_optimize: bool,
    ) -> Dict[str, int]:
        all_cells = []
        for level in range(0, max_level + 1):
            for cell in cells_by_level[level]:
                if not cell.muted and cell.trajectories:
                    all_cells.append(cell)

        if not all_cells:
            return {"shrunk_alpha": 0, "shrunk_beta": 0, "both_shrunk": 0}

        self.logger.info(
            "Parallel signature computation for %s active cells (workers=%s)",
            len(all_cells),
            self.max_workers,
        )

        max_workers = min(self.max_workers, len(all_cells))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(
                executor.map(
                    lambda cell: self._process_single_cell(cell, get_points, enable_optimize),
                    all_cells,
                )
            )

        stats = {"shrunk_alpha": 0, "shrunk_beta": 0, "both_shrunk": 0}
        for result in results:
            stats["shrunk_alpha"] += result["shrunk_alpha"]
            stats["shrunk_beta"] += result["shrunk_beta"]
            stats["both_shrunk"] += result["both_shrunk"]

        return stats

    def _process_single_cell(
        self,
        cell: QuadTreeCell,
        get_points: Callable[[int], List],
        enable_optimize: bool,
    ) -> Dict[str, int]:
        stats = {"shrunk_alpha": 0, "shrunk_beta": 0, "both_shrunk": 0}

        if cell.muted or not cell.trajectories:
            cell.alpha, cell.beta = self.global_alpha, self.global_beta
            cell.signatures.clear()
            return stats

        traj_pts_list = []
        valid_tids = []
        for tid in cell.trajectories:
            pts = get_points(tid)
            if pts is not None:
                traj_pts_list.append(np.array(pts))
                valid_tids.append(tid)

        if not traj_pts_list:
            cell.alpha, cell.beta = self.global_alpha, self.global_beta
            cell.signatures.clear()
            return stats

        if enable_optimize:
            best_alpha, best_beta = self.optimizer.find_best_config(
                self.global_alpha, self.global_beta, traj_pts_list, cell
            )
            if best_alpha < self.global_alpha:
                stats["shrunk_alpha"] = 1
            if best_beta < self.global_beta:
                stats["shrunk_beta"] = 1
            if best_alpha < self.global_alpha and best_beta < self.global_beta:
                stats["both_shrunk"] = 1
            cell.alpha, cell.beta = best_alpha, best_beta
        else:
            cell.alpha, cell.beta = self.global_alpha, self.global_beta

        cell.signatures.clear()
        ee_bbox = cell.get_enlarged_element_bbox(self.global_alpha, self.global_beta)
        for tid, pts_np in zip(valid_tids, traj_pts_list):
            cell.signatures[tid] = compute_signature_vectorized(
                cell.alpha,
                cell.beta,
                pts_np,
                ee_bbox,
            )

        return stats

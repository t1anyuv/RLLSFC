import numpy as np

from src.core.bounding_box import SpatialBoundingBox
from src.indexing.quadtree_cell import QuadTreeCell
from src.indexing.signature_processor import SignatureProcessor
from src.training.signature_optimizer import PartitionMetrics, SignatureOptimizer


def make_cell(alpha: int = 3, beta: int = 3) -> QuadTreeCell:
    return QuadTreeCell(
        bbox=SpatialBoundingBox(0.0, 0.0, 2.0, 2.0),
        level=1,
        quadrant_sequence=[0],
        code=7,
        alpha=alpha,
        beta=beta,
    )


def test_evaluate_partition_uses_occupancy_and_discrimination():
    optimizer = SignatureOptimizer()
    ee_bbox = SpatialBoundingBox(0.0, 0.0, 4.0, 4.0)
    trajectories = [
        np.array([(0.5, 0.5), (0.9, 0.9)]),
        np.array([(2.5, 2.5), (2.9, 2.9)]),
    ]

    metrics = optimizer.evaluate_partition(2, 2, trajectories, ee_bbox)

    assert metrics.occupancy_ratio == 0.5
    assert metrics.discrimination == 0.5
    assert metrics.score == 0.5


class StubOptimizer(SignatureOptimizer):
    def __init__(self, score_table):
        super().__init__(alpha_range=(2, 5), beta_range=(2, 5))
        self.score_table = score_table

    def evaluate_partition(self, alpha, beta, traj_points_list, ee_bbox, seed=0):
        score = self.score_table.get((alpha, beta), float("-inf"))
        return PartitionMetrics(alpha, beta, 0.0, 0.0, score)


def test_find_best_config_uses_grid_search_and_global_anchor():
    optimizer = StubOptimizer(
        {
            (2, 2): 1.00,
            (2, 3): 1.25,
            (2, 4): 1.20,
            (2, 5): 1.10,
            (3, 2): 1.05,
            (3, 3): 1.02,
            (3, 4): 0.95,
            (3, 5): 0.90,
            (4, 2): 1.40,
            (4, 3): 1.18,
            (4, 4): 1.12,
            (4, 5): 1.08,
            (5, 2): 1.30,
            (5, 3): 1.15,
            (5, 4): 1.05,
            (5, 5): 1.00,
        }
    )
    cell = make_cell()

    best_alpha, best_beta = optimizer.find_best_config(
        global_alpha=3,
        global_beta=3,
        traj_points_list=[np.array([(0.0, 0.0)])],
        cell=cell,
    )

    assert (best_alpha, best_beta) == (4, 2)


def test_find_best_config_prefers_global_anchor_when_scores_tie():
    optimizer = StubOptimizer(
        {
            (2, 2): 1.00,
            (2, 3): 1.20,
            (2, 4): 1.00,
            (2, 5): 0.80,
            (3, 2): 1.20,
            (3, 3): 1.20,
            (3, 4): 0.90,
            (3, 5): 0.80,
            (4, 2): 0.90,
            (4, 3): 0.85,
            (4, 4): 0.80,
            (4, 5): 0.75,
            (5, 2): 0.70,
            (5, 3): 0.65,
            (5, 4): 0.60,
            (5, 5): 0.55,
        }
    )
    cell = make_cell()

    best_alpha, best_beta = optimizer.find_best_config(
        global_alpha=3,
        global_beta=3,
        traj_points_list=[np.array([(0.0, 0.0)])],
        cell=cell,
    )

    assert (best_alpha, best_beta) == (3, 3)


def test_signature_processor_applies_adaptive_partition_and_refreshes_signatures():
    processor = SignatureProcessor(global_alpha=3, global_beta=3)
    processor.optimizer = StubOptimizer(
        {
            (2, 2): 1.00,
            (3, 3): 0.98,
            (3, 2): 0.95,
            (2, 3): 1.20,
            (2, 4): 1.18,
        }
    )

    cell = make_cell(alpha=3, beta=3)
    cell.trajectories = {1, 2}
    cell.signatures = {999: 123456}

    trajectories = {
        1: [(0.2, 0.2), (0.4, 0.4)],
        2: [(0.2, 3.1), (0.4, 3.3)],
    }

    result = processor.process_cell_signatures(
        cells_by_level=[[cell]],
        trajectory_points=trajectories,
        max_level=0,
        enable_optimize=True,
        parallel=False,
    )

    assert (cell.alpha, cell.beta) == (2, 3)
    assert set(cell.signatures.keys()) == {1, 2}
    assert 999 not in cell.signatures
    assert result["shrunk_alpha"] == 1

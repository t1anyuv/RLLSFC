"""
Synthetic order generator used for experiments without access to real data.
"""
from typing import List, Tuple

import numpy as np

from src.common import SpatialBoundingBox


class SyntheticTrajectoryFactory:
    """Utility class responsible for generating synthetic trajectories."""

    @staticmethod
    def generate(num_trajectories: int, bbox: SpatialBoundingBox, seed: int = 42) -> List[Tuple[int, List[Tuple[float, float]]]]:
        np.random.seed(seed)
        trajectories: List[Tuple[int, List[Tuple[float, float]]]] = []

        for trajectory_id in range(num_trajectories):
            num_points = np.random.randint(10, 50)
            points: List[Tuple[float, float]] = []

            x_position = np.random.uniform(bbox.min_x, bbox.max_x)
            y_position = np.random.uniform(bbox.min_y, bbox.max_y)

            for _ in range(num_points):
                x_position += np.random.uniform(-0.1, 0.1) * (bbox.max_x - bbox.min_x)
                y_position += np.random.uniform(-0.1, 0.1) * (bbox.max_y - bbox.min_y)

                x_position = float(np.clip(x_position, bbox.min_x, bbox.max_x))
                y_position = float(np.clip(y_position, bbox.min_y, bbox.max_y))

                points.append((x_position, y_position))

            trajectories.append((trajectory_id, points))

        return trajectories


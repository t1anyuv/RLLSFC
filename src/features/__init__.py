"""Feature engineering for state representation and trajectory statistics.

This module provides feature builders for RL state representation:
- StateFeatureBuilder: Builds feature vectors for policy input
- TrajectoryStatistics: Computes trajectory-level statistics
"""

from .state_feature_builder import StateFeatureBuilder
from .trajectory_statistics import TrajectoryStatistics

__all__ = ["StateFeatureBuilder", "TrajectoryStatistics"]

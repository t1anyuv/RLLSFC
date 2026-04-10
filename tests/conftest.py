"""Shared pytest fixtures."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.config import (
    DataConfig,
    DatasetCatalogConfig,
    DatasetProfileConfig,
    IndexConfig,
    PathConfig,
    RewardConfig,
    TShapeConfig,
    TrainConfig,
)
from src.core.bounding_box import SpatialBoundingBox
from src.indexing.quadtree_index import QuadTreeIndex


@pytest.fixture
def test_bbox():
    return SpatialBoundingBox(min_x=0.0, min_y=0.0, max_x=100.0, max_y=100.0)


@pytest.fixture
def test_config():
    return TShapeConfig(
        index=IndexConfig(
            max_level=4,
            alpha=2,
            beta=2,
            min_x=0.0,
            min_y=0.0,
            max_x=100.0,
            max_y=100.0,
            use_prune=False,
            min_cell_trajs=0,
        ),
        data=DataConfig(
            num_trajectories=100,
            source="synthetic",
            use_similarity_matrix=False,
        ),
        reward=RewardConfig(query_distribution_type="uniform"),
        train=TrainConfig(
            num_episodes=10,
            eval_interval=5,
            save_interval=10,
        ),
        paths=PathConfig(resource_base_dir="tests/test_resources"),
        datasets=DatasetCatalogConfig(
            active="unit_test",
            profiles={
                "unit_test": DatasetProfileConfig(
                    description="unit test dataset",
                    trajectory_path="tests/test_resources/unit_test.txt",
                    query_root="tests/test_resources/queries",
                    min_x=0.0,
                    min_y=0.0,
                    max_x=100.0,
                    max_y=100.0,
                )
            },
        ),
    )


@pytest.fixture
def test_quadtree(test_bbox):
    return QuadTreeIndex(test_bbox, max_level=4, alpha=2, beta=2)


@pytest.fixture
def sample_trajectories():
    trajectories = []
    np.random.seed(42)

    for _ in range(10):
        num_points = np.random.randint(5, 20)
        points = np.random.rand(num_points, 2) * 100
        trajectories.append(points)

    return trajectories


@pytest.fixture
def temp_config_dir():
    base_dir = Path.cwd() / ".pytest_tmp"
    base_dir.mkdir(exist_ok=True)

    with tempfile.TemporaryDirectory(dir=base_dir) as temp_dir:
        config_dir = Path(temp_dir) / "configs"
        config_dir.mkdir()
        yield config_dir


@pytest.fixture
def temp_resource_dir():
    base_dir = Path.cwd() / ".pytest_tmp"
    base_dir.mkdir(exist_ok=True)

    with tempfile.TemporaryDirectory(dir=base_dir) as temp_dir:
        resource_dir = Path(temp_dir) / "resources"
        resource_dir.mkdir()
        (resource_dir / "models").mkdir()
        (resource_dir / "orders").mkdir()
        (resource_dir / "similarity").mkdir()
        (resource_dir / "logs").mkdir()
        yield resource_dir

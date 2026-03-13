"""pytest配置和共享fixtures"""
import pytest
import numpy as np
from pathlib import Path
from src.config import TShapeConfig, IndexConfig, DataConfig, RewardConfig, TrainConfig, PathConfig
from src.core.bounding_box import SpatialBoundingBox
from src.indexing.quadtree_index import QuadTreeIndex


@pytest.fixture
def test_bbox():
    """测试用边界框"""
    return SpatialBoundingBox(min_x=0.0, min_y=0.0, max_x=100.0, max_y=100.0)


@pytest.fixture
def test_config():
    """测试用配置"""
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
            min_cell_trajs=0
        ),
        data=DataConfig(
            num_trajectories=100,
            use_tdrive_data=False,
            use_similarity_matrix=False
        ),
        reward=RewardConfig(
            tau_loc=1.0,
            tau_scan=0.1,
            query_dataset_path="resource/queries",
            query_distribution_type="uniform"
        ),
        train=TrainConfig(
            num_episodes=10,
            eval_interval=5,
            save_interval=10
        ),
        paths=PathConfig(
            resource_base_dir='tests/test_resources'
        )
    )


@pytest.fixture
def test_quadtree(test_bbox):
    """测试用四叉树索引"""
    return QuadTreeIndex(test_bbox, max_level=4, alpha=2, beta=2)


@pytest.fixture
def sample_trajectories():
    """生成示例轨迹数据"""
    trajectories = []
    np.random.seed(42)
    
    for i in range(10):
        num_points = np.random.randint(5, 20)
        points = np.random.rand(num_points, 2) * 100
        trajectories.append(points)
    
    return trajectories


@pytest.fixture
def temp_config_dir(tmp_path):
    """临时配置目录"""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    return config_dir


@pytest.fixture
def temp_resource_dir(tmp_path):
    """临时资源目录"""
    resource_dir = tmp_path / "resources"
    resource_dir.mkdir()
    (resource_dir / "models").mkdir()
    (resource_dir / "orders").mkdir()
    (resource_dir / "similarity").mkdir()
    (resource_dir / "logs").mkdir()
    return resource_dir

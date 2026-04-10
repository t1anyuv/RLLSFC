"""Tests for the configuration module."""

from src.config import (
    DataConfig,
    DatasetCatalogConfig,
    DatasetProfileConfig,
    IndexConfig,
    NetworkConfig,
    PathConfig,
    TShapeConfig,
)


class TestConfigCreation:
    def test_default_config_creation(self):
        config = TShapeConfig()
        assert config.index.max_level == 8
        assert config.data.num_trajectories == 3000
        assert config.train.num_episodes == 1000

    def test_custom_config_creation(self):
        config = TShapeConfig(
            index=IndexConfig(max_level=10, alpha=3, beta=3),
            data=DataConfig(num_trajectories=5000),
        )
        assert config.index.max_level == 10
        assert config.index.alpha == 3
        assert config.data.num_trajectories == 5000


class TestConfigSerialization:
    def test_to_dict(self, test_config):
        config_dict = test_config.to_dict()
        assert isinstance(config_dict, dict)
        assert "index" in config_dict
        assert "data" in config_dict
        assert config_dict["index"]["max_level"] == 4

    def test_from_dict(self):
        data = {
            "index": {"max_level": 6, "alpha": 3, "beta": 3},
            "data": {"num_trajectories": 2000},
            "train": {"num_episodes": 500},
        }
        config = TShapeConfig.from_dict(data)
        assert config.index.max_level == 6
        assert config.index.alpha == 3
        assert config.data.num_trajectories == 2000

    def test_save_and_load_yaml(self, test_config, temp_config_dir):
        yaml_path = temp_config_dir / "test_config.yaml"
        test_config.save_yaml(str(yaml_path))

        assert yaml_path.exists()

        loaded_config = TShapeConfig.from_yaml(str(yaml_path))
        assert loaded_config.index.max_level == test_config.index.max_level
        assert loaded_config.data.num_trajectories == test_config.data.num_trajectories

    def test_save_and_load_json(self, test_config, temp_config_dir):
        json_path = temp_config_dir / "test_config.json"
        test_config.save_json(str(json_path))

        assert json_path.exists()

        loaded_config = TShapeConfig.from_json(str(json_path))
        assert loaded_config.index.max_level == test_config.index.max_level
        assert loaded_config.train.num_episodes == test_config.train.num_episodes


class TestPathConfig:
    def test_path_properties(self):
        path_config = PathConfig(resource_base_dir="test_resources")

        assert path_config.model_dir.name == "models"
        assert path_config.order_dir.name == "orders"
        assert path_config.similarity_dir.name == "similarity"
        assert path_config.log_dir.name == "logs"
        assert "test_resources" in str(path_config.model_dir) or "resource" in str(path_config.model_dir)


class TestNetworkConfig:
    def test_device_auto_detection(self):
        config = NetworkConfig(device="auto")
        device = config.get_torch_device()
        assert device.type in ["cuda", "cpu"]

    def test_device_manual_setting(self):
        config = NetworkConfig(device="cpu")
        device = config.get_torch_device()
        assert device.type == "cpu"


class TestIndexConfig:
    def test_bbox_tuple(self):
        config = IndexConfig(min_x=10.0, min_y=20.0, max_x=30.0, max_y=40.0)
        bbox_tuple = config.get_bbox_tuple()
        assert bbox_tuple == (10.0, 20.0, 30.0, 40.0)


class TestBoundingBoxResolution:
    def test_get_original_bbox(self, test_config):
        bbox = test_config.get_original_bbox()
        assert bbox.min_x == test_config.index.min_x
        assert bbox.max_y == test_config.index.max_y


class TestSimilarityMatrixPaths:
    def test_default_similarity_matrix_filename(self):
        config = TShapeConfig(
            index=IndexConfig(max_level=9, alpha=3, beta=5),
            data=DataConfig(num_trajectories=-1),
            datasets=DatasetCatalogConfig(
                active="demo",
                profiles={
                    "demo": DatasetProfileConfig(
                        trajectory_path="demo.txt",
                        query_root="resource/demo_queries",
                        min_x=0.0,
                        min_y=0.0,
                        max_x=1.0,
                        max_y=1.0,
                    )
                },
            ),
        )
        assert config.get_default_similarity_matrix_filename() == "sim_mtx_demo_L9_A3_B5_T-1.npz"

    def test_effective_similarity_matrix_path_uses_default_name(self, test_config):
        matrix_path = test_config.get_effective_similarity_matrix_path()
        assert matrix_path.name == "sim_mtx_unit_test_L4_A2_B2_T100.npz"
        assert "shared" in str(matrix_path)
        assert "similarity" in str(matrix_path)

    def test_explicit_similarity_matrix_path_supports_nested_relative_path(self):
        config = TShapeConfig(
            data=DataConfig(similarity_matrix_path="resource/experiments/demo/sim/custom.npz")
        )
        matrix_path = config.get_effective_similarity_matrix_path()
        assert matrix_path.name == "custom.npz"
        assert "resource" in str(matrix_path)
        assert "experiments" in str(matrix_path)


class TestDatasetProfiles:
    def test_dataset_profile_resolves_query_root(self, test_config):
        query_root = test_config.get_query_dataset_root()
        assert query_root.name == "queries"
        assert "tests" in str(query_root)

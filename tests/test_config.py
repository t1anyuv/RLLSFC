"""配置管理模块测试"""
import pytest
import json
import yaml
from pathlib import Path
from src.config import (
    TShapeConfig, IndexConfig, DataConfig, RewardConfig, 
    TrainConfig, NetworkConfig, PathConfig
)


class TestConfigCreation:
    """测试配置对象创建"""
    
    def test_default_config_creation(self):
        """测试默认配置创建"""
        config = TShapeConfig()
        assert config.index.max_level == 8
        assert config.data.num_trajectories == 3000
        assert config.train.num_episodes == 1000
    
    def test_custom_config_creation(self):
        """测试自定义配置创建"""
        config = TShapeConfig(
            index=IndexConfig(max_level=10, alpha=3, beta=3),
            data=DataConfig(num_trajectories=5000)
        )
        assert config.index.max_level == 10
        assert config.index.alpha == 3
        assert config.data.num_trajectories == 5000


class TestConfigSerialization:
    """测试配置序列化和反序列化"""
    
    def test_to_dict(self, test_config):
        """测试转换为字典"""
        config_dict = test_config.to_dict()
        assert isinstance(config_dict, dict)
        assert 'index' in config_dict
        assert 'data' in config_dict
        assert config_dict['index']['max_level'] == 4
    
    def test_from_dict(self):
        """测试从字典创建"""
        data = {
            'index': {'max_level': 6, 'alpha': 3, 'beta': 3},
            'data': {'num_trajectories': 2000},
            'train': {'num_episodes': 500}
        }
        config = TShapeConfig.from_dict(data)
        assert config.index.max_level == 6
        assert config.index.alpha == 3
        assert config.data.num_trajectories == 2000
    
    def test_save_and_load_yaml(self, test_config, temp_config_dir):
        """测试YAML保存和加载"""
        yaml_path = temp_config_dir / "test_config.yaml"
        test_config.save_yaml(str(yaml_path))
        
        assert yaml_path.exists()
        
        loaded_config = TShapeConfig.from_yaml(str(yaml_path))
        assert loaded_config.index.max_level == test_config.index.max_level
        assert loaded_config.data.num_trajectories == test_config.data.num_trajectories
    
    def test_save_and_load_json(self, test_config, temp_config_dir):
        """测试JSON保存和加载"""
        json_path = temp_config_dir / "test_config.json"
        test_config.save_json(str(json_path))
        
        assert json_path.exists()
        
        loaded_config = TShapeConfig.from_json(str(json_path))
        assert loaded_config.index.max_level == test_config.index.max_level
        assert loaded_config.train.num_episodes == test_config.train.num_episodes


class TestPathConfig:
    """测试路径配置"""
    
    def test_path_properties(self):
        """测试路径属性"""
        path_config = PathConfig(resource_base_dir='test_resources')
        
        # 检查路径的末尾部分，因为PathManager返回绝对路径
        assert path_config.model_dir.name == 'models'
        assert path_config.order_dir.name == 'orders'
        assert path_config.similarity_dir.name == 'similarity'
        assert path_config.log_dir.name == 'logs'
        
        # 检查路径包含resource_base_dir
        assert 'test_resources' in str(path_config.model_dir) or 'resource' in str(path_config.model_dir)


class TestNetworkConfig:
    """测试网络配置"""
    
    def test_device_auto_detection(self):
        """测试设备自动检测"""
        config = NetworkConfig(device='auto')
        device = config.get_torch_device()
        assert device.type in ['cuda', 'cpu']
    
    def test_device_manual_setting(self):
        """测试手动设置设备"""
        config = NetworkConfig(device='cpu')
        device = config.get_torch_device()
        assert device.type == 'cpu'


class TestIndexConfig:
    """测试索引配置"""
    
    def test_bbox_tuple(self):
        """测试边界框元组"""
        config = IndexConfig(
            min_x=10.0, min_y=20.0,
            max_x=30.0, max_y=40.0
        )
        bbox_tuple = config.get_bbox_tuple()
        assert bbox_tuple == (10.0, 20.0, 30.0, 40.0)


class TestCompatibility:
    """测试向后兼容性"""
    
    def test_legacy_property_access(self, test_config):
        """测试旧版属性访问方式"""
        assert test_config.max_level == test_config.index.max_level
        assert test_config.alpha == test_config.index.alpha
        assert test_config.beta == test_config.index.beta
        assert test_config.num_trajectories == test_config.data.num_trajectories
        assert test_config.num_episodes == test_config.train.num_episodes
    
    def test_get_original_bbox(self, test_config):
        """测试获取原始边界框"""
        bbox = test_config.get_original_bbox()
        assert bbox.min_x == test_config.index.min_x
        assert bbox.max_y == test_config.index.max_y


class TestSimilarityMatrixPaths:
    """测试相似度矩阵默认命名与路径解析。"""

    def test_default_similarity_matrix_filename(self):
        config = TShapeConfig(
            index=IndexConfig(max_level=9, alpha=3, beta=5),
            data=DataConfig(num_trajectories=-1),
        )
        assert config.get_default_similarity_matrix_filename() == "sim_mtx_L9_A3_B5_T-1.npz"

    def test_effective_similarity_matrix_path_uses_default_name(self, test_config):
        matrix_path = test_config.get_effective_similarity_matrix_path()
        assert matrix_path.name == "sim_mtx_L4_A2_B2_T100.npz"
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

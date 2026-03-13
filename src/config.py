"""统一的配置管理模块

本模块提供了项目的配置管理功能，采用模块化设计，将配置分为多个子模块：
- IndexConfig: 四叉树索引配置
- DataConfig: 数据加载配置
- RewardConfig: 奖励函数配置
- TrainConfig: 训练流程配置
- NetworkConfig: 神经网络配置
- PathConfig: 路径配置

使用示例：
    # 从YAML文件加载配置
    config = TShapeConfig.from_yaml('configs/default.yaml')
    
    # 访问配置
    max_level = config.index.max_level
    num_trajs = config.data.num_trajectories
    
    # 保存配置
    config.save_yaml('configs/my_config.yaml')
"""
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
import json
import torch
import yaml


@dataclass
class ExperimentConfig:
    """实验配置

    Attributes:
        name: 实验名称（用于组织输出目录）
        description: 实验描述
    """
    name: str = "default"
    description: str = ""

    def get_output_dir(self) -> Path:
        """获取实验输出目录：resource/experiments/{name}/"""
        return Path("resource/experiments") / self.name
    
    def get_models_dir(self) -> Path:
        """获取模型输出目录"""
        return self.get_output_dir() / "models"
    
    def get_orders_dir(self) -> Path:
        """获取顺序输出目录"""
        return self.get_output_dir() / "orders"
    
    def get_logs_dir(self) -> Path:
        """获取日志输出目录"""
        return self.get_output_dir() / "logs"
    
    def get_curves_dir(self) -> Path:
        """获取曲线图输出目录"""
        return self.get_output_dir() / "curves"


@dataclass
class PathConfig:
    """路径配置
    
    Attributes:
        resource_base_dir: 资源文件基础目录（支持相对路径和绝对路径）
        tdrive_data_dir: TDrive数据集目录路径（支持相对路径和绝对路径）
        tdrive_data_path: TDrive数据集文件路径（支持相对路径和绝对路径）
        
    注意：
        - 相对路径将相对于项目根目录解析
        - 可以通过环境变量覆盖：
          * PROJECT_ROOT: 项目根目录
          * RESOURCE_BASE_DIR: 资源基础目录
          * TDRIVE_DATA_DIR: TDrive数据目录
          * TDRIVE_DATA_PATH: TDrive数据文件路径
    """
    resource_base_dir: str = 'resource'
    tdrive_data_dir: Optional[str] = None
    tdrive_data_path: Optional[str] = None

    def _get_path_manager(self):
        """获取路径管理器实例"""
        from src.utils.path_manager import get_path_manager
        return get_path_manager()

    @property
    def model_dir(self) -> Path:
        """模型保存目录"""
        pm = self._get_path_manager()
        if pm:
            return pm.get_model_dir()
        return Path(self.resource_base_dir) / "models"

    @property
    def order_dir(self) -> Path:
        """遍历顺序保存目录"""
        pm = self._get_path_manager()
        if pm:
            return pm.get_order_dir()
        return Path(self.resource_base_dir) / "orders"

    @property
    def similarity_dir(self) -> Path:
        """相似度矩阵保存目录"""
        pm = self._get_path_manager()
        if pm:
            return pm.get_similarity_dir()
        return Path(self.resource_base_dir) / "similarity"

    @property
    def log_dir(self) -> Path:
        """日志保存目录"""
        pm = self._get_path_manager()
        if pm:
            return pm.get_log_dir()
        return Path(self.resource_base_dir) / "logs"

    def apply_to_path_manager(self):
        """将配置应用到全局路径管理器"""
        pm = self._get_path_manager()
        if pm:
            if self.resource_base_dir:
                pm.set_resource_base(self.resource_base_dir)
            if self.tdrive_data_dir:
                pm.set_tdrive_data_dir(self.tdrive_data_dir)
            if self.tdrive_data_path:
                pm.set_tdrive_data_path(self.tdrive_data_path)


@dataclass
class IndexConfig:
    """四叉树与TShape空间索引配置
    
    Attributes:
        max_level: 四叉树的最大层级深度
        alpha: 扩大元素在X方向的单元数量
        beta: 扩大元素在Y方向的单元数量
        min_x: 原始边界框的最小X坐标（经度）
        min_y: 原始边界框的最小Y坐标（纬度）
        max_x: 原始边界框的最大X坐标（经度）
        max_y: 原始边界框的最大Y坐标（纬度）
        use_original_bbox: 是否使用原始边界框（而非归一化的[0,0,1,1]）
        use_prune: 是否启用节点剪枝逻辑
        min_cell_trajs: 节点内最少轨迹数阈值，低于此值的节点将被剪枝
        baseline_include_muted: baseline遍历顺序是否包含哑节点
        enable_sig_optimize: 是否启用轨迹签名优化
        parallel_signatures: 是否并行计算签名
        signature_workers: 签名计算的工作进程数
    """
    max_level: int = 8
    alpha: int = 2
    beta: int = 2

    # 空间边界配置
    min_x: float = 115.29  # 104.04
    min_y: float = 39.00   # 30.65
    max_x: float = 117.83  # 104.13
    max_y: float = 41.50   # 30.73
    use_original_bbox: bool = True

    # 剪枝与过滤
    use_prune: bool = True
    min_cell_trajs: int = 0
    baseline_include_muted: bool = False
    enable_sig_optimize: bool = False
    
    # 并行计算配置
    parallel_signatures: bool = True
    signature_workers: Optional[int] = None

    def get_bbox_tuple(self) -> Tuple[float, float, float, float]:
        """获取边界框元组
        
        Returns:
            (min_x, min_y, max_x, max_y) 四元组
        """
        return self.min_x, self.min_y, self.max_x, self.max_y


@dataclass
class DataConfig:
    """轨迹数据加载配置
    
    Attributes:
        num_trajectories: 加载的轨迹数量，-1表示加载全部
        use_tdrive_data: 是否使用TDrive数据集（False则使用合成数据）
        use_similarity_matrix: 是否使用预计算的相似度矩阵
        similarity_matrix_path: 相似度矩阵文件路径（相对于resource/shared/similarity/）
        similarity_num_workers: 计算相似度矩阵时的工作进程数（None表示使用CPU核心数）
        similarity_use_gpu: 是否使用GPU加速相似度矩阵计算
        similarity_gpu_batch_size: GPU计算批大小
        storage_mode: 轨迹存储模式 ('memory'/'disk'/'auto')
        storage_dir: 磁盘存储目录路径（storage_mode='disk'时必需）
        disk_cache_mb: 磁盘存储LRU缓存大小(MB)
        parallel_trajectory_assign: 是否并行分配轨迹
        trajectory_assign_workers: 轨迹分配工作进程数
        storage_mode: 轨迹存储模式 ('memory'/'disk'/'auto')
        storage_dir: 磁盘存储目录路径（storage_mode='disk'时必需）
        disk_cache_mb: 磁盘存储LRU缓存大小(MB)
    """
    num_trajectories: int = 3000
    use_tdrive_data: bool = True
    use_similarity_matrix: bool = True
    similarity_matrix_path: Optional[str] = None
    similarity_num_workers: Optional[int] = 4
    
    # GPU加速配置
    similarity_use_gpu: bool = True  # 自动检测GPU
    similarity_gpu_batch_size: int = 1024
    
    # 并行处理配置
    parallel_trajectory_assign: bool = True
    trajectory_assign_workers: Optional[int] = None
    
    # 轨迹存储配置
    storage_mode: str = "auto"  # 'memory', 'disk', 'auto'
    storage_dir: Optional[str] = None
    disk_cache_mb: int = 2048
    
    def get_similarity_matrix_path(self) -> Optional[Path]:
        """获取相似度矩阵完整路径"""
        if not self.similarity_matrix_path:
            return None
        path = Path(self.similarity_matrix_path)
        if path.is_absolute():
            return path
        # 相对路径相对于 resource/shared/similarity/
        return Path("resource/shared/similarity") / path


@dataclass
class RewardConfig:
    """奖励模型与评估配置
    
    Attributes:
        tau_loc: 定位成本系数（控制位置访问代价权重）
        tau_scan: 扫描成本系数（控制节点扫描代价权重）
        local_reward_weight: 局部奖励在总奖励中的权重
        global_reward_weight: 全局奖励在总奖励中的权重
        global_reward_scale: 全局奖励缩放倍数
        global_reward_num_evals: 全局奖励计算次数
        query_dataset_path: 查询数据集目录路径
        query_distribution_type: 查询分布类型（'uniform'/'skewed'/'gaussian'）
        train_val_test_split: 训练集/验证集/测试集划分比例（用于原始数据划分）
        query_sample_ratio: 从预划分文件中采样的比例
    """
    tau_loc: float = 1.0
    tau_scan: float = 0.1
    local_reward_weight: float = 0.4
    global_reward_weight: float = 1.0
    global_reward_scale: float = 2.0
    global_reward_num_evals: int = 1

    # 查询数据集配置
    query_dataset_path: str = "resource/queries"
    query_distribution_type: str = "skewed"
    query_sample_ratio: float = 1.0


@dataclass
class NetworkConfig:
    """神经网络架构配置
    
    Attributes:
        hidden_dims: 神经网络的隐藏层维度列表
        device: 设备选项（'auto'/'cuda'/'cpu'）
        dropout: Dropout比例
        state_dim: 状态特征维度
    """
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    device: str = 'auto'
    dropout: float = 0.1
    state_dim: int = 14

    def get_torch_device(self) -> torch.device:
        """获取PyTorch设备
        
        Returns:
            torch.device对象，自动检测CUDA可用性
        """
        if self.device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(self.device)


@dataclass
class TrainConfig:
    """训练流程配置
    
    Attributes:
        lr_actor: Actor网络学习率
        lr_critic: Critic网络学习率
        gamma: 折扣因子（discount factor）
        gae_lambda: GAE (Generalized Advantage Estimation) lambda参数，用于平衡偏差和方差
        num_episodes: 总训练轮数（episode数）
        eval_interval: 每隔多少episode执行一次完整评估
        save_interval: 每隔多少episode保存一次模型
        
        # PPO相关参数
        eps_clip: PPO裁剪阈值
        k_epochs: 每轮交互数据重复学习的次数
        gradient_clip_norm: 梯度裁剪阈值
        
        # 熵正则化参数
        entropy_coef_start: 训练初期的熵系数（鼓励探索）
        entropy_coef_end: 训练后期的熵系数（鼓励利用）
        entropy_decay_episodes: 熵系数衰减的episode数（-1表示不衰减，使用固定值entropy_coef_start）
        
        # 动作空间控制
        topk_actions: 若设定，则根据节点相似度筛选Top-K动作候选
        topk_decay_episodes: topK衰减episodes数
        topk_multipliers: (起始倍数, 结束倍数)，用于动态调整topk
        
        # 早停机制
        enable_early_stopping: 是否启用早停机制
        early_stopping_patience: 耐心值，连续多少轮无改进则停止
        early_stopping_min_delta: 最小改进阈值
        max_negative_streak: 最大连续负收益轮数
    """
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_episodes: int = 1000
    eval_interval: int = 20
    save_interval: int = 100

    # PPO相关参数
    eps_clip: float = 0.2
    k_epochs: int = 4
    gradient_clip_norm: float = 1.0

    # 熵正则化参数
    entropy_coef_start: float = 0.05
    entropy_coef_end: float = 0.001
    entropy_decay_episodes: int = -1  # -1表示使用固定值entropy_coef_start

    # 动作空间控制
    topk_actions: Optional[int] = None
    topk_decay_episodes: int = 100
    topk_multipliers: Tuple[float, float] = (2.0, 1.0)

    # 早停机制
    enable_early_stopping: bool = True
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 0.01
    max_negative_streak: int = 5


@dataclass
class TShapeConfig:
    """TShape索引系统的统一配置类
    
    这是配置系统的顶层类，整合了所有子配置模块。
    
    Attributes:
        index: 四叉树索引配置
        data: 数据加载配置
        reward: 奖励函数配置
        train: 训练流程配置
        network: 神经网络配置
        paths: 路径配置
    
    Examples:
        >>> # 从YAML文件加载
        >>> config = TShapeConfig.from_yaml('configs/default.yaml')
        >>> 
        >>> # 访问配置
        >>> max_level = config.index.max_level
        >>> num_trajs = config.data.num_trajectories
        >>> 
        >>> # 保存配置
        >>> config.save_yaml('configs/my_config.yaml')
    """
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    data: DataConfig = field(default_factory=DataConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    paths: PathConfig = field(default_factory=PathConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TShapeConfig':
        """从字典创建配置对象
        
        Args:
            data: 包含配置信息的字典
            
        Returns:
            TShapeConfig实例
        """
        config_dict = {}

        if 'experiment' in data:
            config_dict['experiment'] = ExperimentConfig(**data['experiment'])
        if 'index' in data:
            config_dict['index'] = IndexConfig(**data['index'])
        if 'data' in data:
            config_dict['data'] = DataConfig(**data['data'])
        if 'reward' in data:
            reward_data = data['reward'].copy()
            # 将list转换回tuple
            if 'train_val_test_split' in reward_data and isinstance(reward_data['train_val_test_split'], list):
                reward_data['train_val_test_split'] = tuple(reward_data['train_val_test_split'])
            config_dict['reward'] = RewardConfig(**reward_data)
        if 'train' in data:
            train_data = data['train'].copy()
            # 将list转换回tuple
            if 'topk_multipliers' in train_data and isinstance(train_data['topk_multipliers'], list):
                train_data['topk_multipliers'] = tuple(train_data['topk_multipliers'])
            config_dict['train'] = TrainConfig(**train_data)
        if 'network' in data:
            network_data = data['network'].copy()
            # 将list转换回tuple（如果hidden_dims需要是tuple）
            config_dict['network'] = NetworkConfig(**network_data)
        if 'paths' in data:
            config_dict['paths'] = PathConfig(**data['paths'])

        return cls(**config_dict)

    @classmethod
    def from_yaml(cls, path: str) -> 'TShapeConfig':
        """从YAML文件加载配置
        
        Args:
            path: YAML配置文件路径
            
        Returns:
            TShapeConfig实例
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_json(cls, path: str) -> 'TShapeConfig':
        """从JSON文件加载配置
        
        Args:
            path: JSON配置文件路径
            
        Returns:
            TShapeConfig实例
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典
        
        Returns:
            包含所有配置信息的字典
        """
        data = asdict(self)
        # 将tuple转换为list以支持YAML序列化
        self._convert_tuples_to_lists(data)
        return data

    @staticmethod
    def _convert_tuples_to_lists(obj):
        """递归地将字典中的tuple转换为list"""
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, tuple):
                    obj[key] = list(value)
                elif isinstance(value, dict):
                    TShapeConfig._convert_tuples_to_lists(value)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            TShapeConfig._convert_tuples_to_lists(item)

    def save_yaml(self, path: str):
        """保存为YAML文件
        
        Args:
            path: 保存路径
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)

    def save_json(self, path: str):
        """保存为JSON文件
        
        Args:
            path: 保存路径
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    def get_original_bbox(self):
        """获取原始边界框对象
        
        Returns:
            SpatialBoundingBox实例
        """
        from src.core.bounding_box import SpatialBoundingBox
        return SpatialBoundingBox(
            min_x=self.index.min_x,
            min_y=self.index.min_y,
            max_x=self.index.max_x,
            max_y=self.index.max_y
        )

    def get_original_bbox_tuple(self) -> Tuple[float, float, float, float]:
        """获取原始边界框的元组形式
        
        Returns:
            (min_x, min_y, max_x, max_y) 四元组
        """
        return self.index.get_bbox_tuple()

    # 向后兼容性属性
    @property
    def max_level(self) -> int:
        """向后兼容：访问index.max_level"""
        return self.index.max_level

    @property
    def alpha(self) -> int:
        """向后兼容：访问index.alpha"""
        return self.index.alpha

    @property
    def beta(self) -> int:
        """向后兼容：访问index.beta"""
        return self.index.beta

    @property
    def num_trajectories(self) -> int:
        """向后兼容：访问data.num_trajectories"""
        return self.data.num_trajectories

    @property
    def tau_loc(self) -> float:
        """向后兼容：访问reward.tau_loc"""
        return self.reward.tau_loc

    @property
    def num_episodes(self) -> int:
        """向后兼容：访问train.num_episodes"""
        return self.train.num_episodes

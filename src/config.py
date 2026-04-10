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
    config = TShapeConfig.from_yaml('default.yaml')
    
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


def _default_dataset_profiles() -> Dict[str, "DatasetProfileConfig"]:
    return {
        "tdrive": DatasetProfileConfig(
            description="Beijing TDrive trajectory dataset",
            trajectory_path=r"D:\dataset\Trajectory\TDrive\complete_clean\tdrive.txt",
            query_root="resource/queries",
            min_x=115.29,
            min_y=39.00,
            max_x=117.83,
            max_y=41.50,
        ),
        "cdtaxi": DatasetProfileConfig(
            description="Chengdu CDTaxi trajectory dataset",
            trajectory_path=r"D:\dataset\Trajectory\Chengdu\cleaned_cd_taxi.txt",
            query_root="resource/queries_chengdu",
            min_x=104.04,
            min_y=30.65,
            max_x=104.13,
            max_y=30.73,
        ),
    }


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

    def get_similarity_dir(self) -> Path:
        """获取实验私有相似度矩阵目录"""
        return self.get_output_dir() / "similarity"


@dataclass
class PathConfig:
    """路径配置
    
    Attributes:
        resource_base_dir: 资源文件基础目录（支持相对路径和绝对路径）
        
    注意：
        - 相对路径将相对于项目根目录解析
        - 可以通过环境变量覆盖：
          * PROJECT_ROOT: 项目根目录
          * RESOURCE_BASE_DIR: 资源基础目录
    """
    resource_base_dir: str = 'resource'

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


@dataclass
class DatasetProfileConfig:
    """单个数据集的路径与空间边界配置。"""

    description: str = ""
    trajectory_path: Optional[str] = None
    query_root: Optional[str] = None
    min_x: float = 0.0
    min_y: float = 0.0
    max_x: float = 1.0
    max_y: float = 1.0

    def get_bbox_tuple(self) -> Tuple[float, float, float, float]:
        return self.min_x, self.min_y, self.max_x, self.max_y


@dataclass
class DatasetCatalogConfig:
    """数据集目录表配置。"""

    active: str = "tdrive"
    profiles: Dict[str, DatasetProfileConfig] = field(default_factory=_default_dataset_profiles)


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
        quadcode_include_muted: quadCode遍历顺序是否包含哑节点
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
    quadcode_include_muted: bool = False
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
        source: 轨迹来源，'dataset' 使用当前激活数据集，'synthetic' 使用合成数据
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
    source: str = "dataset"
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
        from src.utils.path_manager import get_path_manager

        return get_path_manager().get_similarity_matrix_path(self.similarity_matrix_path)


@dataclass
class RewardConfig:
    """奖励模型与评估配置
    
    Attributes:
        tau_loc: 定位成本系数（控制位置访问代价权重）
        tau_scan: 扫描成本系数（控制节点扫描代价权重）
        local_reward_weight: 局部奖励在总奖励中的权重
        global_reward_weight: 全局奖励在总奖励中的权重
        reward_schedule_episodes: 奖励权重从预热过渡到目标权重的轮数
        global_reward_start_scale: 训练初期全局奖励权重缩放比例
        local_reward_start_scale: 训练初期局部奖励权重缩放比例
        global_reward_scale: 全局奖励缩放倍数
        global_reward_num_evals: 全局奖励计算次数
        global_reward_query_sample_size: 训练期每次全局奖励估计采样的查询数，None表示使用全部
        global_reward_frontload_exponent: 全局奖励checkpoint前置指数，越大越偏向前期触发
        query_distribution_type: 查询分布类型（'uniform'/'skewed'/'gaussian'）
        query_sample_ratio: 从预划分文件中采样的比例
    """
    tau_loc: float = 1.0
    tau_scan: float = 0.1
    local_reward_weight: float = 0.4
    global_reward_weight: float = 1.0
    reward_schedule_episodes: int = 100
    global_reward_start_scale: float = 0.25
    local_reward_start_scale: float = 1.0
    global_reward_scale: float = 2.0
    global_reward_num_evals: int = 1
    global_reward_query_sample_size: Optional[int] = 64
    global_reward_frontload_exponent: float = 1.5

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
        >>> config = TShapeConfig.from_yaml('default.yaml')
        >>> 
        >>> # 访问配置
        >>> max_level = config.index.max_level
        >>> num_trajs = config.data.num_trajectories
        >>> 
        >>> # 保存配置
        >>> config.save_yaml('configs/my_config.yaml')
    """
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    datasets: DatasetCatalogConfig = field(default_factory=DatasetCatalogConfig)
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
        if 'datasets' in data:
            datasets_data = data['datasets'].copy()
            profiles_data = datasets_data.get('profiles', {}) or {}
            datasets_data['profiles'] = {
                name: DatasetProfileConfig(**profile_data)
                for name, profile_data in profiles_data.items()
            }
            config_dict['datasets'] = DatasetCatalogConfig(**datasets_data)
        if 'index' in data:
            config_dict['index'] = IndexConfig(**data['index'])
        if 'data' in data:
            config_dict['data'] = DataConfig(**data['data'])
        if 'reward' in data:
            config_dict['reward'] = RewardConfig(**data['reward'])
        if 'train' in data:
            train_data = data['train'].copy()
            # 将list转换回tuple
            if 'topk_multipliers' in train_data and isinstance(train_data['topk_multipliers'], list):
                train_data['topk_multipliers'] = tuple(train_data['topk_multipliers'])
            config_dict['train'] = TrainConfig(**train_data)
        if 'network' in data:
            network_data = data['network'].copy()
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

    def _resolve_optional_path(self, path: Optional[str]) -> Optional[Path]:
        if not path:
            return None

        candidate = Path(path)
        if candidate.is_absolute():
            return candidate

        from src.utils.path_manager import get_path_manager

        return (get_path_manager().project_root / candidate).resolve()

    def get_active_dataset_profile(self) -> Optional[DatasetProfileConfig]:
        """获取当前激活的数据集配置。"""
        return self.datasets.profiles.get(self.datasets.active)

    def get_dataset_trajectory_path(self) -> Optional[Path]:
        """获取当前数据集轨迹文件路径。"""
        profile = self.get_active_dataset_profile()
        if profile and profile.trajectory_path:
            return self._resolve_optional_path(profile.trajectory_path)
        return None

    def get_query_dataset_root(self) -> Path:
        """获取当前数据集对应的查询集根目录。"""
        profile = self.get_active_dataset_profile()
        if profile and profile.query_root:
            resolved = self._resolve_optional_path(profile.query_root)
            if resolved is not None:
                return resolved
        raise ValueError(f"当前数据集 {self.datasets.active} 未配置 query_root")

    def get_effective_bbox_tuple(self) -> Tuple[float, float, float, float]:
        """获取当前有效边界框，优先使用激活数据集的边界。"""
        profile = self.get_active_dataset_profile()
        if profile is not None:
            return profile.get_bbox_tuple()
        return self.index.get_bbox_tuple()

    def get_original_bbox(self):
        """获取原始边界框对象
        
        Returns:
            SpatialBoundingBox实例
        """
        from src.core.bounding_box import SpatialBoundingBox
        min_x, min_y, max_x, max_y = self.get_effective_bbox_tuple()
        return SpatialBoundingBox(
            min_x=min_x,
            min_y=min_y,
            max_x=max_x,
            max_y=max_y
        )

    def get_original_bbox_tuple(self) -> Tuple[float, float, float, float]:
        """获取原始边界框的元组形式
        
        Returns:
            (min_x, min_y, max_x, max_y) 四元组
        """
        return self.get_effective_bbox_tuple()

    def get_default_similarity_matrix_filename(self) -> str:
        """获取默认相似度矩阵文件名。"""
        return (
            f"sim_mtx_{self.datasets.active}_"
            f"L{self.index.max_level}_"
            f"A{self.index.alpha}_"
            f"B{self.index.beta}_"
            f"T{self.data.num_trajectories}.npz"
        )

    def get_default_similarity_matrix_path(self) -> Path:
        """获取默认共享相似度矩阵路径。"""
        from src.utils.path_manager import get_path_manager

        return get_path_manager().get_similarity_matrix_path(
            self.get_default_similarity_matrix_filename()
        )

    def get_effective_similarity_matrix_path(self) -> Path:
        """获取当前配置实际使用的相似度矩阵路径。"""
        explicit_path = self.data.get_similarity_matrix_path()
        if explicit_path is not None:
            return explicit_path
        return self.get_default_similarity_matrix_path()

    def get_experiment_similarity_matrix_path(self) -> Path:
        """获取当前实验的私有相似度矩阵路径。"""
        return self.experiment.get_similarity_dir() / self.get_default_similarity_matrix_filename()


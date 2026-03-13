"""
统一路径管理模块

提供项目中所有路径的统一管理，支持：
1. 自动检测项目根目录
2. 环境变量配置
3. 相对路径和绝对路径的灵活处理
4. 路径验证和创建

使用示例：
    # 在 src 模块中使用
    from src.utils.path_manager import get_path_manager
    
    pm = get_path_manager()
    model_dir = pm.get_model_dir('uniform')
    tdrive_path = pm.tdrive_data_path
    
    # 在 scripts 脚本中使用
    import scripts  # 自动初始化项目路径
    from src.utils.path_manager import get_path_manager
    
    pm = get_path_manager()
    # ... 使用路径管理器
"""
import os
from pathlib import Path
from typing import Optional, Union


class PathManager:
    """路径管理器，单例模式"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._project_root: Optional[Path] = None
            self._resource_base: Optional[Path] = None
            self._tdrive_data_dir: Optional[Path] = None
            self._tdrive_data_path: Optional[Path] = None
            self._experiment_name: Optional[str] = None  # 当前实验名称
            PathManager._initialized = True
    
    @property
    def project_root(self) -> Path:
        """获取项目根目录"""
        if self._project_root is None:
            self._project_root = self._detect_project_root()
        return self._project_root
    
    @staticmethod
    def _detect_project_root() -> Path:
        """自动检测项目根目录
        
        检测策略：
        1. 从当前文件向上查找包含 'src' 目录的父目录
        2. 检查环境变量 PROJECT_ROOT
        3. 使用当前工作目录
        """
        # 策略1: 从环境变量获取
        if 'PROJECT_ROOT' in os.environ:
            root = Path(os.environ['PROJECT_ROOT'])
            if root.exists() and (root / 'src').exists():
                return root.resolve()
        
        # 策略2: 从当前文件向上查找
        current = Path(__file__).resolve()
        for parent in [current] + list(current.parents):
            if (parent / 'src').exists() and (parent / 'src').is_dir():
                return parent
        
        # 策略3: 使用当前工作目录
        cwd = Path.cwd()
        if (cwd / 'src').exists():
            return cwd
        
        # 如果都失败，返回当前工作目录
        return cwd
    
    def set_project_root(self, path: Union[str, Path]) -> None:
        """手动设置项目根目录"""
        self._project_root = Path(path).resolve()
    
    @property
    def resource_base(self) -> Path:
        """获取资源基础目录"""
        if self._resource_base is None:
            # 从环境变量获取，或使用默认值
            base = os.environ.get('RESOURCE_BASE_DIR', 'resource')
            self._resource_base = self._resolve_path(base)
        return self._resource_base
    
    def set_resource_base(self, path: Union[str, Path]) -> None:
        """设置资源基础目录"""
        self._resource_base = self._resolve_path(path)
    
    @property
    def tdrive_data_dir(self) -> Optional[Path]:
        """获取TDrive数据集目录"""
        if self._tdrive_data_dir is None:
            # 从环境变量获取
            if 'TDRIVE_DATA_DIR' in os.environ:
                self._tdrive_data_dir = Path(os.environ['TDRIVE_DATA_DIR']).resolve()
        return self._tdrive_data_dir
    
    def set_tdrive_data_dir(self, path: Union[str, Path]) -> None:
        """设置TDrive数据集目录"""
        self._tdrive_data_dir = Path(path).resolve()
    
    @property
    def tdrive_data_path(self) -> Optional[Path]:
        """获取TDrive数据集文件路径"""
        if self._tdrive_data_path is None:
            # 从环境变量获取
            if 'TDRIVE_DATA_PATH' in os.environ:
                self._tdrive_data_path = Path(os.environ['TDRIVE_DATA_PATH']).resolve()
            # 或从数据目录推断
            elif self.tdrive_data_dir is not None:
                candidate = self.tdrive_data_dir / 'tdrive_cleaned.txt'
                if candidate.exists():
                    self._tdrive_data_path = candidate
        return self._tdrive_data_path
    
    def set_tdrive_data_path(self, path: Union[str, Path]) -> None:
        """设置TDrive数据集文件路径"""
        self._tdrive_data_path = Path(path).resolve()
    
    @property
    def experiment_name(self) -> Optional[str]:
        """获取当前实验名称"""
        return self._experiment_name
    
    def set_experiment_name(self, name: str) -> None:
        """设置当前实验名称"""
        self._experiment_name = name
    
    def get_experiment_dir(self, experiment_name: Optional[str] = None) -> Path:
        """获取实验目录
        
        Args:
            experiment_name: 实验名称，如果为None则使用当前设置的实验名称
            
        Returns:
            实验目录路径
        """
        name = experiment_name or self._experiment_name or 'default'
        exp_dir = self.resource_base / 'experiments' / name
        exp_dir.mkdir(parents=True, exist_ok=True)
        return exp_dir
    
    def _resolve_path(self, path: Union[str, Path]) -> Path:
        """解析路径，支持相对路径和绝对路径
        
        Args:
            path: 路径字符串或Path对象
            
        Returns:
            解析后的绝对路径
        """
        path = Path(path)
        if path.is_absolute():
            return path.resolve()
        else:
            return (self.project_root / path).resolve()
    
    def get_model_dir(self, experiment_name: Optional[str] = None) -> Path:
        """获取模型保存目录
        
        Args:
            experiment_name: 实验名称（可选），如果为None则使用当前设置的实验名称
            
        Returns:
            模型目录路径
        """
        name = experiment_name or self._experiment_name
        if name:
            base = self.get_experiment_dir(name) / 'models'
        else:
            base = self.resource_base / 'models'
        base.mkdir(parents=True, exist_ok=True)
        return base
    
    def get_order_dir(self, experiment_name: Optional[str] = None) -> Path:
        """获取遍历顺序保存目录
        
        Args:
            experiment_name: 实验名称（可选），如果为None则使用当前设置的实验名称
            
        Returns:
            顺序目录路径
        """
        name = experiment_name or self._experiment_name
        if name:
            base = self.get_experiment_dir(name) / 'orders'
        else:
            base = self.resource_base / 'orders'
        base.mkdir(parents=True, exist_ok=True)
        return base
    
    def get_similarity_dir(self) -> Path:
        """获取相似度矩阵保存目录（共享资源）"""
        base = self.resource_base / 'shared' / 'similarity'
        base.mkdir(parents=True, exist_ok=True)
        return base
    
    def get_log_dir(self, experiment_name: Optional[str] = None) -> Path:
        """获取日志保存目录
        
        Args:
            experiment_name: 实验名称（可选），如果为None则使用当前设置的实验名称
            
        Returns:
            日志目录路径
        """
        name = experiment_name or self._experiment_name
        if name:
            base = self.get_experiment_dir(name) / 'logs'
        else:
            base = self.resource_base / 'logs'
        base.mkdir(parents=True, exist_ok=True)
        return base
    
    def get_queries_dir(self, experiment_name: Optional[str] = None) -> Path:
        """获取查询集保存目录
        
        Args:
            experiment_name: 实验名称（可选），如果为None则使用当前设置的实验名称
            
        Returns:
            查询集目录路径
        """
        name = experiment_name or self._experiment_name
        if name:
            base = self.get_experiment_dir(name) / 'queries'
        else:
            base = self.resource_base / 'queries'
        base.mkdir(parents=True, exist_ok=True)
        return base
        """获取训练曲线保存目录
        
        Args:
            experiment_name: 实验名称（可选），如果为None则使用当前设置的实验名称
            
        Returns:
            曲线目录路径
        """
        name = experiment_name or self._experiment_name
        if name:
            base = self.get_experiment_dir(name) / 'curves'
        else:
            base = self.resource_base / 'curves'
        base.mkdir(parents=True, exist_ok=True)
        return base
    
    def get_similarity_matrix_path(self, filename: str) -> Path:
        """获取相似度矩阵文件路径
        
        Args:
            filename: 文件名或相对路径
            
        Returns:
            完整的文件路径
        """
        # 如果是绝对路径，直接返回
        path = Path(filename)
        if path.is_absolute():
            return path
        
        # 如果包含路径分隔符，相对于项目根目录
        if '/' in filename or '\\' in filename:
            return self._resolve_path(filename)
        
        # 否则相对于相似度目录
        return self.get_similarity_dir() / filename
    
    def ensure_dir(self, path: Union[str, Path]) -> Path:
        """确保目录存在
        
        Args:
            path: 目录路径
            
        Returns:
            目录路径
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def validate_path(self, path: Union[str, Path], must_exist: bool = False) -> bool:
        """验证路径
        
        Args:
            path: 要验证的路径
            must_exist: 是否必须存在
            
        Returns:
            路径是否有效
        """
        if path is None:
            return False
        
        path = Path(path)
        if must_exist:
            return path.exists()
        
        # 检查路径格式是否有效
        try:
            path.resolve()
            return True
        except (OSError, RuntimeError):
            return False


# 全局单例实例
_path_manager = PathManager()


def get_path_manager() -> PathManager:
    """获取全局路径管理器实例"""
    return _path_manager


def get_project_root() -> Path:
    """快捷方法：获取项目根目录"""
    return _path_manager.project_root

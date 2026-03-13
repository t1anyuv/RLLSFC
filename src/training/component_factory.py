"""训练组件工厂。"""
import logging
import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

from src.config import TShapeConfig
from src.core.bounding_box import SpatialBoundingBox
from src.data import SyntheticTrajectoryFactory, normalize_trajectories, load_cleaned_dataset
from src.indexing import QuadTreeIndex
from src.reward import TraversalCostEvaluator
from src.rl import TraversalEnvironment
from src.storage import create_storage
from src.utils.path_manager import get_path_manager
from src.utils.similarity_matrix import SimilarityMatrix


class TrainingComponentFactory:
    """训练组件工厂类。
    
    负责创建和初始化训练所需的各种组件，减少主训练类的复杂度。
    """

    def __init__(self, config: TShapeConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def create_quadtree(self) -> QuadTreeIndex:
        """根据配置初始化边界框并构建四叉树索引。
        
        根据data.storage_mode配置自动选择存储后端。
        
        返回:
            四叉树索引实例
        """
        if self.config.index.use_original_bbox:
            bbox = self.config.get_original_bbox()
        else:
            bbox = SpatialBoundingBox(0.0, 0.0, 1.0, 1.0)
        
        # 创建存储后端
        storage = create_storage(
            mode=self.config.data.storage_mode,
            storage_dir=self.config.data.storage_dir,
            cache_mb=self.config.data.disk_cache_mb,
        )
        
        return QuadTreeIndex(
            bbox,
            max_level=self.config.index.max_level,
            alpha=self.config.index.alpha,
            beta=self.config.index.beta,
            storage=storage,
            parallel_signatures=self.config.index.parallel_signatures,
            signature_workers=self.config.index.signature_workers
        )

    def load_trajectories(
            self,
            quadtree: QuadTreeIndex
    ) -> List[Tuple[int, List[Tuple[float, float]]]]:
        """加载轨迹数据，支持TDrive真实数据加载与合成数据生成。
        
        参数:
            quadtree: 四叉树索引
            
        返回:
            轨迹数据列表
        """
        bbox = quadtree.bbox

        if self.config.data.use_tdrive_data and self.config.paths.tdrive_data_dir:
            print("加载 TDrive 真实轨迹数据...")
            trajectories = load_cleaned_dataset(
                self.config.paths.tdrive_data_path,
                max_trajectories=self.config.data.num_trajectories if self.config.data.num_trajectories > 0 else None,
            )

            if not trajectories:
                print("未找到有效 TDrive 数据，回退至合成数据。")
                trajectories = SyntheticTrajectoryFactory.generate(self.config.data.num_trajectories, bbox)
            elif not self.config.index.use_original_bbox:
                print("执行轨迹归一化...")
                trajectories = normalize_trajectories(trajectories, (0.0, 0.0, 1.0, 1.0))
        else:
            print("生成合成轨迹数据...")
            trajectories = SyntheticTrajectoryFactory.generate(self.config.data.num_trajectories, bbox)

        return trajectories

    def create_environment(
            self,
            quadtree: QuadTreeIndex,
            cost_evaluator: TraversalCostEvaluator
    ) -> TraversalEnvironment:
        """创建遍历环境。
        
        参数:
            quadtree: 四叉树索引
            cost_evaluator: 成本评估器
            
        返回:
            遍历环境实例
        """
        # 加载查询数据集并划分
        train_queries, val_queries, test_queries = self._load_and_split_queries()

        # 构建环境实例（使用训练集计算全局奖励）
        env = TraversalEnvironment(
            quadtree=quadtree,
            cost_evaluator=cost_evaluator,
            reference_queries=train_queries,
            alpha=self.config.index.alpha,
            beta=self.config.index.beta,
            exclude_muted_cells=self.config.index.use_prune,
            baseline_include_muted=self.config.index.baseline_include_muted,
            local_reward_weight=self.config.reward.local_reward_weight,
            global_reward_weight=self.config.reward.global_reward_weight,
            global_reward_num_evals=self.config.reward.global_reward_num_evals,
        )

        # 将验证集和测试集附加到环境
        env.val_queries = val_queries
        env.test_queries = test_queries

        return env

    def _load_and_split_queries(self) -> Tuple[List, List, List]:
        """加载查询数据集并划分为训练集/验证集/测试集"""
        from pathlib import Path

        dataset_path = Path(self.config.reward.query_dataset_path)
        if not dataset_path.is_absolute():
            # 相对路径，相对于项目根目录
            project_root = Path(__file__).resolve().parent.parent.parent
            dataset_path = project_root / dataset_path

        dist_type = self.config.reward.query_distribution_type

        # 检查是否存在预划分的查询集目录
        category_dir = dataset_path / dist_type
        if category_dir.exists() and category_dir.is_dir():
            self.logger.info(f"检测到预划分查询集目录: {category_dir}")
            return self._load_pre_split_queries(category_dir, dist_type)

        raise FileNotFoundError(
            f"未找到预划分的查询数据集目录: {dataset_path / dist_type}\n"
            f"请先生成查询数据集: python -m scripts.preprocess.generate_query_dataset"
        )

    def _load_pre_split_queries(self, category_dir: Path, dist_type: str) -> Tuple[List, List, List]:
        """从预划分目录加载训练集/验证集/测试集 (JSON 格式)"""
        import json

        def load_json_queries(file_path: Path) -> List:
            """加载 JSON 查询文件"""
            if not file_path.exists():
                raise FileNotFoundError(f"查询文件不存在: {file_path}")

            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            queries = []
            for item in data:
                query_str = item['query']
                coords = [float(x.strip()) for x in query_str.split(',')]
                queries.append(SpatialBoundingBox(coords[0], coords[1], coords[2], coords[3]))

            return queries

        # 加载各类查询集
        train_queries = load_json_queries(category_dir / "queries_train.json")
        val_queries = load_json_queries(category_dir / "queries_val.json")
        test_queries = load_json_queries(category_dir / "queries_test.json")

        # 支持按比例采样
        sample_ratio = self.config.reward.query_sample_ratio
        if sample_ratio < 1.0:
            random.seed(42)  # 固定随机种子保证可重复性
            
            # 从训练集采样
            train_sample_size = int(len(train_queries) * sample_ratio)
            if train_sample_size < len(train_queries):
                train_queries = random.sample(train_queries, train_sample_size)
            
            # 从验证集采样
            val_sample_size = int(len(val_queries) * sample_ratio)
            if val_sample_size < len(val_queries):
                val_queries = random.sample(val_queries, val_sample_size)
            
            print(f"  - 按 {sample_ratio * 100:.0f}% 采样: 训练集={len(train_queries)}, 验证集={len(val_queries)}")

        total_loaded = len(train_queries) + len(val_queries) + len(test_queries)
        print(f"加载 {dist_type} 预划分查询集: 总计 {total_loaded} 个")
        print(f"  - 训练集: {len(train_queries)} 个")
        print(f"  - 验证集: {len(val_queries)} 个")
        print(f"  - 测试集: {len(test_queries)} 个")

        return train_queries, val_queries, test_queries

    def setup_similarity_matrix(
            self,
            quadtree: QuadTreeIndex,
            cost_evaluator: TraversalCostEvaluator,
            all_cells: List
    ) -> Optional[SimilarityMatrix]:
        """配置、加载或计算轨迹相似度矩阵。
        
        参数:
            quadtree: 四叉树索引
            cost_evaluator: 成本评估器
            all_cells: 所有单元格列表
            
        返回:
            相似度矩阵实例，如果未启用则返回 None
        """
        if not self.config.data.use_similarity_matrix:
            return None

        similarity_matrix = SimilarityMatrix(quadtree, cost_evaluator)
        pm = get_path_manager()

        if self.config.data.similarity_matrix_path:
            matrix_path = self.config.data.get_similarity_matrix_path()
        else:
            sim_dir = pm.get_similarity_dir()
            # 生成默认文件名
            matrix_path = sim_dir / f"similarity_matrix_L{self.config.index.max_level}_A{self.config.index.alpha}_B{self.config.index.beta}_M{self.config.index.min_cell_trajs or 0}_T{self.config.data.num_trajectories}.npz"
            print(f"使用默认相似度矩阵路径: {matrix_path}")

        if os.path.exists(matrix_path):
            print(f"加载相似度矩阵: {matrix_path}")
            if not similarity_matrix.load(matrix_path, all_cells):
                print("矩阵维度不匹配，重新计算...")
                self._compute_and_save_similarity_matrix(similarity_matrix, matrix_path, all_cells)
        else:
            self._compute_and_save_similarity_matrix(similarity_matrix, matrix_path, all_cells)

        return similarity_matrix

    def _compute_and_save_similarity_matrix(
            self,
            similarity_matrix: SimilarityMatrix,
            matrix_path: str,
            all_cells: List
    ) -> None:
        """计算并持久化相似度矩阵。
        
        参数:
            similarity_matrix: 相似度矩阵实例
            matrix_path: 保存路径
            all_cells: 所有单元格列表
        """
        print(f"开始计算相似度矩阵 (节点数: {len(all_cells)})...")
        num_workers = self.config.data.similarity_num_workers
        similarity_matrix.compute(all_cells, use_symmetric=True, show_progress=True, num_workers=num_workers)
        similarity_matrix.save(matrix_path)

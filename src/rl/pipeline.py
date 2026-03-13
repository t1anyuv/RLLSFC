import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

from src.config import NetworkConfig, TShapeConfig
from src.indexing.quadtree_cell import QuadTreeCell
from src.utils.path_manager import get_path_manager
from src.training import TraversalTrainer
from src.evaluation.lsfc_evaluator import LSFCEvaluator


class LSFCPipeLine:
    """
    资源中心与流程调度器：负责环境初始化、模型训练、权重加载。
    """

    def __init__(self,
                 config: TShapeConfig,
                 network_config: NetworkConfig,
                 logger: Optional[logging.Logger] = None,
                 resource_base_dir: Optional[str] = None,
                 custom_paths: Optional[Dict[str, str]] = None):
        self.config = config
        self.network_config = network_config
        self.logger = logger or logging.getLogger("RLPipeline")

        # 初始化路径
        pm = get_path_manager()
        self.resource_paths = {
            "models": self.config.experiment.get_models_dir(),
            "orders": self.config.experiment.get_orders_dir(),
            "logs": self.config.experiment.get_logs_dir(),
        }
        if custom_paths:
            for key, val in custom_paths.items():
                if key.endswith("_dir"):
                    name = key[:-4] + "s"
                    self.resource_paths[name] = Path(val)
                    self.resource_paths[name].mkdir(parents=True, exist_ok=True)

        # 核心组件
        self.trainer: Optional[TraversalTrainer] = None
        self.postprocessor: Optional[LSFCEvaluator] = None

    def initialize_components(self) -> None:
        """端到端环境 setup。"""
        self.logger.info("正在初始化流水线环境...")

        # 1. 训练器初始化 (执行 quadtree 构建, 轨迹分配, 剪枝)
        self.trainer = TraversalTrainer(self.config, self.network_config)
        self.trainer.setup()
        self.trainer.prepare_agent()

        # 2. 后处理器初始化
        self.postprocessor = LSFCEvaluator(
            config=self.config,
            output_dir=str(self.resource_paths["orders"])
        )

        self.logger.info(f"初始化完成。活跃单元格: {self.trainer.environment.num_cells}")

    def train_model(self) -> str:
        """执行训练循环并返回最终模型路径。"""
        if not self.trainer:
            self.initialize_components()

        self.logger.info("开始强化学习训练流程...")
        self.trainer.train()

        # 获取最终保存路径
        final_path = self.config.experiment.get_models_dir() / "final.pth"

        # 训练结束后，同步更新 Postprocessor 的最佳模型路径
        if self.postprocessor:
            self.postprocessor.best_model_path = Path(final_path)

        return str(final_path)

    def load_trained_model(self, checkpoint_path: str) -> None:
        """加载权重以供推理或评估。"""
        if not self.trainer:
            self.initialize_components()
        self.logger.info(f"加载模型: {checkpoint_path}")
        self.trainer.prepare_agent(model_path=checkpoint_path)

    def generate_learned_order(self) -> List[QuadTreeCell]:
        """执行推断逻辑。"""
        if not self.trainer or not self.trainer.agent:
            raise RuntimeError("Trainer 或 Agent 未就绪")
        return self.trainer.rollout_policy_order(self.trainer.agent, self.trainer.environment)

    def run_post_evaluation(self, model_path: str, export_prefix: str):
        """
        处理筛选与导出流程。
        """
        if not self.postprocessor:
            self.initialize_components()

        self.load_trained_model(model_path)
        order = self.generate_learned_order()
        self.postprocessor.process_learned_order(order, self.trainer.quadtree)
        
        # 使用环境中的测试集进行评估
        test_queries = self.trainer.environment.test_queries if hasattr(self.trainer.environment, 'test_queries') else None
        if test_queries:
            evaluator = self.postprocessor.get_evaluator(self.trainer)
            self.postprocessor.processing_metadata["evaluation"] = evaluator.compute_hgs_score(
                order, test_queries
            )
        
        self.postprocessor.best_model_path = Path(model_path)

        # 统一导出
        return self.postprocessor.export_formats(self.trainer, base_prefix=export_prefix)

    def run_full_pipeline(self, export_prefix: str = "rl_tshape") -> Dict[str, Any]:
        """
        端到端全流程：训练 -> 推理 -> 评估 -> 导出。
        """
        self.logger.info(">>> 启动端到端全流水线 <<<")

        # 1. 训练
        self.train_model()

        # 2. 检查是否有最佳模型记录，使用最佳模型进行后处理
        orders_dir = self.config.experiment.get_orders_dir()
        best_record_path = orders_dir / "best_model_record.json"
        
        if best_record_path.exists():
            # 加载最佳模型记录
            import json
            with open(best_record_path, 'r', encoding='utf-8') as f:
                best_record = json.load(f)
            model_path = best_record['best_model_path']
            self.logger.info(f"使用最佳模型进行后处理: {model_path}")
        else:
            # 回退到最终模型
            model_path = str(self.config.experiment.get_models_dir() / "final.pth")
            self.logger.info(f"使用最终模型进行后处理: {model_path}")

        # 3. 加载最佳模型并生成遍历顺序
        self.load_trained_model(model_path)
        learned_order = self.generate_learned_order()

        # 4. 使用环境中的查询集进行评估
        train_queries = self.trainer.environment.reference_queries
        val_queries = self.trainer.environment.val_queries
        test_queries = self.trainer.environment.test_queries
        
        if train_queries is None or val_queries is None or test_queries is None:
            raise RuntimeError("训练集、验证集或测试集不可用")

        # 5. 在 train、val 和 test 上评估
        evaluator_train = self.postprocessor._create_evaluator(self.trainer, train_queries)
        train_metrics = evaluator_train.evaluate_final_order(learned_order)
        
        evaluator_val = self.postprocessor._create_evaluator(self.trainer, val_queries)
        val_metrics = evaluator_val.evaluate_final_order(learned_order)
        
        evaluator_test = self.postprocessor._create_evaluator(self.trainer, test_queries)
        test_metrics = evaluator_test.evaluate_final_order(learned_order)
        
        hgs_score = (val_metrics['improvement_percent'] + test_metrics['improvement_percent']) / 2

        # 6. 保存评估结果
        self.postprocessor.process_learned_order(learned_order, self.trainer.quadtree)
        self.postprocessor.best_model_path = Path(model_path)
        self.postprocessor.processing_metadata["evaluation"] = {
            'i_val': val_metrics['improvement_percent'],
            'i_test': test_metrics['improvement_percent'],
            'hgs_score': hgs_score,
            'train_improvement': train_metrics['improvement_percent'],
            'val_improvement': val_metrics['improvement_percent'],
            'test_improvement': test_metrics['improvement_percent'],
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics
        }

        export_results = self.postprocessor.export_formats(self.trainer, base_prefix=export_prefix)

        return {
            "model_path": model_path,
            "learned_order_length": len(learned_order),
            "improvement_rate": test_metrics['improvement_percent'],
            "quadtree_stats": self.postprocessor.processing_metadata["quadtree_stats"],
            "export_results": export_results,
            "summary_report": self.postprocessor.get_summary_report()
        }

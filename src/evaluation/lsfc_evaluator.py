import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from src.config import TShapeConfig
from src.indexing import QuadTreeIndex
from src.training import TraversalTrainer
from src.utils.path_manager import get_path_manager


class LSFCEvaluator:
    """
    统一的RL后处理器：专注于模型性能评估、多指标计算及结果导出。
    """

    def __init__(self,
                 config: TShapeConfig,
                 output_dir: Optional[str] = None,
                 enable_logging: bool = True):
        # 初始化配置、日志、输出路径
        self.config = config
        self.enable_logging = enable_logging
        self.logger = logging.getLogger("UnifiedPostprocessor")
        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.output_dir = config.experiment.get_orders_dir()

        # 初始化底层导出器
        from src.rl.order_formatter import TrajectoryOrderFormatter
        self.trajectory_processor = TrajectoryOrderFormatter(
            output_dir=str(self.output_dir),
            config=self.config,
        )

        # 状态缓存
        self.learned_order: Optional[List[Any]] = None
        self.best_model_path: Optional[Path] = None
        self.processing_metadata: Dict[str, Any] = {}

    def select_best_model_from_metrics(self, model_dir: str) -> Tuple[Optional[Path], Optional[Dict]]:
        """
        从保存的 metrics 文件中选择 HGS 得分最高的模型。
        
        不再遍历所有 checkpoint 重新评估，而是直接读取 epXXXXXX_metrics.json 文件计算 HGS。
        
        参数:
            model_dir: 模型目录路径
            
        返回:
            (最佳模型路径, 最佳记录字典)
        """
        model_dir = Path(model_dir)
        metrics_files = sorted(model_dir.glob("ep*_metrics.json"))
        
        if not metrics_files:
            self.logger.warning(f"在 {model_dir} 中未找到任何 metrics 文件")
            return None, None
        
        best_hgs = -float('inf')
        best_record = None
        
        print(f"\n{'Episode':<10} | {'Train Imp':<10} | {'Val Imp':<10} | {'Test Imp':<10} | {'HGS Score':<10}")
        print("-" * 70)
        
        for metrics_file in metrics_files:
            try:
                with open(metrics_file, 'r', encoding='utf-8') as f:
                    metrics = json.load(f)
                
                episode = metrics.get('episode', 0)
                val_imp = metrics.get('val', {}).get('improvement_percent', 0)
                test_imp = metrics.get('test', {}).get('improvement_percent', 0)
                train_imp = metrics.get('train', {}).get('improvement_percent', 0) if metrics.get('train') else 0
                
                # 计算 HGS 得分: (I_val + I_test) / 2
                hgs_score = (val_imp + test_imp) / 2
                
                print(f"{episode:<10} | {train_imp:>+9.2f}% | {val_imp:>+9.2f}% | {test_imp:>+9.2f}% | {hgs_score:>9.4f}")
                
                if hgs_score > best_hgs:
                    best_hgs = hgs_score
                    # 构建对应的模型文件名
                    model_file = model_dir / f"ep{episode:06d}_val{val_imp:+.2f}.pth"
                    if not model_file.exists():
                        # 回退查找任意匹配 episode 的模型文件
                        model_file = model_dir / f"model_ep_{episode}.pth"
                    
                    best_record = {
                        "model_path": model_file,
                        "metrics_file": metrics_file,
                        "episode": episode,
                        "metrics": metrics,
                        "hgs_score": hgs_score,
                        "val_improvement": val_imp,
                        "test_improvement": test_imp,
                        "train_improvement": train_imp,
                    }
            except Exception as e:
                self.logger.warning(f"读取 {metrics_file} 失败: {e}")
                continue
        
        print("-" * 70)
        
        if best_record:
            print(f"[*] 最佳模型: Episode {best_record['episode']}, HGS={best_hgs:.4f}")
            print(f"    路径: {best_record['model_path']}\n")
        
        return best_record["model_path"] if best_record else None, best_record
    
    def process_best_model(self, model_dir: str, pipeline: Any) -> Dict[str, Any]:
        """
        从 metrics 文件中选择最佳模型并处理后导出。
        
        参数:
            model_dir: 模型目录路径
            pipeline: LSFCPipeLine 实例
            
        返回:
            处理结果字典
        """
        # 1. 选择最佳模型
        best_model_path, best_record = self.select_best_model_from_metrics(model_dir)
        
        if best_model_path is None or not best_model_path.exists():
            raise FileNotFoundError(f"未找到有效的最佳模型: {best_model_path}")
        
        # 2. 加载模型并生成遍历顺序
        pipeline.load_trained_model(str(best_model_path))
        pipeline.trainer.agent.actor.eval()
        learned_order = pipeline.generate_learned_order()
        
        # 3. 更新状态
        self.best_model_path = best_model_path
        self.learned_order = learned_order
        self.processing_metadata["evaluation"] = {
            'i_val': best_record['val_improvement'],
            'i_test': best_record['test_improvement'],
            'i_train': best_record['train_improvement'],
            'hgs_score': best_record['hgs_score'],
            'episode': best_record['episode'],
            'metrics': best_record['metrics']
        }
        self.processing_metadata.update({
            "timestamp": datetime.now().isoformat(),
            "learned_order_length": len(learned_order),
            "quadtree_stats": pipeline.trainer.quadtree.get_quadtree_stats(),
        })
        
        # 4. 导出结果
        export_results = self.export_formats(pipeline.trainer, base_prefix="best_order")
        
        # 5. 保存最佳模型记录
        self._save_best_model_record(best_record)
        
        return {
            "best_model_path": str(best_model_path),
            "hgs_score": best_record['hgs_score'],
            "export_results": export_results,
            "summary_report": self.get_summary_report()
        }
    
    def _save_best_model_record(self, best_record: Dict) -> None:
        """保存最佳模型记录。"""
        record_path = self.output_dir / "best_model_record.json"
        with open(record_path, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "best_model_path": str(best_record['model_path']),
                "metrics_file": str(best_record['metrics_file']),
                "episode": best_record['episode'],
                "hgs_score": best_record['hgs_score'],
                "val_improvement": best_record['val_improvement'],
                "test_improvement": best_record['test_improvement'],
                "train_improvement": best_record['train_improvement'],
            }, f, indent=2, ensure_ascii=False)
        self.logger.info(f"最佳模型记录已保存: {record_path}")
    
    def _create_evaluator(self, trainer, queries):
        """创建评估器"""
        from src.evaluation.traversal_evaluator import TraversalPerformanceEvaluator
        return TraversalPerformanceEvaluator(
            trainer.quadtree, trainer.encoder, trainer.cost_evaluator,
            reference_queries=queries,
            baseline_include_muted=self.config.index.baseline_include_muted
        )

    def process_learned_order(self, learned_order: List[Any], quadtree: QuadTreeIndex):
        """缓存推理结果并记录元数据。"""
        self.learned_order = learned_order
        self.processing_metadata.update({
            "timestamp": datetime.now().isoformat(),
            "learned_order_length": len(learned_order),
            "quadtree_stats": quadtree.get_quadtree_stats(),
            "config_summary": self.get_config_summary(),
        })

    def export_formats(self, trainer: TraversalTrainer, base_prefix: str = "learned_order") -> Dict[str, Any]:
        """导出多格式文件并触发报告打印。"""
        if not self.learned_order:
            raise RuntimeError("导出前必须先执行推断或 process_learned_order")

        environment = trainer.environment
        environment.visited_order = self.learned_order

        data, json_path = self.trajectory_processor.generate_config_file(
            environment, trainer.quadtree,
            filename=f"{base_prefix}.json"
        )

        print(self.get_summary_report())
        return {"json": json_path}

    def export_final_config(self, trainer, experiment_name: Optional[str] = None) -> Dict[str, Any]:
        """
        导出最终训练完成的 JSON 配置文件。
        
        使用带时间戳的命名格式，便于识别训练完成时间。
        
        参数:
            trainer: TraversalTrainer 实例
            experiment_name: 实验名称（可选，默认使用配置中的名称）
            
        返回:
            导出结果字典
        """
        if not self.learned_order:
            raise RuntimeError("导出前必须先执行推断或 process_learned_order")
        
        # 生成带时间戳的文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = experiment_name or self.config.experiment.name
        base_prefix = f"{exp_name}_final_{timestamp}"
        
        environment = trainer.environment
        environment.visited_order = self.learned_order
        
        # 生成配置文件
        data, json_path = self.trajectory_processor.generate_config_file(
            environment, trainer.quadtree,
            filename=f"{base_prefix}.json"
        )
        
        # 同时保存简化版本（只有遍历顺序）
        order_only_path = self.output_dir / f"{base_prefix}_order_only.json"
        order_data = {
            "experiment_name": exp_name,
            "timestamp": timestamp,
            "order_length": len(self.learned_order),
            "learned_order": [cell.code for cell in self.learned_order if hasattr(cell, 'code')],
        }
        with open(order_only_path, 'w', encoding='utf-8') as f:
            json.dump(order_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"最终配置已导出: {json_path}")
        self.logger.info(f"遍历顺序已导出: {order_only_path}")
        
        print(self.get_summary_report())
        
        return {
            "json_path": json_path,
            "order_only_path": order_only_path,
            "base_prefix": base_prefix,
            "timestamp": timestamp
        }
    
    def get_summary_report(self) -> str:
        """生成格式化评估汇总。"""
        if "evaluation" not in self.processing_metadata:
            return "Pending evaluation report..."

        eval_m = self.processing_metadata["evaluation"]
        qs = self.processing_metadata.get("quadtree_stats", {})
        
        # 只支持新的评估数据格式
        val_imp = eval_m.get('val_improvement', 0)
        test_imp = eval_m.get('test_improvement', 0)
        train_imp = eval_m.get('train_improvement', 0)
        hgs = eval_m.get('hgs_score', (val_imp + test_imp) / 2 if val_imp and test_imp else 0)

        report = [
            "\n" + "=" * 65,
            f" Unified RL Post-Evaluation Report ".center(65, " "),
            "-" * 65,
            f" Best Model:      {self.best_model_path.name if self.best_model_path else 'N/A'}",
            f" Order Length:    {self.processing_metadata.get('learned_order_length', 'N/A')} "
            f"(Active: {qs.get('active_cells', 'N/A')})",
            "-" * 65,
            f" Train Improvement: {train_imp:>7.2f}%" if train_imp else " Train Improvement:    N/A",
            f" Val Improvement:   {val_imp:>7.2f}%" if val_imp else " Val Improvement:      N/A",
            f" Test Improvement:  {test_imp:>7.2f}%" if test_imp else " Test Improvement:     N/A",
            "-" * 65,
            f" >>> HGS Score:    {hgs:>10.4f} <<<",
            "-" * 65,
        ]
        
        # 添加详细指标（如果有）
        val_metrics = eval_m.get('val_metrics')
        if val_metrics:
            report.extend([
                f" Baseline Cost:    {val_metrics.get('baseline_avg_cost', 'N/A'):>10.2f}",
                f" Learned Cost:     {val_metrics.get('learned_avg_cost', 'N/A'):>10.2f}",
            ])
        
        report.extend([
            f" Output Dir:       {self.output_dir}",
            "=" * 65 + "\n"
        ])
        
        return "\n".join(report)

    def get_config_summary(self) -> Dict[str, Any]:
        """获取配置摘要"""
        return {
            "alpha": self.config.index.alpha,
            "beta": self.config.index.beta,
            "max_level": self.config.index.max_level
        }

    def get_evaluator(self, trainer: TraversalTrainer):
        """获取评估器实例"""
        from src.evaluation.traversal_evaluator import TraversalPerformanceEvaluator
        
        # 尝试加载保存的查询集
        reference_queries = self._load_saved_queries('reference_queries.pkl')
        if reference_queries is None:
            self.logger.warning("未找到保存的参考查询集，使用环境中的查询集")
            reference_queries = trainer.environment.reference_queries
        else:
            self.logger.info(f"已加载保存的参考查询集: {len(reference_queries)} 个查询")
        
        return TraversalPerformanceEvaluator(
            trainer.quadtree, trainer.encoder, trainer.cost_evaluator,
            reference_queries=reference_queries,
            baseline_include_muted=self.config.index.baseline_include_muted
        )

    def generate_test_queries(self, quadtree: QuadTreeIndex):
        # 尝试加载保存的测试查询集
        test_queries = self._load_saved_queries('test_queries.pkl')
        if test_queries is not None:
            self.logger.info(f"已加载保存的测试查询集: {len(test_queries)} 个查询")
            return test_queries
        
        # 如果没有保存的查询集，直接使用保存的测试集
        self.logger.warning("未找到保存的测试查询集，尝试加载 test_queries.pkl")
        test_queries = self._load_saved_queries('test_queries.pkl')
        if test_queries is None:
            raise RuntimeError("无法加载测试查询集")
        return test_queries
    
    def _load_saved_queries(self, filename: str):
        """加载保存的查询集"""
        import pickle
        from src.utils.path_manager import get_path_manager
        
        pm = get_path_manager()
        # 使用实验名称获取正确的查询集目录
        queries_dir = pm.get_queries_dir(self.config.experiment.name)
        query_path = queries_dir / filename
        
        if query_path.exists():
            self.logger.info(f"从 {query_path} 加载查询集")
            with open(query_path, 'rb') as f:
                return pickle.load(f)
        
        self.logger.warning(f"查询集文件不存在: {query_path}")
        return None

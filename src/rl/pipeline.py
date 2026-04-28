import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import NetworkConfig, TShapeConfig
from src.evaluation.lsfc_evaluator import LSFCEvaluator
from src.indexing.quadtree_cell import QuadTreeCell
from src.training import TraversalTrainer


class LSFCPipeLine:
    """Coordinate training, model loading, evaluation, and export."""

    def __init__(
        self,
        config: TShapeConfig,
        network_config: NetworkConfig,
        logger: Optional[logging.Logger] = None,
        custom_paths: Optional[Dict[str, str]] = None,
    ):
        self.config = config
        self.network_config = network_config
        self.logger = logger or logging.getLogger("RLPipeline")
        self.resource_paths = {
            "checkpoints": self.config.experiment.get_checkpoints_dir(self.config.paths),
            "results": self.config.experiment.get_results_dir(self.config.paths),
            "logs": self.config.experiment.get_logs_dir(self.config.paths),
        }
        if custom_paths:
            for key, val in custom_paths.items():
                if key.endswith("_dir"):
                    name = key[:-4] + "s"
                    path = Path(val)
                    path.mkdir(parents=True, exist_ok=True)
                    self.resource_paths[name] = path

        self.trainer: Optional[TraversalTrainer] = None
        self.postprocessor: Optional[LSFCEvaluator] = None

    def initialize_components(self) -> None:
        self.logger.info("Initializing pipeline components...")
        self.trainer = TraversalTrainer(self.config, self.network_config)
        self.trainer.setup()
        self.trainer.prepare_agent()
        self.postprocessor = LSFCEvaluator(
            config=self.config,
            output_dir=str(self.resource_paths["results"]),
        )
        self.logger.info("Pipeline ready with %s active cells.", self.trainer.environment.num_cells)

    def train_model(self) -> str:
        if not self.trainer:
            self.initialize_components()

        self.logger.info("Starting RL training...")
        self.trainer.train()

        latest_model_path = self.config.experiment.get_checkpoints_dir(self.config.paths) / "latest.pth"
        if latest_model_path.exists():
            model_path = latest_model_path
        else:
            best_record_path = self.config.experiment.get_results_dir(self.config.paths) / "best_model_record.json"
            if not best_record_path.exists():
                raise FileNotFoundError(
                    "Training finished but neither checkpoints/latest.pth nor results/best_model_record.json was produced."
                )
            with open(best_record_path, "r", encoding="utf-8") as f:
                best_record = json.load(f)
            model_path = Path(best_record["best_model_path"])

        if self.postprocessor:
            self.postprocessor.best_model_path = model_path

        return str(model_path)

    def load_trained_model(self, checkpoint_path: str) -> None:
        if not self.trainer:
            self.initialize_components()
        self.logger.info("Loading model checkpoint: %s", checkpoint_path)
        self.trainer.prepare_agent(model_path=checkpoint_path)

    def generate_quadorder(self) -> List[QuadTreeCell]:
        if not self.trainer or not self.trainer.agent:
            raise RuntimeError("Trainer or agent is not initialized.")
        return self.trainer.rollout_policy_order(self.trainer.agent, self.trainer.environment)

    def run_post_evaluation(self, model_path: str, export_prefix: str):
        if not self.postprocessor:
            self.initialize_components()

        self.load_trained_model(model_path)
        order = self.generate_quadorder()
        self.postprocessor.process_quadorder(order, self.trainer.quadtree)

        test_queries = self.trainer.environment.test_queries
        if test_queries:
            evaluator = self.postprocessor.get_evaluator(self.trainer)
            self.postprocessor.processing_metadata["evaluation"] = evaluator.compute_hgs_score(order, test_queries)

        self.postprocessor.best_model_path = Path(model_path)
        return self.postprocessor.export_formats(self.trainer, base_prefix=export_prefix)

    def _resolve_model_for_export(self, trained_model_path: str) -> str:
        best_record_path = self.config.experiment.get_results_dir(self.config.paths) / "best_model_record.json"
        if best_record_path.exists():
            with open(best_record_path, "r", encoding="utf-8") as f:
                best_record = json.load(f)
            model_path = best_record["best_model_path"]
            self.logger.info("Using best checkpoint for export: %s", model_path)
            return model_path

        self.logger.info("Falling back to latest trained checkpoint: %s", trained_model_path)
        return trained_model_path

    def run_full_pipeline(self, export_prefix: str = "rl_tshape") -> Dict[str, Any]:
        self.logger.info(">>> Starting end-to-end RL pipeline <<<")

        trained_model_path = self.train_model()
        model_path = self._resolve_model_for_export(trained_model_path)

        self.load_trained_model(model_path)
        quadorder = self.generate_quadorder()

        train_queries = self.trainer.environment.reference_queries
        val_queries = self.trainer.environment.val_queries
        test_queries = self.trainer.environment.test_queries
        if train_queries is None or val_queries is None or test_queries is None:
            raise RuntimeError("Train/val/test query sets must all be available for pipeline evaluation.")

        evaluator_train = self.postprocessor._create_evaluator(self.trainer, train_queries)
        evaluator_val = self.postprocessor._create_evaluator(self.trainer, val_queries)
        evaluator_test = self.postprocessor._create_evaluator(self.trainer, test_queries)
        train_metrics = evaluator_train.evaluate_final_order(quadorder)
        val_metrics = evaluator_val.evaluate_final_order(quadorder)
        test_metrics = evaluator_test.evaluate_final_order(quadorder)
        hgs_score = (val_metrics["improvement_percent"] + test_metrics["improvement_percent"]) / 2

        self.postprocessor.process_quadorder(quadorder, self.trainer.quadtree)
        self.postprocessor.best_model_path = Path(model_path)
        self.postprocessor.processing_metadata["evaluation"] = {
            "i_val": val_metrics["improvement_percent"],
            "i_test": test_metrics["improvement_percent"],
            "hgs_score": hgs_score,
            "train_improvement": train_metrics["improvement_percent"],
            "val_improvement": val_metrics["improvement_percent"],
            "test_improvement": test_metrics["improvement_percent"],
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        }

        export_results = self.postprocessor.export_formats(self.trainer, base_prefix=export_prefix)
        return {
            "model_path": model_path,
            "quadorder_length": len(quadorder),
            "improvement_rate": test_metrics["improvement_percent"],
            "quadtree_stats": self.postprocessor.processing_metadata["quadtree_stats"],
            "export_results": export_results,
            "summary_report": self.postprocessor.get_summary_report(),
        }

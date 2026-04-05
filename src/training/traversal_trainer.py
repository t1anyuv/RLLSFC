"""
训练循环协调器：负责数据加载、环境创建和强化学习更新。
"""
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import NetworkConfig, TShapeConfig
from src.core.bounding_box import SpatialBoundingBox
from src.evaluation import TraversalPerformanceEvaluator
from src.indexing import QuadTreeIndex, QuadTreeCell, TraversalOrderEncoder
from src.reward import TraversalCostEvaluator
from src.rl import TraversalEnvironment, TraversalPolicyAgent
from src.training.component_factory import TrainingComponentFactory
from src.training.training_state import TrainingState
from src.utils.path_manager import get_path_manager
from src.utils.similarity_matrix import SimilarityMatrix

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
warnings.filterwarnings("ignore", category=UserWarning)


class TraversalTrainer:
    """训练循环协调器。

    负责协调整个训练流程，包括环境初始化、智能体训练、评估和检查点保存。
    """

    def __init__(self, config: TShapeConfig, network_config: Optional[NetworkConfig] = None):
        self.config = config
        self.network_config = network_config or self.config.network

        # 应用路径配置并设置实验名称
        self.config.paths.apply_to_path_manager()
        pm = get_path_manager()
        pm.set_experiment_name(self.config.experiment.name)
        
        # 确保实验输出目录存在
        self.config.experiment.get_models_dir().mkdir(parents=True, exist_ok=True)
        self.config.experiment.get_orders_dir().mkdir(parents=True, exist_ok=True)
        self.config.experiment.get_logs_dir().mkdir(parents=True, exist_ok=True)
        self.config.experiment.get_curves_dir().mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        from datetime import datetime
        from src.utils.logger import setup_logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.config.experiment.get_logs_dir() / f"training_{timestamp}.log"
        self.logger = setup_logging(f"Trainer_{self.config.experiment.name}", log_file=log_file)
        self.logger.info(f"=== 初始化实验: {self.config.experiment.name} ===")

        # 核心组件
        self.quadtree: Optional[QuadTreeIndex] = None
        self.environment: Optional[TraversalEnvironment] = None
        self.agent: Optional[TraversalPolicyAgent] = None
        self.encoder: Optional[TraversalOrderEncoder] = None
        self.cost_evaluator: Optional[TraversalCostEvaluator] = None
        self.similarity_matrix: Optional[SimilarityMatrix] = None

        # 训练状态
        self.state = TrainingState()

        # 组件工厂
        self.factory = TrainingComponentFactory(config)

    def setup(self) -> Tuple[TraversalEnvironment, QuadTreeIndex]:
        """构建四叉树、分配轨迹、配置环境组件及相似度矩阵。
        
        返回:
            (environment, quadtree) 元组
        """
        print(f"\n初始化训练环境 (四叉树最大层级: {self.config.index.max_level}) ---")


        # 1. 初始化四叉树
        self.quadtree = self.factory.create_quadtree()

        # 2. 加载并分配轨迹数据
        trajectories = self.factory.load_trajectories(self.quadtree)
        print("正在分配轨迹数据到单元格...")
        for trajectory_id, points in trajectories:
            self.quadtree.assign_trajectory(trajectory_id, points)

        # 3. 剪枝处理
        if self.config.index.min_cell_trajs is not None:
            self.quadtree.post_prune_tree(self.config.index.min_cell_trajs)

        # 4. 优化签名
        self.quadtree.compute_signatures(self.config.index.enable_sig_optimize)

        # 5. 配置核心组件
        self.encoder = TraversalOrderEncoder(
            self.quadtree,
            alpha=self.config.index.alpha,
            beta=self.config.index.beta
        )
        self.cost_evaluator = TraversalCostEvaluator(self.quadtree)

        # 6. 构建环境实例
        self.environment = self.factory.create_environment(self.quadtree, self.cost_evaluator)

        # 7. 加载或计算相似度矩阵
        self.similarity_matrix = self.factory.setup_similarity_matrix(
            self.quadtree,
            self.cost_evaluator,
            self.environment.all_cells
        )

        return self.environment, self.quadtree

    def prepare_agent(self, model_path: Optional[str] = None) -> TraversalPolicyAgent:
        """创建Actor-Critic模型，可选加载预训练权重"""
        if self.environment is None:
            raise RuntimeError("请先调用 setup() 初始化环境。")

        self.agent = self.create_agent(
            self.environment.state_dimension,
            self.environment.action_dimension
        )

        if model_path:
            self.agent.load(model_path)
            self.agent.actor.eval()
            print(f"成功加载模型权重: {model_path}")

        return self.agent

    def train(self) -> Dict[str, Any]:
        """主训练流程：执行完整的强化学习训练循环。"""
        if self.environment is None or self.quadtree is None:
            self.setup()
        if self.agent is None:
            self.prepare_agent()

        self.logger.info(f"开始训练 - Episodes: {self.config.train.num_episodes}")
        self.logger.info(f"评估间隔: {self.config.train.eval_interval} | "
                         f"早停耐心值: {self.config.train.early_stopping_patience} | "
                         f"早停最小改进值: {self.config.train.early_stopping_min_delta * 100}%")

        for episode in tqdm(range(self.config.train.num_episodes)):
            total_reward, steps = self._run_training_episode()

            update_info = self.agent.update()
            if update_info and 'loss' in update_info:
                self.state.record_episode(total_reward, steps, update_info['loss'])
            else:
                self.state.record_episode(total_reward, steps)

            # 每10个episode记录一次
            if (episode + 1) % 10 == 0:
                self.logger.info(f"Episode {episode + 1}: Reward={total_reward:.2f}, Steps={steps}")

            # 定期评估
            if self._handle_periodic_evaluation(episode + 1):
                self.state.early_stop_episode = episode + 1
                self.logger.info(f"触发早停：已连续 {self.config.train.early_stopping_patience} 次评估未见实质改进")
                break

        # 绘制曲线与最终评估
        return self._finalize_training()

    def _run_training_episode(self) -> Tuple[float, int]:
        """运行单个训练episode，返回总奖励和步数。"""
        state, action_mask = self.environment.reset()
        total_reward = 0.0
        steps = 0

        while True:
            # Top-K 动作过滤
            refined_mask = self._refine_action_mask(self.environment, action_mask)

            # 选择动作
            action, log_prob, value = self.agent.select_action(state, refined_mask)

            # 环境步进
            next_state, next_mask, reward, done, _ = self.environment.step(action)

            # 存储经验并更新
            self.agent.store_transition(state, action, reward, log_prob, value, refined_mask, done)

            total_reward += reward
            steps += 1
            state, action_mask = next_state, next_mask

            if done:
                return total_reward, steps

    def rollout_policy_order(
            self,
            agent: TraversalPolicyAgent,
            environment: TraversalEnvironment,
            max_rollout_multiplier: int = 4,
    ) -> List[QuadTreeCell]:
        """使用当前策略执行确定性 Rollout，获取完整节点遍历顺序。"""
        state, action_mask = environment.reset()
        max_steps = max_rollout_multiplier * max(1, environment.num_cells)
        steps = 0

        while len(environment.visited_cells) < environment.num_cells and steps < max_steps:
            refined_mask = self._refine_action_mask(environment, action_mask)
            action, _, _ = agent.select_action(state, refined_mask, deterministic=True)
            next_state, next_mask, _, done, _ = environment.step(action)
            state, action_mask = next_state, next_mask
            steps += 1
            if done:
                break

        if len(environment.visited_cells) < environment.num_cells:
            raise RuntimeError(f"Rollout 失败：步数超限 ({steps}) 仍未覆盖全部节点。")

        return environment.quadorder()

    def _create_quadtree(self) -> QuadTreeIndex:
        """根据配置初始化边界框并构建四叉树索引"""
        if self.config.index.use_original_bbox:
            bbox = self.config.get_original_bbox()
        else:
            bbox = SpatialBoundingBox(0.0, 0.0, 1.0, 1.0)
        return QuadTreeIndex(
            bbox,
            max_level=self.config.index.max_level,
            alpha=self.config.index.alpha,
            beta=self.config.index.beta
        )

    def _setup_similarity_matrix(self):
        """配置、加载或计算轨迹相似度矩阵"""
        if not self.config.data.use_similarity_matrix:
            return

        self.similarity_matrix = SimilarityMatrix(self.quadtree, self.cost_evaluator)
        matrix_path = self.config.get_effective_similarity_matrix_path()

        if matrix_path.exists():
            print(f"加载相似度矩阵: {matrix_path}")
            if not self.similarity_matrix.load(str(matrix_path), self.environment.all_cells):
                fallback_path = self.config.get_experiment_similarity_matrix_path()
                print(
                    "矩阵维度不匹配，将切换到实验私有矩阵: "
                    f"{fallback_path}"
                )
                if fallback_path.exists() and self.similarity_matrix.load(str(fallback_path), self.environment.all_cells):
                    return
                self._compute_and_save_similarity_matrix(str(fallback_path), self.environment.all_cells)
        else:
            print(f"⚠ 相似度矩阵文件不存在: {matrix_path}")
            self._compute_and_save_similarity_matrix(str(matrix_path), self.environment.all_cells)

    def _compute_and_save_similarity_matrix(self, matrix_path: str, all_cells: List[QuadTreeCell]) -> None:
        """计算并持久化相似度矩阵。"""
        print(f"开始计算相似度矩阵 (节点数: {len(all_cells)})...")
        num_workers = self.config.data.similarity_num_workers
        self.similarity_matrix.compute(all_cells, use_symmetric=True, show_progress=True, num_workers=num_workers)
        self.similarity_matrix.save(matrix_path)

    def create_agent(self, state_dim: int, action_dim: int) -> TraversalPolicyAgent:
        """根据环境维度和网络配置初始化 PPO 智能体。"""
        device = self.network_config.get_torch_device()
        print(f"使用设备: {device}")
        
        initial_entropy_coef = self.config.train.entropy_coef_start
        
        return TraversalPolicyAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            lr_actor=self.config.train.lr_actor,
            lr_critic=self.config.train.lr_critic,
            eps_clip=self.config.train.eps_clip,
            k_epochs=self.config.train.k_epochs,
            gamma=self.config.train.gamma,
            entropy_coef=initial_entropy_coef,
            gae_lambda=self.config.train.gae_lambda,
            gradient_clip_norm=self.config.train.gradient_clip_norm,
            device=self.network_config.get_torch_device(),
            hidden_dims=self.network_config.hidden_dims,
        )

    def _refine_action_mask(self, environment: TraversalEnvironment, action_mask: np.ndarray) -> np.ndarray:
        """基于相似度过滤动作掩码，缩小智能体搜索空间。"""
        refined_mask = action_mask.copy()
        current_limit = self._current_action_limit(environment)

        if current_limit and current_limit < action_mask.sum():
            top_indices = self._select_top_similar_cells(environment, refined_mask, current_limit)
            new_mask = np.zeros_like(refined_mask)
            new_mask[top_indices] = 1
            refined_mask = new_mask

        environment.last_action_filter_info = {
            "top_k": bool(current_limit),
            "available_count": int(refined_mask.sum()),
        }
        return refined_mask

    def _current_action_limit(self, environment: TraversalEnvironment) -> Optional[int]:
        """根据训练步数进度，计算当前 Top-K 动作限制。"""
        base_limit = self.config.train.topk_actions
        if not base_limit or base_limit <= 0:
            return None

        start_multiplier, final_multiplier = self.config.train.topk_multipliers
        start_limit = max(1, int(round(base_limit * start_multiplier)))
        final_limit = max(1, int(round(base_limit * final_multiplier)))
        decay = max(1, self.config.train.topk_decay_episodes)

        progress = min(1.0, environment.current_episode / decay)
        current = start_limit + (final_limit - start_limit) * progress
        return max(1, int(round(current)))

    def _select_top_similar_cells(self, environment: TraversalEnvironment,
                                  action_mask: np.ndarray,
                                  limit: int) -> List[int]:
        """计算当前节点与所有候选节点的相似度，返回 Top-K 索引。"""
        available_indices = np.where(action_mask > 0.5)[0]
        if available_indices.size <= limit:
            return available_indices.tolist()

        current_node = environment.current_cell
        if current_node is None:
            return available_indices[:limit].tolist()

        get_sim = (self.similarity_matrix.get_similarity if self.similarity_matrix
                   else lambda n1, n2: environment.cost_evaluator.jaccard_similarity(n1, n2))

        similarities = []
        for idx in available_indices:
            candidate = environment.all_cells[idx]
            similarities.append((idx, get_sim(current_node, candidate)))

        similarities.sort(key=lambda x: x[1], reverse=True)
        top_k = [idx for idx, _ in similarities[:limit]]
        return top_k if top_k else available_indices[:limit].tolist()

    def _build_evaluator(
            self,
            reference_queries: List[SpatialBoundingBox],
    ) -> TraversalPerformanceEvaluator:
        """基于当前训练上下文创建统一评估器。"""
        return TraversalPerformanceEvaluator(
            self.quadtree,
            self.encoder,
            self.cost_evaluator,
            reference_queries=reference_queries,
            quadcode_include_muted=self.config.index.quadcode_include_muted,
        )

    def _evaluate_query_sets(
            self,
            quadorder: List[QuadTreeCell],
            *,
            train_queries: Optional[List[SpatialBoundingBox]] = None,
            val_queries: Optional[List[SpatialBoundingBox]] = None,
            test_queries: Optional[List[SpatialBoundingBox]] = None,
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        """在可用的查询集上统一执行评估。"""
        metrics: Dict[str, Optional[Dict[str, Any]]] = {
            "train": None,
            "val": None,
            "test": None,
        }
        datasets = {
            "train": train_queries,
            "val": val_queries,
            "test": test_queries,
        }

        for split_name, queries in datasets.items():
            if not queries:
                continue
            metrics[split_name] = self._build_evaluator(queries).evaluate_final_order(quadorder)

        return metrics

    def _log_split_metrics(self, split_name: str, metrics: Dict[str, Any], suffix: str = "") -> None:
        """统一输出单个数据集的评估结果。"""
        split_label = split_name.capitalize()
        improvement = metrics["improvement_percent"]
        quadcode_cost = metrics["quadcode_avg_cost"]
        quadorder_cost = metrics["quadorder_avg_cost"]

        if suffix:
            self.logger.info(f"{split_label}集改进率: {improvement:.2f}% {suffix}")
        else:
            self.logger.info(f"{split_label}集改进率: {improvement:.2f}%")
        self.logger.info(f"{split_label}集 Baseline 平均成本: {quadcode_cost:.2f}")
        self.logger.info(f"{split_label}集 学习顺序平均成本: {quadorder_cost:.2f}")

    def _training_stats_snapshot(self) -> Dict[str, Any]:
        """构建当前训练统计快照。"""
        return {
            "total_episodes": len(self.state.episode_rewards),
            "avg_reward": float(np.mean(self.state.episode_rewards)) if self.state.episode_rewards else 0,
            "final_reward": float(self.state.episode_rewards[-1]) if self.state.episode_rewards else 0,
            "best_improvement": float(max(self.state.val_improvement_history)) if self.state.val_improvement_history else 0,
            "early_stop_episode": self.state.early_stop_episode,
        }

    def _training_config_snapshot(self) -> Dict[str, Any]:
        """构建用于指标落盘的训练配置快照。"""
        return {
            "max_level": self.config.index.max_level,
            "alpha": self.config.index.alpha,
            "beta": self.config.index.beta,
            "num_trajectories": self.config.data.num_trajectories,
            "num_episodes": len(self.state.episode_rewards),
            "lr_actor": self.config.train.lr_actor,
            "lr_critic": self.config.train.lr_critic,
        }

    def _write_json(self, path: Path, payload: Dict[str, Any], log_message: Optional[str] = None) -> None:
        """统一写入 JSON 文件。"""
        import json

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        if log_message:
            self.logger.info(log_message.format(path=path))

    def _handle_periodic_evaluation(self, episode: int) -> bool:
        """执行周期性评估逻辑。"""
        should_stop = False
        if episode % self.config.train.eval_interval == 0 or episode == self.config.train.num_episodes:
            self._evaluate_and_checkpoint(episode)
            if self.config.train.enable_early_stopping:
                should_stop = self._should_stop_early()

        elif episode % 10 == 0:
            avg_reward = self.state.get_recent_avg_reward(10)
            self.logger.info(f"Episode {episode}: 近10轮平均奖励 = {avg_reward:.2f}")

        return should_stop

    def _evaluate_and_checkpoint(self, episode: int) -> None:
        """评估当前策略在三类数据集上的表现，并保存模型检查点。"""
        avg_reward = self.state.get_recent_avg_reward(self.config.train.eval_interval)
        self.logger.info(f"\n[评估] Episode {episode}: 平均奖励 = {avg_reward:.2f}")

        environment = self.environment
        agent = self.agent
        train_queries = environment.reference_queries
        val_queries = environment.val_queries
        test_queries = environment.test_queries
        if val_queries is None:
            self.logger.error("验证集不可用，跳过评估")
            return

        quadorder = self.rollout_policy_order(agent, environment)
        metrics = self._evaluate_query_sets(
            quadorder,
            train_queries=train_queries,
            val_queries=val_queries,
            test_queries=test_queries,
        )
        train_metrics = metrics["train"]
        val_metrics = metrics["val"]
        test_metrics = metrics["test"]

        if train_metrics:
            self._log_split_metrics("train", train_metrics)
        self._log_split_metrics("val", val_metrics, suffix="(vs Z-Order)")
        if test_metrics:
            self._log_split_metrics("test", test_metrics)

        train_imp = train_metrics["improvement_percent"] if train_metrics else 0.0
        val_imp = val_metrics['improvement_percent']
        test_imp = test_metrics['improvement_percent'] if test_metrics else 0.0

        self.state.record_evaluation(
            episode,
            train_improvement=train_imp,
            val_improvement=val_imp,
            test_improvement=test_imp
        )

        # 保存评估历史
        self._save_evaluation_history()

        # 计算 HGS 得分
        hgs_score = (val_imp + test_imp) / 2  # HGS = (I_val + I_test) / 2
        self.logger.info(f"HGS Score: {hgs_score:.4f}")

        # 保存检查点
        if episode % self.config.train.save_interval == 0:
            pm = get_path_manager()
            path = pm.get_model_dir() / f"ep{episode:06d}_val{val_imp:+.2f}.pth"
            agent.save(str(path))
            self.logger.info(f"模型已保存: {path}")
            
            metrics_path = pm.get_model_dir() / f"ep{episode:06d}_metrics.json"
            self._save_checkpoint_metrics(metrics_path, episode, train_metrics, val_metrics, test_metrics)

    def _save_evaluation_history(self) -> None:
        """保存评估历史到 JSON 文件。"""
        from datetime import datetime
        
        history = {
            "timestamp": datetime.now().isoformat(),
            "episodes": self.state.improvement_episodes,
            "train_improvements": self.state.train_improvement_history,
            "val_improvements": self.state.val_improvement_history,
            "test_improvements": self.state.test_improvement_history,
        }
        
        history_path = self.config.experiment.get_logs_dir() / "evaluation_history.json"
        self._write_json(history_path, history)

    def _save_checkpoint_metrics(self, path: Path, episode: int, 
                                  train_metrics: Optional[Dict], 
                                  val_metrics: Dict, 
                                  test_metrics: Optional[Dict]) -> None:
        """保存检查点评估指标。"""
        metrics = {
            "episode": episode,
            "timestamp": datetime.now().isoformat(),
            "train": train_metrics if train_metrics else None,
            "val": val_metrics,
            "test": test_metrics if test_metrics else None,
        }
        self._write_json(path, metrics, "评估指标已保存: {path}")

    def _finalize_training(self) -> Dict[str, Any]:
        """训练结束后的最终评估、绘图及模型保存"""
        self.logger.info("=== 训练结束，执行最终评估 ===")
        self._plot_training_curves()

        final_metrics = self._final_evaluation()
        
        # 生成带时间戳和指标的最终模型名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_imp = final_metrics.get('test_metrics', {}).get('improvement_percent', 0)
        val_imp = final_metrics.get('val_metrics', {}).get('improvement_percent', 0)
        hgs = (val_imp + test_imp) / 2
        
        # 最终模型保存 - 使用命名格式: final_{timestamp}_hgs{HGS}.pth
        final_model_name = f"final_{timestamp}_hgs{hgs:+.2f}_val{val_imp:+.2f}_test{test_imp:+.2f}.pth"
        final_model_path = self.config.experiment.get_models_dir() / final_model_name
        self.agent.save(str(final_model_path))
        self.logger.info(f"最终模型已保存: {final_model_path}")
        
        latest_path = self.config.experiment.get_models_dir() / "latest.pth"
        self.agent.save(str(latest_path))
        
        final_metrics_path = self.config.experiment.get_models_dir() / f"final_{timestamp}_metrics.json"
        self._save_final_metrics(final_metrics_path, final_metrics, final_model_name)
        
        # 保存训练总结
        self._save_training_summary(final_metrics)
        
        self._auto_select_best_model()
        
        return final_metrics

    def _auto_select_best_model(self) -> None:
        """从保存的 metrics 文件中自动选择最佳模型。"""
        from src.evaluation.lsfc_evaluator import LSFCEvaluator
        
        self.logger.info("=== 自动选择最佳模型 ===")
        
        model_dir = self.config.experiment.get_models_dir()
        output_dir = self.config.experiment.get_orders_dir()
        
        # 创建 evaluator
        evaluator = LSFCEvaluator(self.config, output_dir=str(output_dir))
        
        best_model_path, best_record = evaluator.select_best_model_from_metrics(str(model_dir))
        
        if best_model_path is None:
            self.logger.warning("未找到最佳模型")
            return
        
        self.logger.info(f"最佳模型: {best_model_path.name}")
        self.logger.info(f"  Episode: {best_record['episode']}")
        self.logger.info(f"  HGS Score: {best_record['hgs_score']:.4f}")
        self.logger.info(f"  Val Improvement: {best_record['val_improvement']:+.2f}%")
        self.logger.info(f"  Test Improvement: {best_record['test_improvement']:+.2f}%")
        
        # 保存最佳模型记录
        evaluator._save_best_model_record(best_record)
        self.logger.info(f"最佳模型记录已保存到 orders/best_model_record.json")

    def _save_final_metrics(self, path: Path, final_metrics: Dict[str, Any], model_name: str) -> None:
        """保存最终评估指标。"""
        summary = {
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
            "experiment_name": self.config.experiment.name,
            "final_metrics": final_metrics,
            "training_config": self._training_config_snapshot(),
            "training_stats": self._training_stats_snapshot(),
        }
        self._write_json(path, summary, "最终指标已保存: {path}")

    def _final_evaluation(self) -> Dict[str, Any]:
        """使用保存的查询集进行最终评估，同时评估 val 和 test"""
        environment = self.environment
        val_queries = environment.val_queries
        test_queries = environment.test_queries
        
        if val_queries is None or test_queries is None:
            raise RuntimeError("验证集或测试集不可用")

        quadorder = self.rollout_policy_order(self.agent, environment)
        metrics = self._evaluate_query_sets(
            quadorder,
            val_queries=val_queries,
            test_queries=test_queries,
        )
        val_metrics = metrics["val"]
        test_metrics = metrics["test"]

        self.logger.info("=== Val 查询集评估结果 ===")
        self._log_split_metrics("val", val_metrics)
        
        self.logger.info("=== Test 查询集评估结果 ===")
        self._log_split_metrics("test", test_metrics)

        return {
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'improvement_percent': test_metrics['improvement_percent']
        }
    
    def _load_saved_queries(self, filename: str):
        """加载保存的查询集"""
        import pickle
        pm = get_path_manager()
        queries_dir = pm.get_queries_dir(self.config.experiment.name)
        query_path = queries_dir / filename
        
        if query_path.exists():
            with open(query_path, 'rb') as f:
                return pickle.load(f)
        
        self.logger.error(f"查询集文件不存在: {query_path}")
        return None
    
    def _save_training_summary(self, final_metrics: Dict[str, Any]):
        """保存训练总结到日志文件"""
        from datetime import datetime
        
        test_metrics = final_metrics.get('test_metrics', final_metrics)
        
        summary = {
            "experiment_name": self.config.experiment.name,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "max_level": self.config.index.max_level,
                "alpha": self.config.index.alpha,
                "beta": self.config.index.beta,
                "num_trajectories": self.config.data.num_trajectories,
                "num_episodes": self.config.train.num_episodes,
                "lr_actor": self.config.train.lr_actor,
                "lr_critic": self.config.train.lr_critic,
            },
            "training_stats": self._training_stats_snapshot(),
            "final_metrics": {
                "quadcode_avg_cost": float(test_metrics['quadcode_avg_cost']),
                "quadorder_avg_cost": float(test_metrics['quadorder_avg_cost']),
                "improvement_percent": float(test_metrics['improvement_percent']),
                "quadcode_nodes_hit": float(test_metrics['quadcode_nodes_hit']),
                "quadorder_nodes_hit": float(test_metrics.get('quadorder_nodes_hit', 0)),
            }
        }
        
        summary_path = self.config.experiment.get_logs_dir() / "training_summary.json"
        self._write_json(summary_path, summary, "训练总结已保存: {path}")

    def _should_stop_early(self) -> bool:
        """
        基于耐心值（Patience）机制与趋势稳定性判断是否触发早停。
        逻辑：如果当前改进率未能超越历史最高值，则消耗耐心。
        返回:
            是否应该早停
        """
        if not self.config.train.enable_early_stopping:
            return False

        if not self.state.val_improvement_history:
            return False

        current_improvement = self.state.val_improvement_history[-1]

        self.state.update_negative_streak(current_improvement)

        if self.state.negative_streak_counter >= self.config.train.max_negative_streak:
            print(f"[EarlyStop] Circuit Breaker: Continuous negative improvement "
                  f"for {self.config.train.max_negative_streak} times. Stopping.")
            return True

        is_significant_improvement = (
                current_improvement > (self.state.best_improvement + self.config.train.early_stopping_min_delta))

        if is_significant_improvement:
            self.state.update_best_improvement(current_improvement)
            print(f"[EarlyStop] New best: {current_improvement:.2f}%. Resetting patience.")
            return False
        else:
            # 未能刷新纪录，消耗耐心
            self.state.patience_counter += 1

        # 计算稳定性指标 (仅用于日志打印)
        stability_info = ""
        if len(self.state.val_improvement_history) >= self.config.train.early_stopping_patience:
            recent = self.state.val_improvement_history[-self.config.train.early_stopping_patience:]
            max_diff = max(recent) - min(recent)
            stability_info = f" (Recent stability: {max_diff:.4f})"

        # 判定是否停止
        if self.state.patience_counter >= self.config.train.early_stopping_patience:
            print(f"[EarlyStop] Triggered! Patience ({self.state.patience_counter}) exhausted.{stability_info}")
            return True

        return False

    def _plot_training_curves(self) -> None:
        """绘制并保存训练过程曲线 (Reward, Loss, Improvement)。"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = self.config.experiment.get_curves_dir() / f"training_progress_{timestamp}.png"

        # (Reward, Loss, Improvement)
        fig, (ax_rev, ax_loss, ax_imp) = plt.subplots(1, 3, figsize=(20, 6))

        has_stopped = self.state.early_stop_episode is not None
        stop_ep = self.state.early_stop_episode

        # --- 1. Reward ---
        rewards = np.array(self.state.episode_rewards)
        ax_rev.plot(rewards, color='steelblue', alpha=0.3)
        if len(rewards) >= 10:
            ma = pd.Series(rewards).rolling(window=10).mean()
            ax_rev.plot(ma, color='firebrick', label='MA (10)')

        if has_stopped:
            ax_rev.axvline(x=stop_ep, color='orange', linestyle='--', linewidth=2, label=f'Early Stop ({stop_ep})')

        ax_rev.set_title("Episode Reward")
        ax_rev.set_xlabel("Episode")
        ax_rev.grid(True, alpha=0.3)
        ax_rev.legend()

        # --- 2. Loss  ---
        if self.state.loss_history:
            losses = np.array(self.state.loss_history)
            ax_loss.plot(losses, color='purple', alpha=0.4)
            if len(losses) >= 10:
                ma_loss = pd.Series(losses).rolling(window=10).mean()
                ax_loss.plot(ma_loss, color='darkviolet', label='MA Loss')

            if has_stopped:
                ax_loss.axvline(x=stop_ep, color='orange', linestyle='--', linewidth=2, label=f'Early Stop ({stop_ep})')

            ax_loss.set_title("Training Loss (Convergence)")
            ax_loss.set_xlabel("Episode")
            ax_loss.set_yscale('log')
            ax_loss.grid(True, which="both", ls="-", alpha=0.2)
            ax_loss.legend()
        else:
            ax_loss.text(0.5, 0.5, "No Loss Data", ha='center')

        # --- 3. Improvement ---
        if self.state.val_improvement_history:
            ax_imp.plot(self.state.improvement_episodes, self.state.val_improvement_history,
                        marker='o', markersize=4, color='forestgreen', label='Val Improvement %')
            ax_imp.axhline(y=0, color='black', linestyle='--', alpha=0.5)

            if has_stopped:
                ax_imp.axvline(x=stop_ep, color='orange', linestyle='--', linewidth=2, label=f'Early Stop ({stop_ep})')

            # 趋势线
            if len(self.state.val_improvement_history) > 1:
                z = np.polyfit(self.state.improvement_episodes, self.state.val_improvement_history, 1)
                p = np.poly1d(z)
                ax_imp.plot(self.state.improvement_episodes, p(self.state.improvement_episodes), "r--", alpha=0.7)
            ax_imp.set_title("Improvement vs QuadCode")
            ax_imp.set_ylabel("%")
            ax_imp.set_xlabel("Episode")
            ax_imp.grid(True, alpha=0.3)

        plt.suptitle(f"TShape RL Training Metrics - {timestamp}", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(save_path, dpi=300)
        plt.close()

    def describe_action_limit_schedule(self) -> Optional[Dict[str, int]]:
        """返回Top-K动作限制的调度参数。

        返回:
            包含调度参数的字典，如果未启用则返回 None
        """
        if not self.config.train.topk_actions:
            return None
        
        start_multiplier, final_multiplier = self.config.train.topk_multipliers
        return {
            "start_multiplier": start_multiplier,
            "final_multiplier": final_multiplier,
            "decay_episodes": self.config.train.topk_decay_episodes,
        }



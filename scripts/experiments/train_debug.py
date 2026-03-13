"""使用少量episode对TraversalEnvironment与策略网络进行调试，展示关键中间结果

运行方式：
    python -m scripts.experiments.train_debug --config resource/experiments/test/config.yaml
"""
import argparse
import json
import random
from datetime import datetime
from typing import Dict, List, Tuple, Union

import numpy as np
import torch

from src.config import TShapeConfig
from src.evaluation import TraversalPerformanceEvaluator
from src.indexing import TraversalOrderEncoder
from src.training import TraversalTrainer


def set_seed(seed: int = 42) -> None:
    """固定随机种子，确保输出可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_header(config: TShapeConfig, trainer: TraversalTrainer) -> None:
    """输出关键配置参数，便于对照调试结果"""
    print("\n" + "=" * 40)
    print("       ==== 调试会话配置 ====       ")
    print("=" * 40)
    print(f"exclude_muted_nodes               : {config.index.use_prune}")
    print(f"baseline_include_muted            : {config.index.baseline_include_muted}")
    print(f"min_cell_trajs                    : {config.index.min_cell_trajs}")
    print(f"enable_sig_optimize               : {config.index.enable_sig_optimize}")
    print(f"top_k_actions                     : {config.train.topk_actions}")
    print(f"local_reward_weight               : {config.reward.local_reward_weight}")
    print(f"global_reward_weight              : {config.reward.global_reward_weight}")
    print(f"global_reward_num_evals           : {config.reward.global_reward_num_evals}")
    print(f"topk_multipliers                  : {config.train.topk_multipliers}")
    print(f"topk_decay_episodes               : {config.train.topk_decay_episodes}")

    schedule = trainer.describe_action_limit_schedule()
    if schedule:
        print(
            "  ↳ 动作限制调度: "
            f"start={schedule['start_multiplier']}, "
            f"final={schedule['final_multiplier']}, "
            f"decay={schedule['decay_episodes']} episodes"
        )
    print(f"device                            : {trainer.network_config.get_torch_device()}")
    print("=" * 40 + "\n")


def print_action_diagnostics(
        step: int,
        environment,
        original_mask: np.ndarray,
        filtered_mask: np.ndarray,
        probs: np.ndarray,
        similarities: np.ndarray,
) -> None:
    """打印动作筛选前后的候选数量及 Top 几项概率/相似度。"""
    original_candidates = int(np.sum(original_mask))
    filtered_candidates = int(np.sum(filtered_mask))
    print(f"\n[Step {step}]")
    print(
        f"  可行动作数: 原始={original_candidates}, 过滤后={filtered_candidates}"
        + (" (top-k 过滤)" if not np.array_equal(original_mask, filtered_mask) else "")
    )

    available_indices = np.where(filtered_mask > 0.5)[0]
    if available_indices.size == 0:
        print("  !!! 无可用动作，环境将使用回退策略。")
        return

    current_cell = environment.current_cell
    if current_cell:
        print(
            f"  当前位置: level={current_cell.level}, seq={current_cell.quadrant_sequence}, "
            f"覆盖轨迹={len(current_cell.trajectories)}"
        )

    # 按概率排序展示 Top 10
    sorted_indices = available_indices[np.argsort(probs[available_indices])][::-1]
    top_k_preview = min(10, sorted_indices.size)

    print(f"  策略前瞻 (Top {top_k_preview}):")
    for idx in sorted_indices[:top_k_preview]:
        node = environment.all_cells[idx]
        sim = similarities[idx]
        print(
            f"    Action {idx:3d} | L{node.level} Seq {node.quadrant_sequence} "
            f"| sim={sim:.4f} | prob={probs[idx]:.4f} | trajs={len(node.trajectories)}"
        )
    if sorted_indices.size > top_k_preview:
        print(f"    ...（共有 {sorted_indices.size} 个可选动作）")


def run_debug_episode(
        trainer: TraversalTrainer,
        environment,
        agent,
        episodes: int = 3,
        verbose_logging: bool = False,
) -> Tuple[List[dict], List[dict], Dict[str, Union[float, str, None]]]:
    """运行若干 episode 并打印关键调试信息。"""
    total_nodes = len(environment.quadtree.all_cells)
    active_nodes = environment.num_cells

    print(f"--- 环境拓扑统计 ---")
    print(f"全体节点总数                 : {total_nodes}")
    print(f"活跃节点数量                 : {active_nodes}")
    print(f"哑节点数量                   : {total_nodes - active_nodes}")
    print("-" * 20)

    logs: List[dict] = []
    summaries: List[dict] = []
    device = agent.device

    for episode in range(episodes):
        state, action_mask = environment.reset()
        done = False
        step = 0
        print(f"\n>>>> 开始调试 Episode {episode + 1}/{episodes}")

        episode_reward = 0.0
        episode_global_trigger_count = 0
        episode_steps = 0

        while not done and step < environment.num_cells:
            # 1. 动作过滤
            filtered_mask = trainer._refine_action_mask(environment, action_mask)

            # 2. 获取概率分布与价值评估
            with torch.no_grad():
                state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                mask_t = torch.as_tensor(filtered_mask, dtype=torch.float32, device=device).unsqueeze(0)

                # 获取策略概率
                logits = agent.actor(state_t)
                # Softmax with mask
                masked_logits = logits + (mask_t - 1.0) * 1e9
                probs = torch.softmax(masked_logits, dim=-1).squeeze(0).cpu().numpy()

            # 3. 计算实时相似度
            available_indices = np.where(filtered_mask > 0.5)[0]
            similarities = np.zeros_like(probs)
            if available_indices.size > 0 and environment.current_cell:
                for idx in available_indices:
                    similarities[idx] = environment.cost_evaluator.jaccard_similarity(
                        environment.current_cell, environment.all_cells[idx]
                    )

            # 4. 打印诊断信息
            print_action_diagnostics(step, environment, action_mask, filtered_mask, probs, similarities)

            # 5. 执行动作
            action, log_prob, value = agent.select_action(state, filtered_mask)
            next_state, next_mask, reward, done, info = environment.step(action)

            # 6. 解析 Reward 信息
            local_reward = info.get("local_reward", 0.0)
            global_reward = info.get("global_reward", 0.0)
            global_triggered = info.get("global_reward_triggered", False)
            current_step = info.get("current_step", step + 1)
            penalty = info.get("penalty")

            # 7：记录被选中动作的概率和节点轨迹数
            chosen_prob = float(probs[action])
            cell_trajs = len(environment.all_cells[action].trajectories)

            global_flag = " [GLOBAL✓]" if global_triggered else ""
            print(
                f"  STEP {step} 执行 Action {action:3d} -> R={reward:.4f} "
                f"[Local:{local_reward:.4f}, Global:{global_reward:.4f}{global_flag}] "
                f"Prob:{chosen_prob:.4f}, V={value.item():.4f}"
            )
            if penalty is not None and penalty < 0:
                print(f"    ⤷ !!! 触发非法动作惩罚: {penalty:.4f}")

            if verbose_logging:
                logs.append({
                    "stage": "step", "episode": episode, "step": step, "action": int(action),
                    "reward": float(reward), "local_reward": float(local_reward),
                    "global_reward": float(global_reward), "global_triggered": bool(global_triggered),
                    "current_step": int(current_step),
                    "value": float(value.item()), "penalty": float(penalty) if penalty is not None else None,
                    "chosen_prob": chosen_prob, "cell_trajs": cell_trajs,
                    "top_k_candidates": int(np.sum(filtered_mask))
                })

            episode_reward += reward
            episode_global_trigger_count += 1 if global_triggered else 0
            episode_steps += 1

            agent.store_transition(state, action, reward, log_prob, value, filtered_mask, done)
            state, action_mask = next_state, next_mask
            step += 1

        # 更新策略
        update_info = agent.update()

        summary = {
            "stage": "episode_summary", "episode": episode, "steps": episode_steps,
            "total_reward": episode_reward, "global_trigger_count": episode_global_trigger_count,
            "loss_stats": update_info,  # 新增：记录 Loss 详情
            "lr_actor": agent.actor_optimizer.param_groups[0]['lr']  # 新增：记录当前学习率
        }
        summaries.append(summary)
        logs.append(summary)
        print(f"\n[Episode {episode + 1} 结束] 总步数: {episode_steps}, 累积奖励: {episode_reward:.4f}")
        if update_info:
            print(f"  ↳ Loss: {update_info.get('loss', 0):.6f} (Actor: {update_info.get('actor_loss', 0):.6f})")

    # 最终评估
    print("\n" + "=" * 40)
    print("        ==== 最终调试评估 ====        ")
    print("=" * 40)
    learned_order = environment.learned_order()
    evaluator = TraversalPerformanceEvaluator(
        environment.quadtree,
        TraversalOrderEncoder(environment.quadtree, environment.alpha, environment.beta),
        environment.cost_evaluator,
        reference_queries=environment.reference_queries,
        baseline_include_muted=environment.baseline_include_muted,
    )
    comparison = evaluator.evaluate_final_order(learned_order)

    print(f"Baseline 平均成本: {comparison['baseline_avg_cost']:.4f}")
    print(f"学习顺序平均成本: {comparison['learned_avg_cost']:.4f}")
    print(f"改进百分比     : {comparison['improvement_percent']:.2f}%")
    print(f"活跃节点总数   : {len(learned_order)}")

    final_entry = {
        "stage": "final_evaluation",
        "improvement_percent": float(comparison['improvement_percent']),
        "learned_avg_cost": float(comparison['learned_avg_cost']),
        "baseline_avg_cost": float(comparison['baseline_avg_cost']),
        "learned_scan_counts":
            float(np.mean(comparison['learned_scan_counts'])) if comparison['learned_scan_counts'] else 0.0,
        "baseline_scan_counts":
            float(np.mean(comparison['baseline_scan_counts'])) if comparison['baseline_scan_counts'] else 0.0,
        "total_active_nodes": active_nodes
    }
    logs.append(final_entry)

    return logs, summaries, final_entry


def main() -> None:
    """主函数：执行调试会话"""
    parser = argparse.ArgumentParser(description="调试RL训练过程")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="YAML配置文件路径（如：resource/experiments/test/config.yaml）"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=15,
        help="调试 episode 数量（默认：15）"
    )
    args = parser.parse_args()

    set_seed(42)
    num_episodes = args.episodes

    # 从 YAML 加载配置
    config = TShapeConfig.from_yaml(args.config)
    # 强制使用调试模式的小规模训练
    config.train.num_episodes = num_episodes

    trainer = TraversalTrainer(config)
    print_header(config, trainer)

    print("--- 系统初始化 ---")
    environment, quadtree = trainer.setup()
    agent = trainer.prepare_agent()

    print(f"状态维度: {environment.state_dimension}, 动作维度: {environment.action_dimension}")
    print(f"初始化完成。开始调试会话...\n")

    logs, summaries, final_entry = run_debug_episode(
        trainer, environment, agent, episodes=num_episodes, verbose_logging=True
    )

    log_dir = config.experiment.get_logs_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"debug_log_{timestamp}.json"

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)

    print(f"\n[Done] 调试日志: {log_path}")

    # 打印简要汇总
    reward_vals = [s["total_reward"] for s in summaries]
    final_loss = summaries[-1]["loss_stats"].get("loss", 0) if summaries[-1].get("loss_stats") else 0

    print(f"平均奖励: {np.mean(reward_vals):.4f}")
    print(f"最终 Loss: {final_loss:.6f}")
    print(f"最终改进: {final_entry['improvement_percent']:.2f}%")
    print("-" * 30)


if __name__ == "__main__":
    main()

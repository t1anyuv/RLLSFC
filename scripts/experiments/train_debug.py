"""
使用少量episode对TraversalEnvironment与策略网络进行调试，展示关键中间结果
运行方式：
    python -m scripts.experiments.train_debug --config configs/experiments/debug/config.yaml
"""
import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
    print("\n" + "=" * 40)
    print("       ==== Debug Session ====       ")
    print("=" * 40)
    print(f"experiment                         : {config.experiment.name}")
    print(f"exclude_muted_nodes               : {config.index.use_prune}")
    print(f"quadcode_include_muted            : {config.index.quadcode_include_muted}")
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
            "  -> action_limit_schedule: "
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
    original_candidates = int(np.sum(original_mask))
    filtered_candidates = int(np.sum(filtered_mask))
    print(f"\n[Step {step}]")
    print(
        f"  available_actions: original={original_candidates}, filtered={filtered_candidates}"
        + (" (top-k filtered)" if not np.array_equal(original_mask, filtered_mask) else "")
    )

    available_indices = np.where(filtered_mask > 0.5)[0]
    if available_indices.size == 0:
        print("  no available action after filtering")
        return

    current_cell = environment.current_cell
    if current_cell is not None:
        print(
            f"  current_cell: level={current_cell.level}, seq={current_cell.quadrant_sequence}, "
            f"trajs={len(current_cell.trajectories)}"
        )

    sorted_indices = available_indices[np.argsort(probs[available_indices])][::-1]
    preview = min(10, sorted_indices.size)
    print(f"  policy preview (top {preview}):")
    for idx in sorted_indices[:preview]:
        node = environment.all_cells[idx]
        print(
            f"    action={idx:3d} | L{node.level} seq={node.quadrant_sequence} "
            f"| sim={similarities[idx]:.4f} | prob={probs[idx]:.4f} | trajs={len(node.trajectories)}"
        )
    if sorted_indices.size > preview:
        print(f"    ... total candidates={sorted_indices.size}")


def run_debug_episode(
    trainer: TraversalTrainer,
    environment,
    agent,
    episodes: int = 3,
    verbose_logging: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """运行若干 episode 并打印关键调试信息。"""
    total_nodes = len(environment.quadtree.all_cells)
    active_nodes = environment.num_cells

    print("--- Environment Stats ---")
    print(f"all_nodes                          : {total_nodes}")
    print(f"active_nodes                       : {active_nodes}")
    print(f"muted_nodes                        : {total_nodes - active_nodes}")
    print("-" * 20)

    logs: List[dict] = []
    summaries: List[dict] = []
    device = agent.device

    for episode in range(episodes):
        state, action_mask = environment.reset()
        done = False
        step = 0
        print(f"\n>>>> Episode {episode + 1}/{episodes}")

        episode_reward = 0.0
        episode_global_trigger_count = 0
        episode_steps = 0
        episode_local_reward_sum = 0.0
        episode_global_reward_sum = 0.0
        episode_abs_local_reward_sum = 0.0
        episode_abs_global_reward_sum = 0.0

        while not done and step < environment.num_cells:
            filtered_mask = trainer._refine_action_mask(environment, action_mask)

            with torch.no_grad():
                state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                mask_t = torch.as_tensor(filtered_mask, dtype=torch.float32, device=device).unsqueeze(0)
                logits = agent.actor(state_t)
                masked_logits = logits + (mask_t - 1.0) * 1e9
                probs = torch.softmax(masked_logits, dim=-1).squeeze(0).cpu().numpy()

            available_indices = np.where(filtered_mask > 0.5)[0]
            similarities = np.zeros_like(probs)
            if available_indices.size > 0 and environment.current_cell is not None:
                for idx in available_indices:
                    similarities[idx] = environment.cost_evaluator.jaccard_similarity(
                        environment.current_cell,
                        environment.all_cells[idx],
                    )

            print_action_diagnostics(step, environment, action_mask, filtered_mask, probs, similarities)

            action, log_prob, value = agent.select_action(state, filtered_mask)
            next_state, next_mask, reward, done, info = environment.step(action)

            local_reward = info.get("local_reward", 0.0)
            global_reward = info.get("global_reward", 0.0)
            global_triggered = info.get("global_reward_triggered", False)
            current_step = info.get("current_step", step + 1)
            penalty = info.get("penalty")
            global_diag = info.get("global_reward_diagnostics") or {}
            chosen_prob = float(probs[action])
            cell_trajs = len(environment.all_cells[action].trajectories)

            global_flag = " [GLOBAL]" if global_triggered else ""
            raw_cost_diff_str = f", RawDiff:{info.get('raw_cost_diff', 0):.2f}" if global_triggered else ""
            print(
                f"  step={step} action={action:3d} reward={reward:.4f} "
                f"[local={local_reward:.4f}, global={global_reward:.4f}{global_flag}{raw_cost_diff_str}] "
                f"prob={chosen_prob:.4f}, value={value.item():.4f}"
            )
            if global_triggered and global_diag:
                print(
                    "    global_diag: "
                    f"step={current_step}, "
                    f"prefix={global_diag.get('visited_prefix_len', 0)}, "
                    f"tail={global_diag.get('tail_len', 0)}, "
                    f"queries={global_diag.get('num_queries', 0)}, "
                    f"quadcode_avg={global_diag.get('avg_quadcode_cost', 0.0):.4f}, "
                    f"quadorder_avg={global_diag.get('avg_quadorder_cost', 0.0):.4f}, "
                    f"intervals_avg={global_diag.get('avg_quadorder_interval_count', 0.0):.4f}"
                )
            if penalty is not None and penalty < 0:
                print(f"    invalid action penalty: {penalty:.4f}")

            if verbose_logging:
                logs.append({
                    "stage": "step",
                    "episode": episode,
                    "step": step,
                    "action": int(action),
                    "reward": float(reward),
                    "local_reward": float(local_reward),
                    "global_reward": float(global_reward),
                    "global_triggered": bool(global_triggered),
                    "current_step": int(current_step),
                    "value": float(value.item()),
                    "penalty": float(penalty) if penalty is not None else None,
                    "chosen_prob": chosen_prob,
                    "cell_trajs": cell_trajs,
                    "top_k_candidates": int(np.sum(filtered_mask)),
                })

            episode_reward += reward
            episode_global_trigger_count += int(global_triggered)
            episode_steps += 1
            episode_local_reward_sum += float(local_reward)
            episode_global_reward_sum += float(global_reward)
            episode_abs_local_reward_sum += abs(float(local_reward))
            episode_abs_global_reward_sum += abs(float(global_reward))

            agent.store_transition(state, action, reward, log_prob, value, filtered_mask, done)
            state, action_mask = next_state, next_mask
            step += 1

        update_info = agent.update()
        combined_abs_reward = episode_abs_local_reward_sum + episode_abs_global_reward_sum
        global_abs_share = (
            episode_abs_global_reward_sum / combined_abs_reward if combined_abs_reward > 0 else 0.0
        )
        avg_global_reward_when_triggered = (
            episode_global_reward_sum / episode_global_trigger_count
            if episode_global_trigger_count > 0
            else 0.0
        )
        avg_abs_global_reward_when_triggered = (
            episode_abs_global_reward_sum / episode_global_trigger_count
            if episode_global_trigger_count > 0
            else 0.0
        )
        global_to_local_abs_ratio = (
            episode_abs_global_reward_sum / episode_abs_local_reward_sum
            if episode_abs_local_reward_sum > 0
            else 0.0
        )
        summary = {
            "stage": "episode_summary",
            "episode": episode,
            "steps": episode_steps,
            "total_reward": episode_reward,
            "global_trigger_count": episode_global_trigger_count,
            "local_reward_sum": episode_local_reward_sum,
            "global_reward_sum": episode_global_reward_sum,
            "abs_local_reward_sum": episode_abs_local_reward_sum,
            "abs_global_reward_sum": episode_abs_global_reward_sum,
            "global_abs_share": global_abs_share,
            "global_to_local_abs_ratio": global_to_local_abs_ratio,
            "avg_global_reward_when_triggered": avg_global_reward_when_triggered,
            "avg_abs_global_reward_when_triggered": avg_abs_global_reward_when_triggered,
            "loss_stats": update_info,
            "lr_actor": agent.actor_optimizer.param_groups[0]["lr"],
        }
        summaries.append(summary)
        logs.append(summary)
        print(f"\n[Episode {episode + 1} finished] steps={episode_steps}, total_reward={episode_reward:.4f}")
        print(
            "  reward_mix: "
            f"local_sum={episode_local_reward_sum:.4f}, "
            f"global_sum={episode_global_reward_sum:.4f}, "
            f"abs_global_share={global_abs_share:.2%}, "
            f"abs_g_over_l={global_to_local_abs_ratio:.4f}"
        )
        print(
            "  global_events: "
            f"count={episode_global_trigger_count}, "
            f"avg={avg_global_reward_when_triggered:.4f}, "
            f"avg_abs={avg_abs_global_reward_when_triggered:.4f}"
        )
        if update_info:
            print(f"  -> Loss: {update_info.get('loss', 0):.6f} (Actor: {update_info.get('actor_loss', 0):.6f})")

    print("\n" + "=" * 40)
    print("        ==== Final Evaluation ====        ")
    print("=" * 40)
    quadorder = environment.quadorder()
    evaluator = TraversalPerformanceEvaluator(
        environment.quadtree,
        TraversalOrderEncoder(environment.quadtree, environment.alpha, environment.beta),
        environment.cost_evaluator,
        reference_queries=environment.reference_queries,
        quadcode_include_muted=environment.quadcode_include_muted,
    )
    comparison = evaluator.evaluate_final_order(quadorder)

    print(f"quadcode_avg_cost                 : {comparison['quadcode_avg_cost']:.4f}")
    print(f"quadorder_avg_cost                : {comparison['quadorder_avg_cost']:.4f}")
    print(f"improvement_percent               : {comparison['improvement_percent']:.2f}%")
    print(f"active_nodes_in_order             : {len(quadorder)}")

    final_entry = {
        "stage": "final_evaluation",
        "improvement_percent": float(comparison["improvement_percent"]),
        "quadorder_avg_cost": float(comparison["quadorder_avg_cost"]),
        "quadcode_avg_cost": float(comparison["quadcode_avg_cost"]),
        "quadorder_scan_counts": float(np.mean(comparison["quadorder_scan_counts"])) if comparison["quadorder_scan_counts"] else 0.0,
        "quadcode_scan_counts": float(np.mean(comparison["quadcode_scan_counts"])) if comparison["quadcode_scan_counts"] else 0.0,
        "total_active_nodes": active_nodes,
    }
    logs.append(final_entry)

    return logs, summaries, final_entry


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Debug RL training process")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    set_seed(42)
    config = TShapeConfig.from_yaml(args.config)
    if config.experiment.name == "default":
        config.experiment.name = Path(args.config).resolve().parent.name

    trainer = TraversalTrainer(config, config.network)
    print_header(config, trainer)

    print("--- System Setup ---")
    environment, _ = trainer.setup()
    agent = trainer.prepare_agent()

    print(f"state_dim                          : {environment.state_dimension}")
    print(f"action_dim                         : {environment.action_dimension}")
    print("setup complete, starting debug episodes...\n")

    logs, summaries, final_entry = run_debug_episode(
        trainer,
        environment,
        agent,
        episodes=config.train.num_episodes,
        verbose_logging=True,
    )

    log_dir = config.experiment.get_logs_dir(config.paths)
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"debug_log_{timestamp}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)

    reward_vals = [summary["total_reward"] for summary in summaries]
    local_reward_vals = [summary["local_reward_sum"] for summary in summaries]
    global_reward_vals = [summary["global_reward_sum"] for summary in summaries]
    global_share_vals = [summary["global_abs_share"] for summary in summaries]
    final_loss = summaries[-1]["loss_stats"].get("loss", 0) if summaries and summaries[-1].get("loss_stats") else 0
    print(f"\n[Done] debug_log                    : {log_path}")
    print(f"average_reward                     : {np.mean(reward_vals):.4f}")
    print(f"average_local_reward_sum           : {np.mean(local_reward_vals):.4f}")
    print(f"average_global_reward_sum          : {np.mean(global_reward_vals):.4f}")
    print(f"average_global_abs_share           : {np.mean(global_share_vals):.2%}")
    print(f"final_loss                         : {final_loss:.6f}")
    print(f"final_improvement                  : {final_entry['improvement_percent']:.2f}%")
    print("-" * 30)


if __name__ == "__main__":
    main()



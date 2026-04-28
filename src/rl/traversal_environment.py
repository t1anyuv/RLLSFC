from typing import List, Optional, Set, Tuple

import numpy as np

from src.common import SpatialBoundingBox
from src.features.state_feature_builder import StateFeatureBuilder
from src.indexing.quadtree_cell import QuadTreeCell
from src.indexing.quadtree_index import QuadTreeIndex
from src.common import TraversalCostEvaluator


class TraversalEnvironment:
    """Environment for learning a traversal order over quadtree cells."""

    def __init__(
        self,
        quadtree: QuadTreeIndex,
        cost_evaluator: TraversalCostEvaluator,
        reference_queries: Optional[List[SpatialBoundingBox]] = None,
        val_queries: Optional[List[SpatialBoundingBox]] = None,
        test_queries: Optional[List[SpatialBoundingBox]] = None,
        alpha: int = 2,
        beta: int = 2,
        exclude_muted_cells: bool = True,
        quadcode_include_muted: bool = True,
        local_reward_weight: float = 0.5,
        global_reward_weight: float = 1.0,
        reward_schedule_episodes: int = 0,
        local_reward_start_scale: float = 1.0,
        global_reward_start_scale: float = 1.0,
        global_reward_scale: float = 1.0,
        global_reward_num_evals: int = 1,
        global_reward_query_sample_size: Optional[int] = None,
        global_reward_frontload_exponent: float = 1.0,
    ):
        """Initialize the traversal environment and precompute reward baselines."""
        if reference_queries is None:
            raise ValueError("reference_queries are required")

        self.quadtree = quadtree
        self.cost_evaluator = cost_evaluator
        self.reference_queries = list(reference_queries)
        self.val_queries = list(val_queries) if val_queries is not None else None
        self.test_queries = list(test_queries) if test_queries is not None else None
        self.alpha = alpha
        self.beta = beta
        self.exclude_muted_cells = exclude_muted_cells
        self.quadcode_include_muted = quadcode_include_muted
        self.local_reward_weight = local_reward_weight
        self.global_reward_weight = global_reward_weight
        self.reward_schedule_episodes = max(0, reward_schedule_episodes)
        self.local_reward_start_scale = local_reward_start_scale
        self.global_reward_start_scale = global_reward_start_scale
        self.global_reward_scale = global_reward_scale
        self.global_reward_clip_max = 100.0
        self.global_reward_clip_min = -100.0
        self.global_reward_num_evals = global_reward_num_evals
        self.global_reward_query_sample_size = global_reward_query_sample_size
        self.global_reward_frontload_exponent = max(1.0, global_reward_frontload_exponent)

        self.all_cells: List[QuadTreeCell] = (
            self.quadtree.get_active_cells()
            if exclude_muted_cells
            else list(self.quadtree.all_cells.values())
        )
        self.cell_to_index = {cell: index for index, cell in enumerate(self.all_cells)}
        self.num_cells = len(self.all_cells)
        self._global_eval_checkpoints = self._build_global_eval_checkpoints()
        self.max_distance = self._estimate_max_distance()
        self.feature_builder = StateFeatureBuilder(quadtree, alpha, beta, self.all_cells)

        self.current_cell: Optional[QuadTreeCell] = None
        self.prev_cell: Optional[QuadTreeCell] = None
        self.visited_cells: Set[QuadTreeCell] = set()
        self.visited_order: List[QuadTreeCell] = []
        self.available_actions: List[int] = []
        self.current_episode = 0
        self.last_action_filter_info = {}

        self._init_global_reward_cache()

    def _init_global_reward_cache(self) -> None:
        """Build reusable encoder/evaluator state for global reward computation."""
        from src.evaluation.traversal_evaluator import TraversalPerformanceEvaluator
        from src.indexing.traversal_encoder import TraversalOrderEncoder

        self._cached_encoder = TraversalOrderEncoder(self.quadtree, self.alpha, self.beta)
        self._cached_quadcode_order = self._cached_encoder.z_curve_order(include_muted=self.quadcode_include_muted)
        self._cached_evaluator = TraversalPerformanceEvaluator(
            self.quadtree,
            self._cached_encoder,
            self.cost_evaluator,
            reference_queries=self.reference_queries,
            quadcode_include_muted=self.quadcode_include_muted,
        )
        self._precompute_quadcode_intervals()

    def _precompute_quadcode_intervals(self) -> None:
        """Cache quadCode interval costs for all reference queries."""
        self._cached_quadcode_intervals = []
        self._cached_quadcode_costs = []
        self._cached_quadcode_gap_move_bits = []

        for query in self.reference_queries:
            intervals, _, quadcode_gap_move_bits = self._cached_evaluator.search_quadcode_intervals(
                query,
                self._cached_quadcode_order,
                skip_muted=not self.quadcode_include_muted,
            )
            self._cached_quadcode_intervals.append(intervals)
            self._cached_quadcode_gap_move_bits.append(quadcode_gap_move_bits)
            quadcode_cost = self.cost_evaluator.query_cost(intervals, gap_move_bits=quadcode_gap_move_bits)
            self._cached_quadcode_costs.append(quadcode_cost)

    @property
    def state_dimension(self) -> int:
        """Return the state feature dimension exposed to the policy."""
        return self.feature_builder.state_dimensions

    @property
    def action_dimension(self) -> int:
        """Return the size of the discrete action space."""
        return self.num_cells

    def _estimate_max_distance(self) -> float:
        """Estimate the maximum center-to-center distance inside the quadtree bbox."""
        width = self.quadtree.bbox.max_x - self.quadtree.bbox.min_x
        height = self.quadtree.bbox.max_y - self.quadtree.bbox.min_y
        return float(np.sqrt(width ** 2 + height ** 2))

    def reset(self, start_cell: Optional[QuadTreeCell] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Reset episode state and return the initial observation plus action mask."""
        self.current_cell = start_cell or self.quadtree.root
        self.prev_cell = None
        self.visited_cells = {self.current_cell}
        self.visited_order = [self.current_cell]
        self.available_actions = self._compute_available_actions()
        self.current_episode += 1
        self._global_eval_checkpoints = self._build_global_eval_checkpoints()
        self._global_query_indices_by_checkpoint = self._prepare_global_query_samples()
        self._prefix_order_lookup_cache = {}
        self._global_reward_result_cache = {}
        return self._build_state()

    def _build_global_eval_checkpoints(self) -> Set[int]:
        """Compute the step indices where global reward should be evaluated."""
        checkpoints: Set[int] = set()
        if self.global_reward_num_evals <= 0 or self.num_cells <= 0:
            return checkpoints

        for idx in range(1, self.global_reward_num_evals + 1):
            fraction = (idx / self.global_reward_num_evals) ** self.global_reward_frontload_exponent
            checkpoints.add(max(1, min(self.num_cells, int(np.ceil(self.num_cells * fraction)))))
        return checkpoints

    def _current_reward_weights(self) -> Tuple[float, float]:
        """Return scheduled local/global reward weights for the current episode."""
        if self.reward_schedule_episodes <= 0:
            return self.local_reward_weight, self.global_reward_weight

        progress = min(1.0, max(0.0, (self.current_episode - 1) / self.reward_schedule_episodes))
        local_scale = self.local_reward_start_scale + (1.0 - self.local_reward_start_scale) * progress
        global_scale = self.global_reward_start_scale + (1.0 - self.global_reward_start_scale) * progress
        return self.local_reward_weight * local_scale, self.global_reward_weight * global_scale

    def _sample_global_query_indices(self, current_step: int, done: bool) -> List[int]:
        """Sample query indices for training-time global reward estimation."""
        cached = self._global_query_indices_by_checkpoint.get(current_step)
        if cached is not None:
            return cached

        total_queries = len(self.reference_queries)
        if total_queries == 0:
            return []

        sample_size = self.global_reward_query_sample_size
        if done or sample_size is None or sample_size <= 0 or sample_size >= total_queries:
            return list(range(total_queries))

        rng_seed = (self.current_episode * 1000003) ^ (current_step * 9176) ^ self.num_cells
        rng = np.random.default_rng(rng_seed)
        sampled = rng.choice(total_queries, size=sample_size, replace=False)
        return sorted(int(idx) for idx in sampled.tolist())

    def _prepare_global_query_samples(self) -> dict:
        """Precompute query samples for each checkpoint within the current episode."""
        samples = {}
        for checkpoint in self._global_eval_checkpoints:
            done = checkpoint >= self.num_cells
            samples[checkpoint] = self._sample_global_query_indices_uncached(checkpoint, done)
        return samples

    def _sample_global_query_indices_uncached(self, current_step: int, done: bool) -> List[int]:
        """Internal sampling helper used while preparing per-checkpoint query batches."""
        total_queries = len(self.reference_queries)
        if total_queries == 0:
            return []

        sample_size = self.global_reward_query_sample_size
        if done or sample_size is None or sample_size <= 0 or sample_size >= total_queries:
            return list(range(total_queries))

        rng_seed = (self.current_episode * 1000003) ^ (current_step * 9176) ^ self.num_cells
        rng = np.random.default_rng(rng_seed)
        sampled = rng.choice(total_queries, size=sample_size, replace=False)
        return sorted(int(idx) for idx in sampled.tolist())

    @staticmethod
    def _prefix_cache_key(quadorder: List[QuadTreeCell]) -> Tuple[int, ...]:
        return tuple(int(cell.code) for cell in quadorder)

    def _is_actionable_cell(self, cell: Optional[QuadTreeCell]) -> bool:
        """Return whether a cell is eligible to appear in the action space."""
        if cell is None or cell in self.visited_cells:
            return False
        if self.exclude_muted_cells and cell.muted:
            return False
        return cell in self.cell_to_index

    def _collect_direct_neighbors(self, current_cell: QuadTreeCell) -> Tuple[Set[QuadTreeCell], List[QuadTreeCell]]:
        """Gather unvisited neighboring candidates around the current cell."""
        candidate_cells: Set[QuadTreeCell] = set()
        valid_neighbors: List[QuadTreeCell] = []

        for neighbor in self.quadtree.get_eight_neighbor_cells(current_cell):
            if not self._is_actionable_cell(neighbor):
                continue
            candidate_cells.add(neighbor)
            valid_neighbors.append(neighbor)

        return candidate_cells, valid_neighbors

    def _search_active_layer(self, layer: Set[QuadTreeCell], direction: str) -> List[QuadTreeCell]:
        """Search upward or downward until a selectable layer of cells is found."""
        current_layer = layer
        while current_layer:
            active_cells = [
                cell for cell in current_layer
                if self._is_actionable_cell(cell)
            ]
            if active_cells:
                return active_cells

            if direction == "up":
                current_layer = {cell.parent for cell in current_layer if cell and cell.parent is not None}
            else:
                current_layer = {
                    child
                    for cell in current_layer if cell
                    for child in cell.children
                    if child is not None
                }

        return []

    def _find_fallback_candidate(self) -> Optional[QuadTreeCell]:
        """Pick the nearest remaining selectable cell when local expansion is exhausted."""
        if self.current_cell is None:
            return None

        remaining = [
            cell for cell in self.all_cells
            if self._is_actionable_cell(cell)
        ]
        if not remaining:
            return None

        cx, cy = self.current_cell.get_center()
        return min(
            remaining,
            key=lambda cell: (cell.get_center()[0] - cx) ** 2 + (cell.get_center()[1] - cy) ** 2,
        )

    def _compute_available_actions(self) -> List[int]:
        """Compute candidate next actions using local neighbors plus hierarchical fallback."""
        if self.current_cell is None:
            return []

        current = self.current_cell
        candidate_cells, valid_neighbors = self._collect_direct_neighbors(current)
        nodes = [current] + valid_neighbors

        parent_layer = {node.parent for node in nodes if node.parent is not None}
        candidate_cells.update(self._search_active_layer(parent_layer, "up"))

        child_layer = {child for node in nodes for child in node.children if child is not None}
        candidate_cells.update(self._search_active_layer(child_layer, "down"))

        if not candidate_cells:
            fallback_cell = self._find_fallback_candidate()
            if fallback_cell is not None:
                candidate_cells.add(fallback_cell)

        return [self.cell_to_index[cell] for cell in candidate_cells if cell in self.cell_to_index]

    def _build_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Build the policy input features and corresponding action mask."""
        visited_ratio = len(self.visited_cells) / max(1, self.num_cells)
        feature_vector = self.feature_builder.get_features(
            self.current_cell,
            self.prev_cell,
            visited_ratio,
            self.visited_cells,
        )

        action_mask = np.zeros(self.num_cells, dtype=np.float32)
        for action_index in self.available_actions:
            action_mask[action_index] = 1.0

        return feature_vector, action_mask

    def _compute_global_reward(self, quadorder: List[QuadTreeCell], current_step: int, done: bool) -> Tuple[float, float, dict]:
        """Evaluate the current prefix against cached quadCode query costs."""
        prefix_key = self._prefix_cache_key(quadorder)
        cached_result = self._global_reward_result_cache.get(prefix_key)
        if cached_result is not None:
            return cached_result

        total_reward = 0.0
        total_raw_cost_diff = 0.0
        query_indices = self._sample_global_query_indices(current_step, done)
        num_queries = len(query_indices)
        quadorder_interval_counts = []
        quadcode_interval_counts = []
        quadorder_costs = []
        quadorder_lookup = self._prefix_order_lookup_cache.get(prefix_key)
        if quadorder_lookup is None:
            quadorder_lookup = self._cached_evaluator._build_order_lookup(quadorder)
            self._prefix_order_lookup_cache[prefix_key] = quadorder_lookup

        for query_idx in query_indices:
            query_bbox = self.reference_queries[query_idx]
            quadorder_intervals, _, quadorder_gap_move_bits = self._cached_evaluator.search_quadorder_intervals(
                query_bbox,
                quadorder,
                skip_muted=True,
                aligned=quadorder_lookup,
            )
            quadcode_cost = self._cached_quadcode_costs[query_idx]
            quadcode_intervals = self._cached_quadcode_intervals[query_idx]
            reward, cost_diff = self.cost_evaluator.global_reward(
                quadorder_intervals,
                quadcode_cost,
                self.global_reward_scale,
                quadorder_gap_move_bits=quadorder_gap_move_bits,
            )
            quadorder_cost = quadcode_cost - cost_diff
            total_reward += reward
            total_raw_cost_diff += cost_diff
            quadorder_interval_counts.append(len(quadorder_intervals))
            quadcode_interval_counts.append(len(quadcode_intervals))
            quadorder_costs.append(float(quadorder_cost))

        avg_reward = total_reward / num_queries if num_queries > 0 else 0.0
        raw_cost_diff = total_raw_cost_diff / num_queries if num_queries > 0 else 0.0
        diagnostics = {
            "num_queries": num_queries,
            "total_reference_queries": len(self.reference_queries),
            "avg_quadcode_cost": float(np.mean(self._cached_quadcode_costs)) if self._cached_quadcode_costs else 0.0,
            "avg_quadorder_cost": float(np.mean(quadorder_costs)) if quadorder_costs else 0.0,
            "avg_quadcode_interval_count": float(np.mean(quadcode_interval_counts)) if quadcode_interval_counts else 0.0,
            "avg_quadorder_interval_count": float(np.mean(quadorder_interval_counts)) if quadorder_interval_counts else 0.0,
            "visited_prefix_len": len(quadorder),
            "tail_len": max(0, self.num_cells - len(quadorder)),
        }
        result = (avg_reward, raw_cost_diff, diagnostics)
        self._global_reward_result_cache[prefix_key] = result
        return result

    def step(self, action: int) -> Tuple[np.ndarray, np.ndarray, float, bool, dict]:
        """Apply one action and return next state, mask, reward, done flag, and diagnostics."""
        if len(self.visited_cells) >= self.num_cells:
            state, mask = self._build_state()
            return state, mask, 0.0, True, {
                "visited_count": self.num_cells,
                "total_cells": self.num_cells,
            }

        if action not in self.available_actions:
            state, mask = self._build_state()
            return state, mask, -0.1, False, {"error": "invalid_action", "penalty": -0.1}

        next_cell = self.all_cells[action]
        local_reward = self.cost_evaluator.step_reward(
            self.current_cell,
            next_cell,
            self.max_distance,
            proximity_weight=0.5,
            similarity_weight=0.5,
        )

        self.prev_cell = self.current_cell
        self.current_cell = next_cell
        self.visited_cells.add(next_cell)
        self.visited_order.append(next_cell)
        self.available_actions = self._compute_available_actions()

        done = len(self.visited_cells) >= self.num_cells
        current_step = len(self.visited_order)
        raw_cost_diff = 0.0
        global_trigger = False
        local_weight, global_weight = self._current_reward_weights()

        if current_step in self._global_eval_checkpoints:
            global_reward, raw_cost_diff, global_diagnostics = self._compute_global_reward(
                self.visited_order,
                current_step=current_step,
                done=done,
            )
            global_reward_clipped = float(
                np.clip(global_reward, self.global_reward_clip_min, self.global_reward_clip_max)
            )
            global_trigger = True
        else:
            global_reward_clipped = 0.0
            global_diagnostics = None

        reward = local_weight * local_reward + global_weight * global_reward_clipped
        info = {
            "visited_count": len(self.visited_cells),
            "total_cells": self.num_cells,
            "local_reward": local_reward,
            "global_reward": global_reward_clipped,
            "local_reward_weight": local_weight,
            "global_reward_weight": global_weight,
            "raw_cost_diff": raw_cost_diff,
            "global_reward_triggered": global_trigger,
            "current_step": current_step,
            "global_reward_diagnostics": global_diagnostics,
        }

        next_state, next_action_mask = self._build_state()
        return next_state, next_action_mask, reward, done, info

    def quadorder(self) -> List[QuadTreeCell]:
        """Return a copy of the current visited traversal order."""
        return self.visited_order.copy()

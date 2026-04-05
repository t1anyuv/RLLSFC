from src.core.bounding_box import SpatialBoundingBox
from src.indexing.quadtree_index import QuadTreeIndex
from src.reward.cost_evaluator import TraversalCostEvaluator
from src.rl.traversal_environment import TraversalEnvironment


def build_environment(exclude_muted_cells: bool) -> TraversalEnvironment:
    bbox = SpatialBoundingBox(0.0, 0.0, 1.0, 1.0)
    quadtree = QuadTreeIndex(bbox, max_level=1, alpha=2, beta=2)
    cost_evaluator = TraversalCostEvaluator(quadtree)
    return TraversalEnvironment(
        quadtree=quadtree,
        cost_evaluator=cost_evaluator,
        reference_queries=[bbox],
        exclude_muted_cells=exclude_muted_cells,
        global_reward_num_evals=0,
    )


def test_reset_exposes_root_children_as_initial_actions():
    env = build_environment(exclude_muted_cells=True)

    state, action_mask = env.reset()

    assert state.shape == (env.state_dimension,)
    assert int(action_mask.sum()) == 4


def test_muted_cells_remain_selectable_when_not_excluded():
    env = build_environment(exclude_muted_cells=False)
    muted_child = env.quadtree.root.children[0]
    muted_child.muted = True

    _, action_mask = env.reset()

    assert env.num_cells == 5
    assert action_mask[env.cell_to_index[muted_child]] == 1.0


def test_reward_weight_schedule_prefers_local_signal_early():
    bbox = SpatialBoundingBox(0.0, 0.0, 1.0, 1.0)
    quadtree = QuadTreeIndex(bbox, max_level=1, alpha=2, beta=2)
    env = TraversalEnvironment(
        quadtree=quadtree,
        cost_evaluator=TraversalCostEvaluator(quadtree),
        reference_queries=[bbox],
        local_reward_weight=0.5,
        global_reward_weight=1.0,
        reward_schedule_episodes=4,
        local_reward_start_scale=1.0,
        global_reward_start_scale=0.25,
        global_reward_num_evals=0,
    )

    env.reset()
    assert env._current_reward_weights() == (0.5, 0.25)

    env.current_episode = 5
    assert env._current_reward_weights() == (0.5, 1.0)


def test_global_reward_query_sampling_uses_subset_until_done():
    bbox = SpatialBoundingBox(0.0, 0.0, 1.0, 1.0)
    queries = [bbox for _ in range(10)]
    quadtree = QuadTreeIndex(bbox, max_level=1, alpha=2, beta=2)
    env = TraversalEnvironment(
        quadtree=quadtree,
        cost_evaluator=TraversalCostEvaluator(quadtree),
        reference_queries=queries,
        global_reward_num_evals=1,
        global_reward_query_sample_size=3,
    )

    env.reset()
    sampled = env._sample_global_query_indices(current_step=2, done=False)
    sampled_done = env._sample_global_query_indices(current_step=env.num_cells, done=True)

    assert len(sampled) == 3
    assert sampled == sorted(sampled)
    assert len(sampled_done) == len(queries)


def test_checkpoint_query_samples_are_precomputed_and_reused():
    bbox = SpatialBoundingBox(0.0, 0.0, 1.0, 1.0)
    queries = [bbox for _ in range(10)]
    quadtree = QuadTreeIndex(bbox, max_level=1, alpha=2, beta=2)
    env = TraversalEnvironment(
        quadtree=quadtree,
        cost_evaluator=TraversalCostEvaluator(quadtree),
        reference_queries=queries,
        global_reward_num_evals=2,
        global_reward_query_sample_size=3,
    )

    env.reset()
    first_checkpoint = min(env._global_eval_checkpoints)
    last_checkpoint = max(env._global_eval_checkpoints)

    first_sample = env._sample_global_query_indices(current_step=first_checkpoint, done=False)
    second_sample = env._sample_global_query_indices(current_step=first_checkpoint, done=False)
    final_sample = env._sample_global_query_indices(current_step=last_checkpoint, done=True)

    assert first_sample is second_sample
    assert len(first_sample) == 3
    assert len(final_sample) == len(queries)


def test_frontloaded_global_checkpoints_bias_early_steps():
    bbox = SpatialBoundingBox(0.0, 0.0, 1.0, 1.0)
    quadtree = QuadTreeIndex(bbox, max_level=1, alpha=2, beta=2)
    env = TraversalEnvironment(
        quadtree=quadtree,
        cost_evaluator=TraversalCostEvaluator(quadtree),
        reference_queries=[bbox],
        global_reward_num_evals=4,
        global_reward_frontload_exponent=2.0,
    )

    env.num_cells = 8
    checkpoints = env._build_global_eval_checkpoints()

    assert checkpoints == {1, 2, 5, 8}


def test_global_reward_reuses_single_order_lookup_for_sampled_queries(monkeypatch):
    bbox = SpatialBoundingBox(0.0, 0.0, 1.0, 1.0)
    queries = [bbox for _ in range(3)]
    quadtree = QuadTreeIndex(bbox, max_level=1, alpha=2, beta=2)
    env = TraversalEnvironment(
        quadtree=quadtree,
        cost_evaluator=TraversalCostEvaluator(quadtree),
        reference_queries=queries,
        global_reward_num_evals=1,
        global_reward_query_sample_size=None,
    )

    env.reset()
    quadorder = env.quadorder()
    evaluator = env._cached_evaluator
    lookup_calls = {"count": 0}
    search_aligned = []
    original_build = evaluator._build_order_lookup
    original_search = evaluator.search_quadorder_intervals

    def wrapped_build(order):
        lookup_calls["count"] += 1
        return original_build(order)

    def wrapped_search(query_bbox, traversal_order, skip_muted=True, aligned=None):
        search_aligned.append(aligned is not None)
        return original_search(query_bbox, traversal_order, skip_muted=skip_muted, aligned=aligned)

    monkeypatch.setattr(evaluator, "_build_order_lookup", wrapped_build)
    monkeypatch.setattr(evaluator, "search_quadorder_intervals", wrapped_search)

    _, _, diagnostics = env._compute_global_reward(quadorder, current_step=1, done=False)

    assert lookup_calls["count"] == 1
    assert search_aligned == [True, True, True]
    assert diagnostics["num_queries"] == 3


def test_global_reward_caches_repeated_prefix_evaluations(monkeypatch):
    bbox = SpatialBoundingBox(0.0, 0.0, 1.0, 1.0)
    queries = [bbox for _ in range(2)]
    quadtree = QuadTreeIndex(bbox, max_level=1, alpha=2, beta=2)
    env = TraversalEnvironment(
        quadtree=quadtree,
        cost_evaluator=TraversalCostEvaluator(quadtree),
        reference_queries=queries,
        global_reward_num_evals=1,
        global_reward_query_sample_size=None,
    )

    env.reset()
    quadorder = env.quadorder()
    evaluator = env._cached_evaluator
    search_calls = {"count": 0}
    original_search = evaluator.search_quadorder_intervals

    def wrapped_search(*args, **kwargs):
        search_calls["count"] += 1
        return original_search(*args, **kwargs)

    monkeypatch.setattr(evaluator, "search_quadorder_intervals", wrapped_search)

    first = env._compute_global_reward(quadorder, current_step=1, done=False)
    second = env._compute_global_reward(quadorder, current_step=1, done=False)

    assert search_calls["count"] == len(queries)
    assert first == second

import os
import tempfile
import unittest

from shapely.geometry import LineString, box

from src.config import TShapeConfig
from src.common import SpatialBoundingBox
from src.data import load_cleaned_dataset
from src.evaluation import TraversalPerformanceEvaluator
from src.indexing.quadtree_index import QuadTreeIndex
from src.indexing.traversal_encoder import TraversalOrderEncoder
from src.common import TraversalCostEvaluator
from src.rl.order_formatter import TrajectoryOrderFormatter
from src.utils.path_manager import get_path_manager
from src.utils.signature import compute_query_signature, compute_traj_signature


class TestTShapeSearchCorrectness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bbox = SpatialBoundingBox(115.29, 39.00, 117.83, 41.50)
        cls.max_level = 8
        cls.alpha, cls.beta = 3, 3

        print("Pre-loading TDrive data...")
        pm = get_path_manager()
        default_config = TShapeConfig.from_yaml(str(pm.project_root / "default.yaml"))
        configured_tdrive_path = default_config.get_dataset_trajectory_path()
        tdrive_path = (
            os.environ.get("TDRIVE_DATA_PATH")
            or (str(configured_tdrive_path) if configured_tdrive_path is not None else None)
            or r"D:\Dataset\Trajectory\TDrive\complete_clean\tdrive.txt"
        )
        if not tdrive_path or not os.path.exists(tdrive_path):
            raise unittest.SkipTest(f"TDrive dataset not found: {tdrive_path}")

        cls.raw_trajectories = load_cleaned_dataset(
            str(tdrive_path),
            max_trajectories=None,
        )

        cls.test_queries = [
            SpatialBoundingBox(116.3, 39.9, 116.4, 40.0),
            SpatialBoundingBox(116.35, 39.95, 116.38, 39.98),
            SpatialBoundingBox(116.5, 39.5, 116.6, 39.6),
            SpatialBoundingBox(115.8, 39.3, 116.0, 39.5),
            SpatialBoundingBox(116.0, 39.9, 116.5, 39.92),
            SpatialBoundingBox(116.3, 39.8, 116.32, 40.1),
            SpatialBoundingBox(116.6, 39.9, 116.7, 40.1),
            SpatialBoundingBox(116.391, 39.901, 116.395, 39.905),
            SpatialBoundingBox(116.1, 39.7, 116.6, 40.2),
        ]

    def _init_new_index(self):
        index = QuadTreeIndex(self.bbox, self.max_level, self.alpha, self.beta)
        for traj_id, points in self.raw_trajectories:
            index.assign_trajectory(traj_id, points)
        return index

    @staticmethod
    def _filter_actual_hits(index, candidate_ids, query_bbox: SpatialBoundingBox) -> set:
        actual = set()
        query_poly = box(query_bbox.min_x, query_bbox.min_y, query_bbox.max_x, query_bbox.max_y)
        for tid in candidate_ids:
            pts = index.trajectory_points.get(tid)
            if pts and len(pts) >= 2 and LineString(pts).intersects(query_poly):
                actual.add(tid)
        return actual

    def brute_force_search(self, index, query_bbox: SpatialBoundingBox) -> set:
        hit_ids = set()
        query_poly = box(query_bbox.min_x, query_bbox.min_y, query_bbox.max_x, query_bbox.max_y)
        for tid, points in index.trajectory_points.items():
            if len(points) < 2:
                continue
            if LineString(points).intersects(query_poly):
                hit_ids.add(tid)
        return hit_ids

    def run_search_suite(self, index, mode_name, search_runner):
        print(f"\n>>> Running Search Test Mode: {mode_name}")

        for i, q_bbox in enumerate(self.test_queries):
            expected = self.brute_force_search(index, q_bbox)
            _, candidate_ids, _ = search_runner(q_bbox)
            actual = self._filter_actual_hits(index, candidate_ids, q_bbox)

            if actual != expected:
                missing = expected - actual
                extra = actual - expected

                true_missing = {
                    tid for tid in missing
                    if any(q_bbox.contains_point(p[0], p[1]) for p in index.trajectory_points[tid])
                }

                if true_missing or extra:
                    print(f"FAILED at Query {i} | True Missing: {len(true_missing)}, Extra: {len(extra)}")
                    self.analyze_trajectory(index, actual, expected, q_bbox)

                    error_msg = f"[{mode_name}] Query {i} has mistakes"
                    if true_missing:
                        error_msg += f" Missing TID: {list(true_missing)[:3]}..."
                    if extra:
                        error_msg += f" Extra TID: {list(extra)[:3]}..."
                    self.fail(error_msg)
                else:
                    print(f"  Query {i}: 忽略 {len(missing)} 条点采样真空导致的漏检。")

        print(f"SUCCESS: {mode_name} all queries passed.")

    def test_01_no_pruning(self):
        index = self._init_new_index()
        encoder = TraversalOrderEncoder(index, self.alpha, self.beta)
        evaluator = TraversalPerformanceEvaluator(index, encoder, TraversalCostEvaluator(index))
        z_order = encoder.z_curve_order()
        self.run_search_suite(
            index,
            "NO_PRUNING",
            lambda q_bbox: evaluator.search_quadcode_intervals(q_bbox, z_order, skip_muted=True),
        )

    def test_02_pruning_no_optimize(self):
        index = self._init_new_index()
        index.post_prune_tree(min_cell_trajs=4)
        index.compute_signatures(enable_optimize=False)

        encoder = TraversalOrderEncoder(index, self.alpha, self.beta)
        evaluator = TraversalPerformanceEvaluator(index, encoder, TraversalCostEvaluator(index))
        z_order = encoder.z_curve_order()
        self.run_search_suite(
            index,
            "PRUNING_NO_OPTIMIZE",
            lambda q_bbox: evaluator.search_quadcode_intervals(q_bbox, z_order, skip_muted=True),
        )

    def test_03_pruning_with_optimize(self):
        index = self._init_new_index()
        index.post_prune_tree(min_cell_trajs=4)
        index.compute_signatures(enable_optimize=True)

        encoder = TraversalOrderEncoder(index, self.alpha, self.beta)
        evaluator = TraversalPerformanceEvaluator(index, encoder, TraversalCostEvaluator(index))
        z_order = encoder.z_curve_order()
        self.run_search_suite(
            index,
            "PRUNING_WITH_OPTIMIZE",
            lambda q_bbox: evaluator.search_quadcode_intervals(q_bbox, z_order, skip_muted=True),
        )

    def test_04_loaded_xz_order_uses_effective_subtree_count_for_cover_intervals(self):
        index = self._init_new_index()
        index.post_prune_tree(min_cell_trajs=4)
        index.compute_signatures(enable_optimize=True)

        baseline_encoder = TraversalOrderEncoder(index, self.alpha, self.beta)
        baseline_evaluator = TraversalPerformanceEvaluator(
            index,
            baseline_encoder,
            TraversalCostEvaluator(index),
        )
        baseline_order = baseline_encoder.z_curve_order()

        with tempfile.TemporaryDirectory() as temp_dir:
            formatter = TrajectoryOrderFormatter(output_dir=temp_dir)
            _, order_path = formatter.generate_config_file_from_order(
                order=baseline_order,
                quadtree=index,
                filename="xz_order.json",
                global_alpha=self.alpha,
                global_beta=self.beta,
                order_source="pruned_default_xz_order",
            )

            loaded_encoder = TraversalOrderEncoder(index, self.alpha, self.beta)
            loaded_encoder.load_quadorder_mapping(order_path)
            loaded_order = loaded_encoder.quadorder()
            self.assertIsNotNone(loaded_order)

            loaded_evaluator = TraversalPerformanceEvaluator(
                index,
                loaded_encoder,
                TraversalCostEvaluator(index),
            )

            def search_runner(q_bbox):
                baseline_intervals, _, _ = baseline_evaluator.search_quadorder_intervals(
                    q_bbox,
                    baseline_order,
                    skip_muted=True,
                )
                loaded_intervals, candidate_ids, gap_bits = loaded_evaluator.search_quadorder_intervals(
                    q_bbox,
                    loaded_order,
                    skip_muted=True,
                )
                self.assertEqual(
                    baseline_intervals,
                    loaded_intervals,
                    f"Loaded XZ order intervals should match baseline for query {q_bbox}",
                )
                return loaded_intervals, candidate_ids, gap_bits

            self.run_search_suite(
                index,
                "LOADED_XZ_ORDER_WITH_COVERAGE",
                search_runner,
            )

    def analyze_trajectory(self, index, actual_ids, expected_ids, q_bbox):
        missing = expected_ids - actual_ids
        if not missing:
            print("\n[INFO] 无漏检轨迹。")
            return

        error_tid = list(missing)[0]
        traj_points = index.trajectory_points.get(error_tid)
        assigned_cell = index.trajectory_to_cells.get(error_tid)
        traj_mbr = index.trajectory_mbrs.get(error_tid)

        print("\n" + "=" * 60)
        print(f"DEBUG INFO | TID: {error_tid} | Cell: {assigned_cell.code if assigned_cell else 'None'}")
        print(f"\n[漏检数量]: {len(missing)}")
        print(f"\n[几何范围]")
        print(f"  Query BBox: {q_bbox}")
        print(f"  Traj  MBR:  {traj_mbr}")

        in_query_count = sum(1 for pt in traj_points if q_bbox.contains_point(pt[0], pt[1]))
        print(f"\n[采样点状态]")
        print(f"  总点数: {len(traj_points)} | 落在查询框内的点数: {in_query_count}")

        print(f"\n[连线相交分析]")
        query_poly = box(q_bbox.min_x, q_bbox.min_y, q_bbox.max_x, q_bbox.max_y)
        segment_intersects = False

        if len(traj_points) >= 2:
            for i in range(len(traj_points) - 1):
                p1, p2 = traj_points[i], traj_points[i + 1]
                seg = LineString([p1, p2])
                if seg.intersects(query_poly):
                    segment_intersects = True
                    print(f"  发现穿透线段: Pt[{i}]({p1}) -> Pt[{i + 1}]({p2})")
                    if not q_bbox.contains_point(*p1) and not q_bbox.contains_point(*p2):
                        print("  >> 判定: 线段穿透 (端点均在框外，但连线穿过查询区域)")
                    break

        if assigned_cell:
            g_alpha, g_beta = index.alpha, index.beta
            l_alpha, l_beta = assigned_cell.alpha, assigned_cell.beta
            stored_sig = assigned_cell.signatures.get(error_tid)
            recalc_sig = compute_traj_signature(g_alpha, g_beta, assigned_cell, traj_points)
            query_sig = compute_query_signature(g_alpha, g_beta, assigned_cell, q_bbox)

            print(f"Local Param: {l_alpha}x{l_beta} | Global: {g_alpha}x{g_beta}")
            print(f"Stored Sig: {bin(stored_sig) if stored_sig is not None else 'None'}")
            print(f"Recalc Sig: {bin(recalc_sig)}")
            print(f"Query  Sig: {bin(query_sig)}")
            print(f"Match (Stored): {(stored_sig & query_sig != 0) if stored_sig is not None else 'N/A'}")
            print(f"Match (Recalc): {(recalc_sig & query_sig != 0)}")

        print(f"\n[定性分析]")
        if segment_intersects and in_query_count == 0:
            print(">> 漏检确认: 点采样真空。TShape 签名仅基于点计算位图，无法捕捉穿透线段。")
        elif not segment_intersects:
            print(">> 漏检确认: 逻辑异常。线段并未与查询框相交，但暴力搜索判定为相交，请检查暴力搜索逻辑。")
        else:
            print(">> 漏检确认: 其他逻辑错误（如 EE BBox 不足、签名映射不一致等）。")

        self.trace_missing_path(index, error_tid, q_bbox)
        print("=" * 60 + "\n")

    def trace_missing_path(self, index, tid, q_bbox):
        cell = index.trajectory_to_cells.get(tid)
        path = []
        curr = cell
        while curr:
            path.append(curr)
            curr = curr.parent
        path.reverse()

        print("\n[Search Path Trace]:")
        for node in path:
            ee = node.get_enlarged_element_bbox(index.alpha, index.beta)
            intersects = q_bbox.intersects(ee)
            print(f"  L{node.level} [{node.code}] | Muted: {node.muted} | Intersects EE: {intersects}")


if __name__ == "__main__":
    unittest.main()

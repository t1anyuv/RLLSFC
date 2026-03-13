import unittest
import os

from shapely.geometry import LineString, box

from src.core.bounding_box import SpatialBoundingBox
from src.data import load_cleaned_dataset
from src.evaluation import TraversalPerformanceEvaluator
from src.indexing.quadtree_index import QuadTreeIndex
from src.indexing.traversal_encoder import TraversalOrderEncoder
from src.reward.cost_evaluator import TraversalCostEvaluator
from src.utils.signature import compute_traj_signature, compute_query_signature
from src.utils.path_manager import get_path_manager


class TestTShapeSearchCorrectness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 1. 统一全局参数
        cls.bbox = SpatialBoundingBox(115.29, 39.00, 117.83, 41.50)
        cls.max_level = 8
        cls.alpha, cls.beta = 3, 3

        # 2. 预加载 TDrive 真实数据
        print("Pre-loading TDrive data...")
        pm = get_path_manager()
        tdrive_path = pm.tdrive_data_path or os.environ.get('TDRIVE_DATA_PATH',
                                                            r'D:\Dataset\Trajectory\TDrive\complete\tdrive_cleaned'
                                                            r'.txt')
        cls.raw_trajectories = load_cleaned_dataset(
            str(tdrive_path),
            max_trajectories=None
        )

        # 3. 定义统一的测试查询框
        cls.test_queries = [
            # 1. 核心区域 (Core / Hotspot)
            SpatialBoundingBox(116.3, 39.9, 116.4, 40.0),  # 稠密：天安门/王府井周边
            SpatialBoundingBox(116.35, 39.95, 116.38, 39.98),  # 精细：局部小范围高频点

            # 2. 边缘与稀疏区域 (Edge / Sparse)
            SpatialBoundingBox(116.5, 39.5, 116.6, 39.6),  # 边缘：东南郊区
            SpatialBoundingBox(115.8, 39.3, 116.0, 39.5),  # 极稀疏：西南偏远区域

            # 3. 特殊几何形状 (Special Shapes)
            # 长条形查询：模拟沿主要干道的水平/垂直搜索，极易触发大量的 Cell 相交判断
            SpatialBoundingBox(116.0, 39.9, 116.5, 39.92),  # 水平长条（东西向长安街）
            SpatialBoundingBox(116.3, 39.8, 116.32, 40.1),  # 垂直长条（南北向中轴线）

            # 4. 边界/跨越情况 (Boundary Crossing)
            # 刚好跨越四叉树 Level 1 或 Level 2 的分割线（通常在经纬度中心点附近）
            SpatialBoundingBox(116.6, 39.9, 116.7, 40.1),  # 跨越经度大分界线

            # 5. 极端尺度 (Scale Extremes)
            # 极小范围：测试签名过滤的极高精度要求
            SpatialBoundingBox(116.391, 39.901, 116.395, 39.905),
            # 极大范围：测试剪枝后的节点收集性能
            SpatialBoundingBox(116.1, 39.7, 116.6, 40.2)
        ]

    def _init_new_index(self):
        """辅助方法：初始化一个全新的索引并分配轨迹"""
        index = QuadTreeIndex(self.bbox, self.max_level, self.alpha, self.beta)
        for traj_id, points in self.raw_trajectories:
            index.assign_trajectory(traj_id, points)
        return index

    def brute_force_search(self, index, query_bbox: SpatialBoundingBox) -> set:
        hit_ids = set()
        query_poly = box(query_bbox.min_x, query_bbox.min_y, query_bbox.max_x, query_bbox.max_y)
        for tid, points in index.trajectory_points.items():
            if len(points) < 2:
                continue
            if LineString(points).intersects(query_poly):
                hit_ids.add(tid)
        return hit_ids

    def run_search_suite(self, index, mode_name):
        """统一执行搜索测试套件"""
        print(f"\n>>> Running Search Test Mode: {mode_name}")

        encoder = TraversalOrderEncoder(index, self.alpha, self.beta)
        evaluator = TraversalPerformanceEvaluator(index, encoder, TraversalCostEvaluator(index))
        z_order = encoder.z_curve_order()

        for i, q_bbox in enumerate(self.test_queries):
            expected = self.brute_force_search(index, q_bbox)
            _, candidate_ids = evaluator.tshape_search_debug(q_bbox, z_order, -1, skip_muted=True)

            # 精筛
            actual = set()
            query_poly = box(q_bbox.min_x, q_bbox.min_y, q_bbox.max_x, q_bbox.max_y)
            for tid in candidate_ids:
                pts = index.trajectory_points.get(tid)
                if pts and LineString(pts).intersects(query_poly):
                    actual.add(tid)

            # 结果验证与深度分析
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

                    error_msg = f"[{mode_name}] Query {i} has mistakes！"
                    if true_missing:
                        error_msg += f" Missing TID: {list(true_missing)[:3]}..."
                    if extra:
                        error_msg += f" Extra TID: {list(extra)[:3]}..."
                    self.fail(error_msg)
                else:
                    print(f"  Query {i}: 忽略了 {len(missing)} 条点采样真空导致的漏检。")

        print(f"SUCCESS: {mode_name} all queries passed.")

    # --- 测试用例 ---

    def test_01_no_pruning(self):
        """情况 1：原始状态，不剪枝"""
        index = self._init_new_index()
        self.run_search_suite(index, "NO_PRUNING")

    def test_02_pruning_no_optimize(self):
        """情况 2：剪枝（轨迹上移重算签名），但不优化参数 (alpha/beta 保持 3,3)"""
        index = self._init_new_index()

        index.post_prune_tree(min_cell_trajs=4)
        index.compute_signatures(enable_optimize=False)

        self.run_search_suite(index, "PRUNING_NO_OPTIMIZE")

    def test_03_pruning_with_optimize(self):
        """情况 3：剪枝 + 自适应参数优化 (alpha/beta 动态变化)"""
        index = self._init_new_index()

        # 开启 enable_optimize
        index.post_prune_tree(min_cell_trajs=4)
        index.compute_signatures(enable_optimize=True)

        self.run_search_suite(index, "PRUNING_WITH_OPTIMIZE")

    # --- 调试辅助方法 ---
    def analyze_trajectory(self, index, actual_ids, expected_ids, q_bbox):
        """分析漏检原因"""

        missing = expected_ids - actual_ids
        if not missing:
            print("\n[INFO] 无漏检轨迹。")
            return

        error_tid = list(missing)[0]
        traj_points = index.trajectory_points.get(error_tid)
        assigned_cell = index.trajectory_to_cell.get(error_tid)
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
                        print(f"  >> 判定: 线段穿透! (端点均在框外，但连线穿过查询区域)")
                    break

        if assigned_cell:
            g_alpha, g_beta = index.alpha, index.beta
            l_alpha, l_beta = assigned_cell.alpha, assigned_cell.beta

            # 获取存储签名
            stored_sig = assigned_cell.signatures.get(error_tid)
            # 重算
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
            print(">> 漏检确认：点采样真空。TShape 签名仅基于点计算位图，由于此轨迹点都在查询框外，")
            print("   签名位图无法捕捉到穿透的线段。")
        elif not segment_intersects:
            print(">> 漏检确认：逻辑异常。线段并未与查询框相交，但暴力搜索判定为相交，请检查暴力搜索逻辑。")
        else:
            print(">> 漏检确认：其他逻辑错误（如 EE BBox 范围不足、签名映射算法不一致等）。")

        self.trace_missing_path(index, error_tid, q_bbox)
        print("=" * 60 + "\n")

    def trace_missing_path(self, index, tid, q_bbox):
        """追踪路径，显式传入 index"""
        cell = index.trajectory_to_cell.get(tid)
        path = []
        curr = cell
        while curr:
            path.append(curr)
            curr = curr.parent
        path.reverse()

        print(f"\n[Search Path Trace]:")
        for node in path:
            # 必须使用 index 自身的 alpha/beta 定位 EE
            ee = node.get_enlarged_element_bbox(index.alpha, index.beta)
            intersects = q_bbox.intersects(ee)
            print(f"  L{node.level} [{node.code}] | Muted: {node.muted} | Intersects EE: {intersects}")


if __name__ == '__main__':
    unittest.main()

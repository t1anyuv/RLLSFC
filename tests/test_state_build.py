import unittest
import os

import numpy as np

from src.data import load_cleaned_dataset
from src.features.state_feature_builder import StateFeatureBuilder
from src.indexing.quadtree_cell import SpatialBoundingBox
from src.indexing.quadtree_index import QuadTreeIndex
from src.utils.path_manager import get_path_manager


class TestStateFeatureReasonableness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 1. 初始化索引
        cls.bbox = SpatialBoundingBox(115.7019, 39.2008, 117.5987, 40.8490)
        cls.max_level = 8
        cls.alpha, cls.beta = 2, 2
        cls.index = QuadTreeIndex(cls.bbox, cls.max_level, cls.alpha, cls.beta)

        # 2. 加载真实 TDrive 数据
        print("\n[Setup] Loading TDrive data...")
        pm = get_path_manager()
        tdrive_path = pm.tdrive_data_path or os.environ.get('TDRIVE_DATA_PATH',
                                                            r'D:\Dataset\Trajectory\TDrive\complete\tdrive_cleaned'
                                                            r'.txt')
        trajectories = load_cleaned_dataset(str(tdrive_path), max_trajectories=None)
        for traj_id, points in trajectories:
            cls.index.assign_trajectory(traj_id, points)

        cls.index.post_prune_tree(min_cell_trajs=4)

        # 3. 获取目标单元格并初始化 Builder
        cls.target_cells = cls.index.get_active_cells()
        cls.builder = StateFeatureBuilder(
            quadtree=cls.index,
            alpha=cls.alpha,
            beta=cls.beta,
            target_cells=cls.target_cells
        )

        cls.static_feature_names = [
            "N_ee", "N_ee / N_int", "N_cov / N_int", "d_center",
            "Δx", "Δy", "Δdx", "Δdy",
            "S", "qcode"
        ]

    def test_feature_statistics_report(self):
        """统计所有特征的分布情况，检查是否存在异常值或梯度消失。"""
        all_vecs = np.array(list(self.builder.feature_vectors.values()))

        print("\n" + "=" * 60)
        print("       TShape 静态特征合理性报告")
        print("=" * 60)
        print(f"活跃单元格总数: {len(self.target_cells)}")

        for i, name in enumerate(self.static_feature_names):
            col = all_vecs[:, i]
            print(f"{name:25s} | Mean: {np.mean(col):.4f} | Std: {np.std(col):.4f} | Max: {np.max(col):.4f}")

    def test_density_correlation(self):
        """核心验证：验证 N_ee_log 特征是否真正反映了轨迹密度。"""
        # 挑选轨迹数量最多的前 5% 的单元格
        sorted_cells = sorted(self.target_cells, key=lambda c: len(c.trajectories), reverse=True)
        top_k = max(1, len(sorted_cells) // 20)

        top_cells = sorted_cells[:top_k]
        bottom_cells = sorted_cells[-top_k:]

        top_features = np.mean([self.builder.feature_vectors[c][0] for c in top_cells])
        bottom_features = np.mean([self.builder.feature_vectors[c][0] for c in bottom_cells])

        print(f"\n[验证] 高密度 Cell 平均 N_ee_log: {top_features:.4f}")
        print(f"[验证] 低密度 Cell 平均 N_ee_log: {bottom_features:.4f}")

        # 断言：高密度 Cell 的特征值必须显著大于低密度 Cell
        self.assertGreater(top_features, bottom_features, "特征向量未能区分轨迹密度差异！")

    def test_dynamic_feature_shift(self):
        """验证 select_action 产生的位移特征 (Δx_rel) 是否符合物理位移。"""
        if len(self.target_cells) < 2:
            return

        cell_start = self.target_cells[0]
        cell_end = self.target_cells[1]

        # 模拟从 A 移动到 B
        feat = self.builder.get_features(cell_end, prev_cell=cell_start, visited_ratio=0.5, visited_cells=set())

        # 获取 Δx_rel (index 12), Δy_rel (index 13)
        dx_rel, dy_rel = feat[12], feat[13]

        # 手动计算
        c1x, c1y = cell_end.get_center()
        c2x, c2y = cell_start.get_center()
        expected_dx = (c1x - c2x) / (self.bbox.max_x - self.bbox.min_x)

        self.assertAlmostEqual(dx_rel, expected_dx, places=5)
        print(f"\n[验证] 动态位移特征 Δx_rel 正确: {dx_rel:.4f}")

    def test_neighborhood_heat_logic(self):
        """测试邻域热度特征。"""
        if len(self.target_cells) < 1: return

        # 找一个已知有轨迹的 Cell 作为中心
        cell = max(self.target_cells, key=lambda c: len(c.trajectories))

        # 打印该 Cell 的坐标和邻居情况
        raw_neighbors = self.builder.quadtree.get_eight_neighbor_cells(cell)
        valid_neighbors = [nb for nb in raw_neighbors if nb is not None]

        print(f"\n[DEBUG] 中心 Cell 轨迹数: {len(cell.trajectories)}")
        print(f"[DEBUG] 找到有效邻居数量: {len(valid_neighbors)}")

        # 检查邻居在 ee_counts 中的数据
        for i, nb in enumerate(valid_neighbors):
            count = self.builder.ee_counts.get(nb, 0)
            print(f"  - 邻居 {i} (Active={not nb.muted}): ee_count = {count}")

        # 模拟场景: 所有 Cell 都未访问
        feat_all_unvisited = self.builder.get_features(cell, None, 0.0, set())
        heat_full = feat_all_unvisited[13]

        # 模拟场景: 所有邻居都已访问
        feat_none_unvisited = self.builder.get_features(cell, None, 0.0, set(self.target_cells))
        heat_empty = feat_none_unvisited[13]

        print(f"[验证] 最终计算 -> heat_full: {heat_full:.4f}, heat_empty: {heat_empty:.4f}")

        self.assertEqual(heat_empty, 0.0)
        if any(self.builder.ee_counts.get(nb, 0) > 0 for nb in valid_neighbors):
            self.assertGreater(heat_full, 0.0, "邻居明明有轨迹，但 heat_full 计算为 0！")

    def test_feature_detailed_diagnostic(self):
        """详细特征诊断"""
        all_vecs = np.array(list(self.builder.feature_vectors.values()))
        print(all_vecs.shape)

        print("\n" + "=" * 95)
        print(f"{'Feature Name':<20} | {'Mean':<8} | {'Min':<8} | {'50%(Med)':<8} | {'Max':<8} | {'Zeros%'}")
        print("-" * 95)

        for i, name in enumerate(self.static_feature_names):
            col = all_vecs[:, i]
            sparsity = (np.abs(col) < 1e-7).mean() * 100
            print(
                f"{name:<20} | {np.mean(col):>8.4f} | "
                f"{np.min(col):>8.4f} | {np.percentile(col, 50):>8.4f} | "
                f"{np.max(col):>8.4f} | {sparsity:>5.1f}%")

        # 验证 Log_size 是否成功归一化到 [0, 1]
        size_col = all_vecs[:, 8]
        self.assertTrue(np.all(size_col >= -1e-7) and np.all(size_col <= 1.0 + 1e-7), "Log_size 归一化失败！")

        # 验证 Z-Score 后的均值是否接近 0
        self.assertAlmostEqual(np.mean(all_vecs[:, 6]), 0.0, places=2, msg="Δdx_den Z-Score 中心化失败")


if __name__ == "__main__":
    unittest.main()

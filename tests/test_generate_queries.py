import unittest
from typing import Tuple

import numpy as np

from src.core.bounding_box import SpatialBoundingBox
from src.indexing import QuadTreeIndex
from src.queries.hotspot_generator import _collect_trajectory_centres, generate_hotspots_from_distribution, \
    generate_uniform_queries


class TestQueryGenerationWithEE(unittest.TestCase):
    def setUp(self):
        """
        初始化测试环境。
        注意：QuadTreeIndex 初始化时会调用 _build_tree，
        all_cells 的键格式为 (level, path_to_quadrant...)
        """
        self.bbox = SpatialBoundingBox(0.0, 0.0, 100.0, 100.0)
        self.max_level = 3
        # 使用你定义的参数初始化
        self.index = QuadTreeIndex(bbox=self.bbox, max_level=self.max_level, alpha=2, beta=2)

        # 构造模拟轨迹数据，分布在右上角热点 (85, 85)
        self._generate_mock_trajectories(num=40, center=(85.0, 85.0), std=5.0)

    def _generate_mock_trajectories(self, num: int, center: Tuple[float, float], std: float):
        """
        调用 index.assign_trajectory 来注入数据，
        这样可以确保 trajectory_mbrs 和 trajectory_to_cell 正确填充。
        """
        for i in range(num):
            # 生成一小段随机轨迹点
            base_x = np.clip(np.random.normal(center[0], std), 0.1, 99.9)
            base_y = np.clip(np.random.normal(center[1], std), 0.1, 99.9)

            # 轨迹包含两个点，形成一个小的 MBR
            points = [
                (base_x, base_y),
                (base_x + 0.5, base_y + 0.5)
            ]

            self.index.assign_trajectory(traj_id=i, points=points)

    def test_collect_centres_logic(self):
        """验证 _collect_trajectory_centres 是否能正确从 all_cells 中提取轨迹中心"""

        centres = _collect_trajectory_centres(self.index)

        # 验证提取的数量是否正确（去重后的轨迹数）
        self.assertEqual(len(centres), 40)

        # 验证热点位置：大部分中心点应该在 (80, 80) 以上
        avg_x = sum(c[0] for c in centres) / len(centres)
        avg_y = sum(c[1] for c in centres) / len(centres)
        self.assertGreater(avg_x, 70.0)
        self.assertGreater(avg_y, 70.0)

    def test_hotspots_from_distribution_with_quadtree(self):
        """验证基于轨迹分布生成的查询是否覆盖了高密度区"""

        num_queries = 10

        queries = generate_hotspots_from_distribution(
            self.index, num_queries=num_queries, grid_size=8, top_k=1
        )

        self.assertEqual(len(queries), num_queries)

        # 统计落在右上角象限的查询数量
        hits = 0
        for q in queries:
            q_center_x = (q.min_x + q.max_x) / 2
            q_center_y = (q.min_y + q.max_y) / 2
            if q_center_x > 50.0 and q_center_y > 50.0:
                hits += 1

        # 热点查询应至少有大部分落在热点区域
        self.assertGreaterEqual(hits, int(num_queries * 0.8))

    def test_uniform_queries_respect_bbox(self):
        """测试均匀查询是否严格遵守 QuadTree 的 bbox 限制"""

        num_queries = 50
        queries = generate_uniform_queries(self.index, num_queries=num_queries)

        for q in queries:
            # 验证不越界
            self.assertTrue(q.min_x >= self.index.bbox.min_x)
            self.assertTrue(q.max_x <= self.index.bbox.max_x)
            self.assertTrue(q.min_y >= self.index.bbox.min_y)
            self.assertTrue(q.max_y <= self.index.bbox.max_y)

    def test_empty_quadtree_fallback(self):
        """测试当四叉树中没有轨迹时，热点生成是否回退到随机模式而不崩溃"""

        empty_index = QuadTreeIndex(bbox=self.bbox, max_level=2)
        # 不调用 assign_trajectory

        num_queries = 5
        # 应该触发 _collect_trajectory_centres 返回空列表，进而调用 generate_hotspot_queries
        queries = generate_hotspots_from_distribution(empty_index, num_queries=num_queries)

        self.assertEqual(len(queries), num_queries)
        self.assertIsInstance(queries[0], SpatialBoundingBox)


if __name__ == '__main__':
    unittest.main()

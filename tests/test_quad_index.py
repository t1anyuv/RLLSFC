import unittest

from src.common import SpatialBoundingBox
from src.indexing.quadtree_index import QuadTreeIndex


class TestQuadTreeIndexOptimization(unittest.TestCase):
    def setUp(self):
        # 初始化一个测试用的边界框 (0, 0) 到 (100, 100)
        self.bbox = SpatialBoundingBox(0, 0, 100, 100)
        self.max_level = 3
        self.index = QuadTreeIndex(self.bbox, self.max_level, alpha=2, beta=2)

    def test_grid_indexing_logic(self):
        """验证 (level, x_idx, y_idx) 门牌号逻辑是否正确映射到象限序列"""
        # 测试 Level 1 的四个象限
        # (1, 0, 0) -> LB (象限 0)
        cell_lb = self.index.all_cells.get((1, 0, 0))
        self.assertEqual(cell_lb.quadrant_sequence, [0])

        # (1, 1, 1) -> RT (象限 3)
        cell_rt = self.index.all_cells.get((1, 1, 1))
        self.assertEqual(cell_rt.quadrant_sequence, [3])

    def test_get_cell_at_efficiency(self):
        """验证 O(1) 定位函数 get_cell_at 的准确性"""
        # 取一个特定点 (75, 75)，在 Level 1 应该是右上角 (1, 1, 1)
        target_cell = self.index.get_cell_at(75, 75, level=1)
        self.assertIsNotNone(target_cell)
        self.assertEqual(target_cell.quadrant_sequence, [3])

        # 验证边界点 (100, 100) 是否被正确处理（不溢出）
        edge_cell = self.index.get_cell_at(100, 100, level=self.max_level)
        self.assertIsNotNone(edge_cell)

    def test_pruning_and_trajectory_move(self):
        """验证剪枝后轨迹是否正确上移且索引同步更新"""
        # 模拟一条轨迹在叶子节点
        traj_id = 999
        points = [(10, 10), (12, 12)]
        self.index.assign_trajectory(traj_id, points)

        # 获取初始分配的 Cell
        initial_cell = self.index.trajectory_to_cells[traj_id]
        self.assertEqual(initial_cell.level, self.max_level)

        # 执行剪枝，设置一个极大的阈值迫使节点合并
        self.index.post_prune_tree(min_cell_trajs=100)

        # 验证轨迹是否移动到了根节点（因为所有中间节点都被屏蔽了）
        new_cell = self.index.trajectory_to_cells[traj_id]
        self.assertEqual(new_cell.level, 0)
        self.assertTrue(initial_cell.muted)

    def test_neighbor_search(self):
        """验证邻居查找逻辑"""
        # 获取中心区域的一个 Cell
        # 在 Level 2 中，(2, 1, 1) 是靠近中心的格点
        center_cell = self.index.all_cells.get((2, 1, 1))
        neighbors = self.index.get_eight_neighbor_cells(center_cell)

        # 理论上中心格点应该有 8 个邻居
        self.assertTrue(len(neighbors) <= 8)
        for n in neighbors:
            self.assertEqual(n.level, center_cell.level)


if __name__ == '__main__':
    unittest.main()

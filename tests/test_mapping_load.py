import unittest
from pathlib import Path

from src.core.bounding_box import SpatialBoundingBox
from src.indexing.lsfc_loader import LSFCMappingLoader
from src.indexing.quadtree_cell import QuadTreeCell


class TestOrderMappingWithRealFiles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """定义真实文件路径"""
        cls.json_path = Path(r"C:\Users\t1anyu\Desktop\Projects\LearnedTShape\src\resource\orders\learned_order.json")

        # 验证文件是否存在
        if not cls.json_path.exists():
            raise FileNotFoundError(
                f"测试需要真实文件：\n{cls.json_path}\n"
                "请确保文件路径正确。"
            )

    def setUp(self):
        self.loader = LSFCMappingLoader()

    def test_json_loading_real_data(self):
        """测试加载真实 JSON 并验证元数据"""
        self.loader.load_from_json(self.json_path)
        self.assertTrue(self.loader.is_loaded())

        stats = self.loader.get_statistics()
        print(f"\n[JSON 统计] 映射项总数: {stats['count']}")

        # 验证 max_level 是否被正确识别
        self.assertIsNotNone(self.loader._max_level)
        self.assertGreater(self.loader._max_level, 0)

    def test_cell_object_mapping_real(self):
        """
        核心测试：验证真实的 QuadTree 节点能否通过 Loader 找到 Order。
        这要求 QuadTreeCell 的 quadrant_sequence 生成逻辑与 quad_code 编码逻辑完全匹配。
        """
        self.loader.load_from_json(self.json_path)

        # 从加载的映射中取一个真实的 QuadCode 进行逆向测试
        target_qc = list(self.loader._quad_code_to_order.keys())[20261]
        expected_order = self.loader._quad_code_to_order[target_qc]

        # 解码得到 level 和 seq
        level, seq = self.loader._decode_quad_code(target_qc)

        # 创建一个模拟的 Cell 对象
        mock_cell = QuadTreeCell(
            bbox=SpatialBoundingBox(0, 0, 1, 1),  # 几何信息不影响 get_order
            level=level,
            quadrant_sequence=seq
        )

        # 验证能否查到
        actual_order = self.loader.get_order(mock_cell)
        self.assertEqual(actual_order, expected_order)
        print(f"[验证通过] QuadCode {target_qc} -> Level {level}, Seq {seq} -> Order {actual_order}")


if __name__ == '__main__':
    unittest.main()

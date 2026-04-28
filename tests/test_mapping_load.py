import unittest
from pathlib import Path

from src.common import SpatialBoundingBox
from src.indexing.lsfc_loader import LSFCMappingLoader
from src.indexing.quadtree_cell import QuadTreeCell


class TestOrderMappingWithRealFiles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        repo_root = Path(__file__).resolve().parents[1]
        candidates = sorted(repo_root.glob('resource/**/*quadorder*.json'))
        if not candidates:
            raise unittest.SkipTest('No exported quadorder JSON found under resource/')
        cls.json_path = candidates[0]

    def setUp(self):
        self.loader = LSFCMappingLoader()

    def test_json_loading_real_data(self):
        self.loader.load_from_json(self.json_path)
        self.assertTrue(self.loader.is_loaded())

        stats = self.loader.get_statistics()
        print(f"\n[JSON Stats] Mapping count: {stats['count']}")

        self.assertIsNotNone(self.loader._max_level)
        self.assertGreater(self.loader._max_level, 0)

    def test_cell_object_mapping_real(self):
        self.loader.load_from_json(self.json_path)

        target_qc = list(self.loader._quad_code_to_order.keys())[0]
        expected_order = self.loader._quad_code_to_order[target_qc]
        level, seq = self.loader._decode_quad_code(target_qc)

        mock_cell = QuadTreeCell(
            bbox=SpatialBoundingBox(0, 0, 1, 1),
            level=level,
            quadrant_sequence=seq
        )

        actual_order = self.loader.get_order(mock_cell)
        self.assertEqual(actual_order, expected_order)
        print(f"[Verified] QuadCode {target_qc} -> Level {level}, Seq {seq} -> Order {actual_order}")


if __name__ == '__main__':
    unittest.main()

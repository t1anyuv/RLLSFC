"""测试四叉树编码/解码一致性"""
import pytest

from src.common import SpatialBoundingBox
from src.indexing.lsfc_loader import LSFCMappingLoader
from src.indexing.quadtree_cell import QuadTreeCell


class TestQuadEncoding:
    """测试四叉树编码和解码的一致性"""

    @pytest.fixture
    def max_level(self):
        """测试用的最大层级"""
        return 8

    @pytest.fixture
    def loader(self, max_level):
        """创建加载器实例"""
        loader = LSFCMappingLoader()
        loader._max_level = max_level
        return loader

    @pytest.fixture
    def root_bbox(self):
        """根节点边界框"""
        return SpatialBoundingBox(min_x=0.0, min_y=0.0, max_x=100.0, max_y=100.0)

    def test_root_node(self, loader, max_level, root_bbox):
        """测试根节点编码/解码"""
        cell = QuadTreeCell(bbox=root_bbox, level=0, quadrant_sequence=[])
        code = cell.get_quadrant_code(max_level)

        assert code == 0, "根节点编码应为 0"

        level, decoded_seq = loader._decode_quad_code(code)
        assert level == 0, "根节点层级应为 0"
        assert decoded_seq == [], "根节点象限序列应为空"

    def test_level1_nodes(self, loader, max_level, root_bbox):
        """测试第一层所有象限的编码/解码"""
        for quadrant in range(4):
            seq = [quadrant]
            cell = QuadTreeCell(bbox=root_bbox, level=1, quadrant_sequence=seq)
            code = cell.get_quadrant_code(max_level)

            # 解码
            level, decoded_seq = loader._decode_quad_code(code)

            # 验证
            assert level == 1, f"象限 {quadrant} 层级应为 1"
            assert decoded_seq == seq, f"象限 {quadrant} 解码序列不匹配"

            # 验证往返编码
            re_cell = QuadTreeCell(bbox=root_bbox, level=level, quadrant_sequence=decoded_seq)
            re_code = re_cell.get_quadrant_code(max_level)
            assert re_code == code, f"象限 {quadrant} 往返编码不匹配"

    def test_level2_nodes(self, loader, max_level, root_bbox):
        """测试第二层节点的编码/解码"""
        test_sequences = [
            [0, 0], [0, 1], [0, 2], [0, 3],
            [1, 0], [1, 2], [2, 1], [3, 3]
        ]

        for seq in test_sequences:
            cell = QuadTreeCell(bbox=root_bbox, level=2, quadrant_sequence=seq)
            code = cell.get_quadrant_code(max_level)

            # 解码
            level, decoded_seq = loader._decode_quad_code(code)

            # 验证
            assert level == 2, f"序列 {seq} 层级应为 2"
            assert decoded_seq == seq, f"序列 {seq} 解码不匹配"

            # 验证往返编码
            re_cell = QuadTreeCell(bbox=root_bbox, level=level, quadrant_sequence=decoded_seq)
            re_code = re_cell.get_quadrant_code(max_level)
            assert re_code == code, f"序列 {seq} 往返编码不匹配"

    def test_level3_nodes(self, loader, max_level, root_bbox):
        """测试第三层节点的编码/解码"""
        test_sequences = [
            [0, 0, 0], [0, 1, 2], [1, 2, 3], [3, 3, 3]
        ]

        for seq in test_sequences:
            cell = QuadTreeCell(bbox=root_bbox, level=3, quadrant_sequence=seq)
            code = cell.get_quadrant_code(max_level)

            # 解码
            level, decoded_seq = loader._decode_quad_code(code)

            # 验证
            assert level == 3, f"序列 {seq} 层级应为 3"
            assert decoded_seq == seq, f"序列 {seq} 解码不匹配"

            # 验证往返编码
            re_cell = QuadTreeCell(bbox=root_bbox, level=level, quadrant_sequence=decoded_seq)
            re_code = re_cell.get_quadrant_code(max_level)
            assert re_code == code, f"序列 {seq} 往返编码不匹配"

    def test_deep_level_nodes(self, loader, max_level, root_bbox):
        """测试深层节点的编码/解码"""
        # 测试 level 4-8 的一些节点
        test_sequences = [
            [0, 0, 0, 0],
            [1, 2, 3, 0, 1],
            [2, 2, 2, 2, 2, 2],
            [3, 3, 3, 3, 3, 3, 3],
            [0, 1, 2, 3, 0, 1, 2, 3]
        ]

        for seq in test_sequences:
            level = len(seq)
            if level > max_level:
                continue

            cell = QuadTreeCell(bbox=root_bbox, level=level, quadrant_sequence=seq)
            code = cell.get_quadrant_code(max_level)

            # 解码
            decoded_level, decoded_seq = loader._decode_quad_code(code)

            # 验证
            assert decoded_level == level, f"序列 {seq} 层级不匹配"
            assert decoded_seq == seq, f"序列 {seq} 解码不匹配"

            # 验证往返编码
            re_cell = QuadTreeCell(bbox=root_bbox, level=decoded_level, quadrant_sequence=decoded_seq)
            re_code = re_cell.get_quadrant_code(max_level)
            assert re_code == code, f"序列 {seq} 往返编码不匹配"

    def test_actual_data_codes(self, loader, max_level, root_bbox):
        """测试实际数据中的编码"""
        # 从 final.json 中提取的实际编码
        actual_codes = [0, 1, 5463, 5464, 5465, 5466, 5467, 5468]

        for code in actual_codes:
            # 解码
            level, decoded_seq = loader._decode_quad_code(code)

            # 验证往返编码
            cell = QuadTreeCell(bbox=root_bbox, level=level, quadrant_sequence=decoded_seq)
            re_code = cell.get_quadrant_code(max_level)

            assert re_code == code, f"实际编码 {code} 往返不匹配，解码为 {decoded_seq}"

    def test_invalid_code(self, loader):
        """测试无效编码"""
        # 测试一个不可能的编码值
        invalid_code = 999999999

        with pytest.raises(ValueError, match="无法解码"):
            loader._decode_quad_code(invalid_code)

    def test_uninitialized_max_level(self):
        """测试未初始化 max_level 的情况"""
        loader = LSFCMappingLoader()

        with pytest.raises(RuntimeError, match="max_level 未初始化"):
            loader._decode_quad_code(100)

    def test_encoding_formula_consistency(self, max_level, root_bbox):
        """测试编码公式的一致性"""
        # 验证编码公式：code = sum(quadrant[i] * ((4^(max_level - i + 1) - 1) // 3) + 1)

        # Level 1, quadrant 0
        seq = [0]
        cell = QuadTreeCell(bbox=root_bbox, level=1, quadrant_sequence=seq)
        code = cell.get_quadrant_code(max_level)
        expected = 0 * ((4 ** max_level - 1) // 3) + 1
        assert code == expected, f"Level 1 quadrant 0 编码不符合公式"

        # Level 1, quadrant 1
        seq = [1]
        cell = QuadTreeCell(bbox=root_bbox, level=1, quadrant_sequence=seq)
        code = cell.get_quadrant_code(max_level)
        expected = 1 * ((4 ** max_level - 1) // 3) + 1
        assert code == expected, f"Level 1 quadrant 1 编码不符合公式"

        # Level 2, quadrant [0, 1]
        seq = [0, 1]
        cell = QuadTreeCell(bbox=root_bbox, level=2, quadrant_sequence=seq)
        code = cell.get_quadrant_code(max_level)
        expected = (0 * ((4 ** max_level - 1) // 3) + 1) + (1 * ((4 ** (max_level - 1) - 1) // 3) + 1)
        assert code == expected, f"Level 2 quadrant [0, 1] 编码不符合公式"

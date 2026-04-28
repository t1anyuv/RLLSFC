"""四叉树单元格定义。"""
from typing import Dict, List, Optional, Tuple

from src.common import SpatialBoundingBox


class QuadTreeCell:
    """表示四叉树中的单个空间单元格。"""

    def __init__(self, bbox: SpatialBoundingBox, level: int,
                 quadrant_sequence: Optional[List[int]] = None,
                 code: int = 0, alpha: int = 2, beta: int = 2):
        self.bbox = bbox
        self.level = level
        self.code: int = code
        self.alpha = alpha
        self.beta = beta
        self.quadrant_sequence = quadrant_sequence or []
        self.children: List[Optional["QuadTreeCell"]] = [None, None, None, None]
        self.parent: Optional["QuadTreeCell"] = None
        self.trajectories: set = set()
        self.signatures: Dict[int, int] = {}
        self.trajectory_mbrs: Dict[int, SpatialBoundingBox] = {}
        self.muted: bool = False

    def get_center(self) -> Tuple[float, float]:
        """返回单元格几何中心。"""
        return self.bbox.get_center()

    def contains_point(self, x: float, y: float) -> bool:
        """判断点是否落在当前单元格内。"""
        return self.bbox.contains_point(x, y)

    def get_quadrant_code(self, max_level: int) -> int:
        """按照 TShape 编码规则计算当前单元格的 element code。"""
        if not self.quadrant_sequence:
            return 0

        code = 0
        for i, quadrant in enumerate(self.quadrant_sequence, 1):
            term = quadrant * ((4 ** (max_level - i + 1) - 1) // 3) + 1
            code += term
        return code

    @staticmethod
    def encode_full_quadrant_path(quadrants: List[int], max_level: int) -> int:
        """长度为 max_level 的象限路径对应的 TShape quadCode（叶层编码）。"""
        if len(quadrants) != max_level:
            raise ValueError(f"路径长度 {len(quadrants)} 与 max_level {max_level} 不一致")
        code = 0
        for i, quadrant in enumerate(quadrants, 1):
            term = quadrant * ((4 ** (max_level - i + 1) - 1) // 3) + 1
            code += term
        return code

    @staticmethod
    def subtree_leaf_quad_codes(prefix: List[int], max_level: int) -> Tuple[int, int]:
        """(quad.elementCode << mb, ((quad.elementCode + IS(level)) << mb) - 1)"""
        if len(prefix) > max_level:
            raise ValueError("prefix 长度超过 max_level")
        pad = max_level - len(prefix)
        lo = QuadTreeCell.encode_full_quadrant_path(list(prefix) + [0] * pad, max_level)
        hi_inclusive = QuadTreeCell.encode_full_quadrant_path(list(prefix) + [3] * pad, max_level)
        return lo, hi_inclusive + 1

    def get_enlarged_element_bbox(self, alpha: int, beta: int) -> SpatialBoundingBox:
        """返回按 alpha×beta 放大的元素边界框。"""

        width = self.bbox.max_x - self.bbox.min_x
        height = self.bbox.max_y - self.bbox.min_y

        return SpatialBoundingBox(
            min_x=self.bbox.min_x,
            min_y=self.bbox.min_y,
            max_x=self.bbox.min_x + alpha * width,
            max_y=self.bbox.min_y + beta * height,
        )

    @staticmethod
    def sequence_to_bbox(
            quadrant_sequence: List[int],
            root_bbox: SpatialBoundingBox
    ) -> SpatialBoundingBox:
        """从quadrant_sequence和层级计算原始边界框"""
        xmin = root_bbox.min_x
        ymin = root_bbox.min_y
        xmax = root_bbox.max_x
        ymax = root_bbox.max_y

        for quadrant in quadrant_sequence:
            x_center = (xmin + xmax) / 2.0
            y_center = (ymin + ymax) / 2.0

            if quadrant == 0:  # 左下
                xmax = x_center
                ymax = y_center
            elif quadrant == 1:  # 右下
                xmin = x_center
                ymax = y_center
            elif quadrant == 2:  # 左上
                xmax = x_center
                ymin = y_center
            elif quadrant == 3:  # 右上
                xmin = x_center
                ymin = y_center
            else:
                raise ValueError(f"Invalid quadrant: {quadrant}, must be 0-3")

        return SpatialBoundingBox(min_x=xmin, min_y=ymin, max_x=xmax, max_y=ymax)

    def get_original_bbox(self, root_bbox: SpatialBoundingBox) -> SpatialBoundingBox:
        """从当前单元格的quadrant_sequence计算原始边界框"""
        return self.sequence_to_bbox(self.quadrant_sequence, root_bbox)

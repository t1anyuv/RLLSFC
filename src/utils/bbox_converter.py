"""边界框转换工具：将归一化的quadrant_sequence转换为原始坐标边界框。"""

from typing import List, Tuple

from src.core.bounding_box import SpatialBoundingBox
from src.indexing.quadtree_cell import QuadTreeCell


def convert_sequences_to_original_bboxes(
    sequences: List[List[int]],
    levels: List[int],
    root_bbox: SpatialBoundingBox
) -> List[SpatialBoundingBox]:
    """
    批量将quadrant_sequence转换为原始边界框。
    
    参数:
        sequences: quadrant_sequence列表
        levels: 对应的层级列表
        root_bbox: 根节点的原始边界框
    
    返回:
        原始边界框列表
    """
    return [
        QuadTreeCell.sequence_to_bbox(seq, root_bbox)
        for seq, level in zip(sequences, levels)
    ]


def get_bbox_center(bbox: SpatialBoundingBox) -> Tuple[float, float]:
    """获取边界框的中心点坐标。"""
    return bbox.get_center()


def format_bbox_string(bbox: SpatialBoundingBox, precision: int = 4) -> str:
    """
    将边界框格式化为字符串，格式：min_x min_y max_x max_y
    
    参数:
        bbox: 边界框
        precision: 小数精度
    
    返回:
        格式化的字符串，例如: "115.7019 39.2008 117.5987 40.8490"
    """
    return (f"{bbox.min_x:.{precision}f} {bbox.min_y:.{precision}f} "
            f"{bbox.max_x:.{precision}f} {bbox.max_y:.{precision}f}")


def parse_bbox_string(bbox_str: str) -> SpatialBoundingBox:
    """
    从字符串解析边界框，格式：min_x min_y max_x max_y
    
    参数:
        bbox_str: 边界框字符串，例如: "115.7019 39.2008 117.5987 40.8490"
    
    返回:
        SpatialBoundingBox对象
    """
    parts = bbox_str.strip().split()
    if len(parts) != 4:
        raise ValueError(f"Invalid bbox string format: {bbox_str}, expected 4 values")
    
    return SpatialBoundingBox(
        min_x=float(parts[0]),
        min_y=float(parts[1]),
        max_x=float(parts[2]),
        max_y=float(parts[3])
    )

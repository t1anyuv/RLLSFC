from typing import List, Tuple

import numpy as np

from src.core.bounding_box import SpatialBoundingBox
from src.indexing.quadtree_cell import QuadTreeCell

eps = 1e-12


def compute_signature_vectorized(local_alpha: int, local_beta: int,
                                 points: np.ndarray, ee_bbox: SpatialBoundingBox) -> int:
    """使用向量化运算快速计算轨迹签名"""
    if len(points) == 0:
        return 0

    ee_w = ee_bbox.max_x - ee_bbox.min_x
    ee_h = ee_bbox.max_y - ee_bbox.min_y

    grid_w = ee_w / (local_alpha + eps)
    grid_h = ee_h / (local_beta + eps)

    # 映射到 [0, local_alpha) 和 [0, local_beta) 的整数索引并裁剪边界情况
    x_idxs = np.floor((points[:, 0] - ee_bbox.min_x + eps) / grid_w).astype(int)
    y_idxs = np.floor((points[:, 1] - ee_bbox.min_y + eps) / grid_h).astype(int)

    x_idxs = np.clip(x_idxs, 0, local_alpha - 1)
    y_idxs = np.clip(y_idxs, 0, local_beta - 1)

    # 计算位偏移并合并去重
    bit_offsets = x_idxs * local_beta + y_idxs
    unique_offsets = np.unique(bit_offsets)

    signature = 0
    for offset in unique_offsets:
        signature |= (1 << int(offset))
    return signature


def compute_traj_signature(global_alpha: int, global_beta: int,
                           cell: QuadTreeCell, points: List[Tuple[float, float]]) -> int:
    """基于 NumPy 的快速签名计算：使用坐标映射替代几何相交"""
    if not points:
        return 0

    # 1. 将轨迹点转换为 numpy 矩阵
    pts = np.array(points)

    # 2. 获取扩大元素
    ee_bbox = cell.get_enlarged_element_bbox(global_alpha, global_beta)

    # 3. 计算签名
    return compute_signature_vectorized(cell.alpha, cell.beta, pts, ee_bbox)


def compute_query_signature(global_alpha: int, global_beta: int,
                            cell: QuadTreeCell, query_bbox: SpatialBoundingBox) -> int:
    ee_bbox = cell.get_enlarged_element_bbox(global_alpha, global_beta)

    ee_w = ee_bbox.max_x - ee_bbox.min_x
    ee_h = ee_bbox.max_y - ee_bbox.min_y

    grid_w = ee_w / (cell.alpha + eps)
    grid_h = ee_h / (cell.beta + eps)

    # 计算查询框覆盖的网格索引范围
    i_start = int(np.floor((query_bbox.min_x - ee_bbox.min_x + eps) / grid_w))
    i_end = int(np.floor((query_bbox.max_x - ee_bbox.min_x - eps) / grid_w))

    j_start = int(np.floor((query_bbox.min_y - ee_bbox.min_y + eps) / grid_h))
    j_end = int(np.floor((query_bbox.max_y - ee_bbox.min_y - eps) / grid_h))

    i_start, i_end = max(0, i_start), min(cell.alpha - 1, i_end)
    j_start, j_end = max(0, j_start), min(cell.beta - 1, j_end)

    signature = 0
    for i in range(i_start, i_end + 1):
        for j in range(j_start, j_end + 1):
            signature |= (1 << (i * cell.beta + j))

    return signature

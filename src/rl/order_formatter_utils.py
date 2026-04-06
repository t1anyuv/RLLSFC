"""轨迹顺序格式化工具函数。"""
from typing import Any, Dict, List, Tuple, Optional

from src.indexing.quadtree_cell import QuadTreeCell
from src.indexing.quadtree_index import QuadTreeIndex


def get_unique_active_nodes(order: List[QuadTreeCell], quadtree: QuadTreeIndex) -> List[QuadTreeCell]:
    """去重并提取活跃节点。
    
    参数:
        order: 原始访问顺序
        quadtree: 四叉树索引
        
    返回:
        去重后的活跃节点列表
    """
    seen_codes = set()
    active_nodes = []

    # 始终确保 Root 存在（如果未屏蔽）
    root = quadtree.root
    if not root.muted:
        active_nodes.append(root)
        seen_codes.add(root.get_quadrant_code(quadtree.max_level))

    for cell in order:
        if not cell.muted:
            qc = cell.get_quadrant_code(quadtree.max_level)
            if qc not in seen_codes:
                active_nodes.append(cell)
                seen_codes.add(qc)
    
    return active_nodes


def build_active_map(active_nodes: List[QuadTreeCell], max_level: int) -> Tuple[Dict, Dict]:
    """建立 Cell 对象到配置容器的映射。
    
    参数:
        active_nodes: 活跃节点列表
        max_level: 四叉树最大层级
        
    返回:
        (active_map, qc_to_info) 元组
        - active_map: Cell -> Info 映射
        - qc_to_info: QC -> Info 映射
    """
    active_map = {}  # Cell -> Info
    qc_to_info = {}  # QC -> Info (用于哑节点通过QC查找)

    for idx, cell in enumerate(active_nodes):
        qc = cell.get_quadrant_code(max_level)
        info = {
            "order": idx,
            "cell": cell,
            "active_qc": qc,
            "muted_set": set()
        }
        active_map[cell] = info
        qc_to_info[qc] = info
    
    return active_map, qc_to_info


def build_parent_descriptor(cell: QuadTreeCell, max_level: int) -> Dict[str, Any]:
    """构建父节点几何描述。
    
    参数:
        cell: 单元格对象
        max_level: 四叉树最大层级
        
    返回:
        包含父节点几何信息的字典
    """
    return {
        "alpha": int(cell.alpha),
        "beta": int(cell.beta),
        "level": int(cell.level),
        "elementCode": int(cell.get_quadrant_code(max_level)),
        "xmin": float(cell.bbox.min_x),
        "ymin": float(cell.bbox.min_y),
        "xmax": float(cell.bbox.max_x),
        "ymax": float(cell.bbox.max_y)
    }


def find_active_parent(cell: QuadTreeCell) -> Optional[QuadTreeCell]:
    """向上追溯直到找到非 muted 的节点。
    
    参数:
        cell: 起始单元格
        
    返回:
        第一个非 muted 的祖先节点，如果不存在则返回 None
    """
    curr = cell.parent
    while curr is not None:
        if not curr.muted:
            return curr
        curr = curr.parent
    return None


def assemble_ordering(qc_to_info: Dict, max_level: int) -> List[Dict]:
    """构造最终导出结构。
    
    参数:
        qc_to_info: QC 到 Info 的映射
        max_level: 四叉树最大层级
        
    返回:
        排序后的配置列表
    """
    # 按 order 升序排列
    sorted_infos = sorted(qc_to_info.values(), key=lambda x: x["order"])
    ordering = []

    for info in sorted_infos:
        cell = info["cell"]
        combined_codes = [info["active_qc"]] + sorted(list(info["muted_set"]))

        ordering.append({
            "quad_code": combined_codes,
            "order": info["order"],
            "parent": build_parent_descriptor(cell, max_level)
        })
    
    return ordering

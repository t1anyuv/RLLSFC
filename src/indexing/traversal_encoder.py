"""基于四叉树索引生成遍历顺序与编码。"""
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

from src.indexing.lsfc_loader import LSFCMappingLoader
from src.indexing.quadtree_cell import QuadTreeCell
from src.indexing.quadtree_index import QuadTreeIndex


def encode_with_order(order: Iterable[QuadTreeCell]) -> Dict[QuadTreeCell, int]:
    """按照给定的遍历顺序对单元格编码。"""
    encoding: Dict[QuadTreeCell, int] = {}
    for index, cell in enumerate(order):
        cell.code = index
        encoding[cell] = index
    return encoding


class TraversalOrderEncoder:
    """根据给定逻辑生成遍历顺序与对应的编码。"""

    def __init__(self, quadtree: QuadTreeIndex, alpha: int = 2, beta: int = 2):
        self.quadtree = quadtree
        self.alpha = alpha
        self.beta = beta
        self._order_loader: Optional[LSFCMappingLoader] = None

    def z_curve_order(self, include_muted: bool = False) -> List[QuadTreeCell]:
        """返回深度优先 Z 曲线遍历顺序。"""
        ordered_cells: List[QuadTreeCell] = []

        def dfs(cell: QuadTreeCell) -> None:
            if include_muted or not cell.muted:
                ordered_cells.append(cell)
            if cell.level < self.quadtree.max_level:
                for child in cell.children:
                    if child:
                        dfs(child)

        dfs(self.quadtree.root)
        return ordered_cells

    def load_quadorder_mapping(self, filepath: Union[str, Path]) -> None:
        """加载学习到的顺序映射文件（JSON或CSV）"""
        self._order_loader = LSFCMappingLoader()
        filepath = Path(filepath)

        if filepath.suffix.lower() == '.json':
            self._order_loader.load_from_json(filepath)
        else:
            raise ValueError(f"不支持的文件格式: {filepath.suffix}，请使用 .json")

    def quadorder(self, include_muted: bool = False) -> Optional[List[QuadTreeCell]]:
        """根据加载的学习顺序映射，返回排序后的单元格列表"""
        if self._order_loader is None or not self._order_loader.is_loaded():
            return None

        # 获取所有活跃单元格
        if include_muted:
            all_cells = list(self.quadtree.all_cells.values())
        else:
            all_cells = self.quadtree.get_active_cells()

        # 使用优化后的映射加载器进行排序
        return self._order_loader.get_ordered_cells(all_cells)

    def encode_with_quadorder(self, include_muted: bool = False) -> Optional[Dict[QuadTreeCell, int]]:
        """使用学习到的顺序对单元格进行编码"""
        if self._order_loader is None or not self._order_loader.is_loaded():
            return None

        # 获取所有单元格
        if include_muted:
            all_cells = list(self.quadtree.all_cells.values())
        else:
            all_cells = self.quadtree.get_active_cells()

        # 使用优化后的批量映射方法
        return self._order_loader.apply_order_to_cells(all_cells)

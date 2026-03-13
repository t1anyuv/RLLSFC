"""高效的学习顺序映射加载器：从 JSON 加载，使用元组 key 优化查询。"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from src.indexing.quadtree_cell import QuadTreeCell


class LSFCMappingLoader:
    """高效的学习顺序映射加载器。
    
    使用元组 key 优化查询性能，仅支持 JSON 格式。
    """

    def __init__(self):
        """初始化映射加载器。"""
        self.logger = logging.getLogger(__name__)
        self._tuple_to_order: Dict[Tuple[int, Tuple[int, ...]], int] = {}
        self._quad_code_to_order: Dict[int, int] = {}
        self._max_level: Optional[int] = None
        self._loaded = False

    def load_from_json(self, filepath: Union[str, Path]) -> None:
        """从 JSON 文件加载映射。
        
        参数:
            filepath: JSON 文件路径
            
        异常:
            FileNotFoundError: 文件不存在
            ValueError: JSON 格式错误
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"映射文件不存在: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 验证必要字段
        if 'ordering' not in data:
            raise ValueError("JSON文件格式错误：缺少 'ordering' 字段")

        # 从 metadata 获取 max_level
        if 'metadata' in data and 'quadtree_max_level' in data['metadata']:
            self._max_level = data['metadata']['quadtree_max_level']
        else:
            raise ValueError("JSON文件格式错误：缺少 metadata.quadtree_max_level 字段")

        self._tuple_to_order.clear()
        self._quad_code_to_order.clear()

        # 处理 ordering 数据
        for item in data['ordering']:
            if 'quad_code' not in item or 'order' not in item:
                self.logger.warning(f"跳过无效项: {item}")
                continue

            order = item['order']
            quad_codes = item['quad_code']

            # 兼容处理：如果 quad_code 是单个数值则转为列表
            if isinstance(quad_codes, (int, float)):
                quad_codes = [int(quad_codes)]

            for q_code in quad_codes:
                q_code = int(q_code)
                try:
                    # 解码并建立索引
                    level, quadrant_sequence = self._decode_quad_code(q_code)
                    key = (level, tuple(quadrant_sequence))

                    self._tuple_to_order[key] = order
                    self._quad_code_to_order[q_code] = order
                except ValueError as e:
                    self.logger.warning(f"解码失败 quad_code={q_code}: {e}")
                    continue

        self._loaded = True
        self.logger.info(f"成功加载 {len(self._tuple_to_order)} 个映射项")


    def _decode_quad_code(self, quad_code: int) -> Tuple[int, List[int]]:
        """从 quad_code 解码得到 level 和 quadrant_sequence。
        
        编码规则（TShape 编码）：
        - Root (level=0): quad_code = 0
        - 其他节点: code = sum(quadrant[i] * ((4^(max_level - i + 1) - 1) // 3) + 1 for i in 1..level)
        
        参数:
            quad_code: 四叉树编码
            
        返回:
            (level, quadrant_sequence) 元组
            
        异常:
            ValueError: 无法解码
        """
        if self._max_level is None:
            raise RuntimeError("max_level 未初始化，请先加载配置文件")

        # Root 节点
        if quad_code == 0:
            return 0, []

        # 尝试从 level 1 到 max_level 逐层解码
        for target_level in range(1, self._max_level + 1):
            quadrant_sequence = []
            remaining = quad_code
            
            # 从第一层到目标层逐层解码
            for i in range(1, target_level + 1):
                # 计算当前层的基数（与编码公式对应）
                base = (4 ** (self._max_level - i + 1) - 1) // 3
                
                # 如果 remaining 太小，无法继续
                if remaining < 1:
                    break
                
                # 减去固定偏移 1
                remaining -= 1
                
                # 计算象限
                quadrant = remaining // base
                
                # 象限必须在 [0, 3] 范围内
                if quadrant < 0 or quadrant > 3:
                    break
                
                quadrant_sequence.append(quadrant)
                
                # 更新 remaining
                remaining -= quadrant * base
            
            # 如果正好解码完成（remaining == 0）且象限序列长度正确
            if remaining == 0 and len(quadrant_sequence) == target_level:
                return target_level, quadrant_sequence
        
        raise ValueError(
            f"无法解码 quad_code={quad_code}, max_level={self._max_level}"
        )

    def get_order(self, cell: QuadTreeCell) -> Optional[int]:
        """获取单元格的 order。
        
        参数:
            cell: 四叉树单元格
            
        返回:
            order 值，如果不存在则返回 None
            
        异常:
            RuntimeError: 映射未加载
        """
        if not self._loaded:
            raise RuntimeError("请先调用 load_from_json() 加载映射文件")
        
        key = (cell.level, tuple(cell.quadrant_sequence))
        return self._tuple_to_order.get(key)

    def get_order_from_tuple(
        self,
        level: int,
        quadrant_sequence: Tuple[int, ...]
    ) -> Optional[int]:
        """从元组获取 order。
        
        参数:
            level: 层级
            quadrant_sequence: 象限序列
            
        返回:
            order 值，如果不存在则返回 None
            
        异常:
            RuntimeError: 映射未加载
        """
        if not self._loaded:
            raise RuntimeError("请先调用 load_from_json() 加载映射文件")

        key = (level, quadrant_sequence)
        return self._tuple_to_order.get(key)

    def apply_order_to_cells(
        self,
        cells: List[QuadTreeCell]
    ) -> Dict[QuadTreeCell, int]:
        """批量应用 order 到单元格列表。
        
        参数:
            cells: 单元格列表
            
        返回:
            单元格到 order 的映射字典
            
        异常:
            RuntimeError: 映射未加载
        """
        if not self._loaded:
            raise RuntimeError("请先调用 load_from_json() 加载映射文件")

        result = {}
        for cell in cells:
            order = self.get_order(cell)
            if order is not None:
                result[cell] = order

        return result

    def get_ordered_cells(self, cells: List[QuadTreeCell]) -> List[QuadTreeCell]:
        """根据映射的 order 对单元格列表进行排序。
        
        参数:
            cells: 单元格列表
            
        返回:
            排序后的单元格列表
            
        异常:
            RuntimeError: 映射未加载
        """
        if not self._loaded:
            raise RuntimeError("请先调用 load_from_json() 加载映射文件")

        # 获取所有单元格的 order
        cell_orders = []
        for cell in cells:
            order = self.get_order(cell)
            if order is not None:
                cell_orders.append((order, cell))

        # 按 order 排序
        cell_orders.sort(key=lambda x: x[0])

        return [cell for _, cell in cell_orders]

    def is_loaded(self) -> bool:
        """检查是否已加载映射。"""
        return self._loaded

    def get_statistics(self) -> Dict[str, Union[int, float]]:
        """获取映射统计信息"""
        if not self._loaded:
            return {"loaded": False, "count": 0}

        return {
            "loaded": True,
            "count": len(self._tuple_to_order),
            "quad_code_count": len(self._quad_code_to_order),
        }


def create_order_mapping_loader(filepath: Union[str, Path]) -> LSFCMappingLoader:
    """便捷函数：从 JSON 文件创建映射加载器。
    
    参数:
        filepath: JSON 文件路径
        
    返回:
        已加载的映射加载器
        
    异常:
        ValueError: 不支持的文件格式
    """
    loader = LSFCMappingLoader()
    filepath = Path(filepath)

    if filepath.suffix.lower() == '.json':
        loader.load_from_json(filepath)
    else:
        raise ValueError(
            f"不支持的文件格式: {filepath.suffix}，仅支持 .json 格式"
        )

    return loader

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.config import TShapeConfig
from src.indexing.quadtree_cell import QuadTreeCell
from src.indexing.quadtree_index import QuadTreeIndex
from src.rl.traversal_environment import TraversalEnvironment
from src.rl.order_formatter_utils import (
    get_unique_active_nodes,
    build_active_map,
    find_active_parent,
    assemble_ordering
)
from src.utils.logger import setup_logging
from src.utils.path_manager import get_path_manager


class TrajectoryOrderFormatter:
    """
    RL轨迹后处理与配置生成Agent。

    主要功能：
    1. 从强化学习环境中提取最终访问顺序
    2. 分离活跃节点和哑节点并建立父子关联
    3. 输出 JSON 配置文件
    """

    # 版本常量
    CONFIG_VERSION = "1.2"

    def __init__(self, output_dir: Optional[str] = None, logger: Optional[logging.Logger] = None,
                 config: Optional[TShapeConfig] = None):
        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.output_dir = config.experiment.get_orders_dir()
        self.logger = logger or setup_logging("TrajectoryOrderFormatter")
        self.config = config

        self.active_ordering: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {}

    def process_order(self, env: TraversalEnvironment, quadtree: QuadTreeIndex) -> Dict[str, Any]:
        """执行完整的轨迹后处理逻辑（主入口）。
        
        参数:
            env: 遍历环境
            quadtree: 四叉树索引
            
        返回:
            包含 ordering 和 metadata 的字典
        """
        self.logger.info("开始RL轨迹后处理流程...")

        # 1. 提取原始学习序列
        order = env.learned_order()
        if not order:
            self.logger.warning("未找到访问序列，请确认环境已运行推断。")
            return {}

        max_level = quadtree.max_level

        # 2. 提取并去重活跃节点（保留原始先后顺序）
        active_nodes = get_unique_active_nodes(order, quadtree)

        # 3. 建立活跃节点快速映射表 (Cell -> Info容器)
        active_map, qc_to_info = build_active_map(active_nodes, max_level)

        # 4. 分配哑节点到对应的活跃祖先
        self._assign_muted_nodes(quadtree, active_map, qc_to_info, max_level)

        # 5. 组装最终 Ordering 列表（按原始 Order 排序）
        self.active_ordering = assemble_ordering(qc_to_info)

        # 6. 生成元数据
        self.metadata = self._generate_metadata(env, quadtree, active_nodes)

        self.logger.info(f"处理完成: 生成了 {len(self.active_ordering)} 个访问序列组")
        return {"ordering": self.active_ordering, "metadata": self.metadata}

    def _assign_muted_nodes(self, quadtree: QuadTreeIndex, active_map: Dict, qc_to_info: Dict, max_level: int) -> None:
        """将所有哑节点映射到其在访问序列中的活跃父辈。
        
        参数:
            quadtree: 四叉树索引
            active_map: Cell -> Info 映射
            qc_to_info: QC -> Info 映射
            max_level: 四叉树最大层级
        """
        active_qcs = set(qc_to_info.keys())

        for cell in quadtree.all_cells.values():
            if not cell.muted:
                continue

            mqc = cell.get_quadrant_code(max_level)
            if mqc in active_qcs:
                continue

            # 寻找最近的活跃祖先
            parent = find_active_parent(cell)
            if parent in active_map:
                active_map[parent]["muted_set"].add(mqc)
            else:
                self.logger.debug(f"哑节点 {mqc} 无有效活跃祖先轨迹，归为孤立节点。")

    def _generate_metadata(
        self,
        env: TraversalEnvironment,
        quadtree: QuadTreeIndex,
        active_nodes: List[QuadTreeCell]
    ) -> Dict[str, Any]:
        """生成配置元数据。
        
        参数:
            env: 遍历环境
            quadtree: 四叉树索引
            active_nodes: 活跃节点列表
            
        返回:
            元数据字典
        """
        all_cells = quadtree.all_cells.values()

        return {
            "total_cells": len(all_cells),
            "active_cells": len(active_nodes),
            "muted_cells": len([c for c in all_cells if c.muted]),
            "spatial_boundary": {
                "xmin": float(quadtree.bbox.min_x),
                "ymin": float(quadtree.bbox.min_y),
                "xmax": float(quadtree.bbox.max_x),
                "ymax": float(quadtree.bbox.max_y),
            },
            "max_shapes": int(quadtree.max_shape_num),
            "quadtree_max_level": int(quadtree.max_level),
            "global_alpha": int(env.alpha),
            "global_beta": int(env.beta),
            "generation_timestamp": datetime.now().isoformat(),
            "version": self.CONFIG_VERSION
        }

    def save_json_config(self, data: Dict[str, Any], filename: str = "learned_order.json") -> str:
        """保存 JSON 配置文件。
        
        参数:
            data: 包含 ordering 和 metadata 的数据
            filename: 文件名
            
        返回:
            保存的文件路径
        """
        filepath = self.output_dir / filename
        export_data = {"ordering": data["ordering"], "metadata": data["metadata"]}

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"JSON 配置已保存: {filepath}")
        return str(filepath)

    def generate_config_file(
        self,
        env: TraversalEnvironment,
        quadtree: QuadTreeIndex,
        filename: str = "learned_order.json"
    ) -> Tuple[Dict[str, Any], str]:
        """生成配置文件的核心公开方法。
        
        参数:
            env: 遍历环境
            quadtree: 四叉树索引
            filename: JSON 文件名
            
        返回:
            (data, json_path) 元组
        """
        data = self.process_order(env, quadtree)
        if not data:
            return {}, ""

        json_path = self.save_json_config(data, filename)
        self.logger.info(f"配置文件生成完成: {filename}")
        
        return data, json_path


def create_rl_postprocessor(output_dir: Optional[str] = None,
                            config: Optional[TShapeConfig] = None) -> TrajectoryOrderFormatter:
    """便捷工厂函数。"""
    return TrajectoryOrderFormatter(output_dir=output_dir, config=config)

"""
优化后的四叉树空间索引。
通过 (level, x_index, y_index) 实现 O(1) 单元格定位，并优化了剪枝与相交判定效率。
"""
import math
import logging
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from src.core.bounding_box import SpatialBoundingBox
from src.indexing.quadtree_cell import QuadTreeCell
from src.indexing.quadtree_validator import QuadTreeValidator
from src.indexing.signature_processor import SignatureProcessor
from src.storage import TrajectoryStorage, InMemoryTrajectoryStorage
from src.utils.trajectory_geometry import compute_trajectory_bounding_box


class QuadTreeIndex:
    """四叉树空间索引。
    
    通过 (level, x_index, y_index) 实现 O(1) 单元格定位，并优化了剪枝与相交判定效率。
    支持注入式轨迹存储后端，可动态选择内存或磁盘存储模式。
    
    常量定义：
    """
    # 几何计算精度
    GEOMETRY_EPSILON = 1e-10

    def __init__(self, bbox: SpatialBoundingBox, max_level: int, alpha=2, beta=2, 
                 storage: Optional[TrajectoryStorage] = None,
                 parallel_signatures: bool = False,
                 signature_workers: Optional[int] = None):
        self.bbox = bbox
        self.max_level = max_level
        self.alpha = alpha
        self.beta = beta
        self.width = self.bbox.max_x - self.bbox.min_x
        self.height = self.bbox.max_y - self.bbox.min_y
        self._next_code = 0

        # 使用 (level, [grid]) 作为键
        self.all_cells: Dict[Tuple[int, int, int], QuadTreeCell] = {}
        self.root = QuadTreeCell(bbox, level=0, quadrant_sequence=[], code=self._get_next_code(),
                                 alpha=alpha, beta=beta)

        # 轨迹存储后端
        self._storage = storage or InMemoryTrajectoryStorage()
        self.trajectory_to_cells: Dict[int, QuadTreeCell] = {}
        
        # 内存与磁盘模式
        if isinstance(self._storage, InMemoryTrajectoryStorage):
            self.trajectory_points = self._storage._trajectory_points
            self.trajectory_mbrs = self._storage._trajectory_mbrs
        else:
            # 磁盘模式下，这些属性不再直接可用
            self.trajectory_points = None
            self.trajectory_mbrs = None

        # 初始化验证器和签名处理器（支持并行）
        self.logger = logging.getLogger(__name__)
        self.validator = QuadTreeValidator(self.logger)
        self.signature_processor = SignatureProcessor(
            alpha, beta, 
            max_workers=signature_workers
        )
        self._parallel_signatures = parallel_signatures

        self._build_tree()

    def _get_next_code(self) -> int:
        """获取下一个可用的唯一编码。"""
        code = self._next_code
        self._next_code += 1
        return code

    def _build_tree(self) -> None:
        """递归构建完整的四叉树拓扑结构。"""
        # 根节点索引设为 (0, 0, 0)
        self.all_cells[(0, 0, 0)] = self.root

        def build_recursive(cell: QuadTreeCell, level: int, x_idx: int, y_idx: int) -> None:
            if level >= self.max_level:
                return

            mid_x = (cell.bbox.min_x + cell.bbox.max_x) / 2
            mid_y = (cell.bbox.min_y + cell.bbox.max_y) / 2

            # 象限逻辑：0:LB, 1:RB, 2:LT, 3:RT
            # 对应的网格偏移：nx = x*2+(0或1), ny = y*2+(0或1)
            sub_configs = [
                (0, cell.bbox.min_x, cell.bbox.min_y, mid_x, mid_y, x_idx * 2, y_idx * 2),
                (1, mid_x, cell.bbox.min_y, cell.bbox.max_x, mid_y, x_idx * 2 + 1, y_idx * 2),
                (2, cell.bbox.min_x, mid_y, mid_x, cell.bbox.max_y, x_idx * 2, y_idx * 2 + 1),
                (3, mid_x, mid_y, cell.bbox.max_x, cell.bbox.max_y, x_idx * 2 + 1, y_idx * 2 + 1)
            ]

            for quad, x1, y1, x2, y2, nx, ny in sub_configs:
                child_seq = cell.quadrant_sequence + [quad]
                child = QuadTreeCell(SpatialBoundingBox(x1, y1, x2, y2), level + 1,
                                     child_seq, self._get_next_code(),
                                     self.alpha, self.beta)
                child.parent = cell
                cell.children[quad] = child

                # 建立逻辑网格索引
                self.all_cells[(level + 1, nx, ny)] = child
                build_recursive(child, level + 1, nx, ny)

        build_recursive(self.root, 0, 0, 0)

    def get_cell_at(self, x: float, y: float, level: int) -> Optional[QuadTreeCell]:
        """依据轨迹 MBR 左下角点定位其所在单元格"""
        if not self.bbox.contains_point(x, y):
            return None

        if level == 0:
            return self.root

        # 计算在指定层级的网格坐标索引
        # 使用 floor + epsilon 避免浮点精度问题
        import math
        num_cells = 1 << level
        eps = self.GEOMETRY_EPSILON
        x_idx = int(math.floor((x - self.bbox.min_x + eps) / self.width * num_cells))
        y_idx = int(math.floor((y - self.bbox.min_y + eps) / self.height * num_cells))

        # 处理坐标恰好等于 max_x/max_y 的情况
        x_idx = min(x_idx, num_cells - 1)
        y_idx = min(y_idx, num_cells - 1)

        return self.all_cells.get((level, x_idx, y_idx))

    def get_cells_at_level(self, level: int) -> List[QuadTreeCell]:
        """获取特定层级的所有单元格（按象限序列排序，即 Z-Order）。"""
        cells = [c for c in self.all_cells.values() if c.level == level]
        return sorted(cells, key=lambda c: c.quadrant_sequence)

    def get_cells_by_level(self):
        """获取按层级分布的所有单元格"""
        return [self.get_cells_at_level(lvl) for lvl in range(self.max_level + 1)]

    def get_all_cells(self) -> List[QuadTreeCell]:
        """返回所有单元格集合，按 (level, x, y) 排序以保证结果稳定性。"""
        return [self.all_cells[key] for key in sorted(self.all_cells.keys())]

    def iter_active_cells(self) -> Iterable[QuadTreeCell]:
        """深度优先遍历所有活跃单元格。"""
        stack: List[QuadTreeCell] = [self.root]
        while stack:
            cell = stack.pop()
            if not cell:
                continue
            for child in reversed(cell.children):
                if child:
                    stack.append(child)
            if not cell.muted:
                yield cell

    def get_active_cells(self) -> List[QuadTreeCell]:
        """获取当前所有未屏蔽的单元格。"""
        return [cell for cell in self.all_cells.values() if not cell.muted]

    def get_eight_neighbor_cells(self, cell: QuadTreeCell) -> List[QuadTreeCell]:
        """利用 get_cell_at 的高效实现获取相邻单元格。"""
        if cell.level == 0:
            return []

        neighbors: List[QuadTreeCell] = []
        cell_w = cell.bbox.max_x - cell.bbox.min_x
        cell_h = cell.bbox.max_y - cell.bbox.min_y
        cx, cy = cell.get_center()

        # 八个方向探测点
        directions = [
            (cx + cell_w, cy), (cx - cell_w, cy), (cx, cy + cell_h), (cx, cy - cell_h),
            (cx + cell_w, cy + cell_h), (cx - cell_w, cy + cell_h),
            (cx + cell_w, cy - cell_h), (cx - cell_w, cy - cell_h)
        ]

        discovered = {cell}
        for tx, ty in directions:
            neighbor = self.get_cell_at(tx, ty, cell.level)
            if neighbor and neighbor not in discovered:
                neighbors.append(neighbor)
                discovered.add(neighbor)
        return neighbors

    def reset_muted_flags(self) -> None:
        """重置所有节点为非静默状态。"""
        for cell in self.all_cells.values():
            cell.muted = False

    def _rebuild_trajectory_index(self) -> None:
        """重建轨迹到单元格的实时映射。"""
        self.trajectory_to_cells.clear()
        # 仅遍历非屏蔽单元格
        for cell in self.iter_active_cells():
            for trajectory_id in cell.trajectories:
                self.trajectory_to_cells[trajectory_id] = cell

    def _perform_bottom_up_merge(self, cells_by_level: list, min_cell_trajs: int) -> Dict:
        """内部方法：执行拓扑剪枝合并。"""
        muted_count = 0
        for level in range(self.max_level, 0, -1):
            for cell in cells_by_level[level]:
                if len(cell.trajectories) < min_cell_trajs:
                    self._mute_cell(cell)
                    muted_count += 1

        return {
            "muted": muted_count,
            "shrunk_alpha": 0,
            "shrunk_beta": 0,
            "both_shrunk": 0
        }

    def _mute_cell(self, cell: QuadTreeCell) -> None:
        """逻辑删除节点，将数据转移至父节点。"""
        if cell == self.root:
            return

        if cell.trajectories and cell.parent:
            parent_ee = cell.parent.get_enlarged_element_bbox(self.alpha, self.beta)
            for tid in list(cell.trajectories):
                traj_mbr = cell.trajectory_mbrs[tid]
                if not parent_ee.contains(traj_mbr):
                    raise AssertionError(f"轨迹 {tid} 移动失败：父级 EE 无法包围其 MBR")

                self.trajectory_to_cells[tid] = cell.parent

            cell.parent.trajectories.update(cell.trajectories)
            cell.parent.trajectory_mbrs.update(cell.trajectory_mbrs)

        cell.trajectories.clear()
        cell.trajectory_mbrs.clear()
        cell.signatures.clear()
        cell.muted = True

    def _validate_reassignment(self, traj_id: int, points: List[Tuple[float, float]]) -> None:
        """验证轨迹重新分配的一致性。"""
        old_cell = self.trajectory_to_cells[traj_id]
        old_points = self._storage.get_trajectory(traj_id)
        self.validator.validate_reassignment(traj_id, points, old_cell, old_points, self)

    def _validate_merge_results(self, initial_tids: set, initial_count: int, min_threshold: int) -> None:
        """验证合并后数据的一致性、完整性及唯一性。"""
        active_cells = self.get_active_cells()
        self.validator.validate_merge_results(
            initial_tids, initial_count, min_threshold,
            active_cells, self.trajectory_to_cells, self.root
        )

    def _process_cell_signatures(self, cells_by_level: list, stats: Dict, enable_optimize: bool) -> None:
        """执行签名计算与自适应优化（可选）。"""
        # 为签名处理器提供轨迹点访问方法
        def get_points_for_signature(tid: int):
            return self._storage.get_trajectory(tid)
        
        result = self.signature_processor.process_cell_signatures(
            cells_by_level, get_points_for_signature, self.max_level, enable_optimize,
            parallel=self._parallel_signatures
        )
        
        # 更新统计信息
        stats.update({
            "shrunk_alpha": result["shrunk_alpha"],
            "shrunk_beta": result["shrunk_beta"],
            "both_shrunk": result["both_shrunk"]
        })

    def post_prune_tree(self, min_cell_trajs: int = 1) -> Dict[str, int]:
        """执行后序遍历剪枝，根据轨迹密度合并节点并重构索引。
        
        参数:
            min_cell_trajs: 最小轨迹数阈值
            
        返回:
            包含剪枝统计信息的字典
        """
        self.logger.info(f"执行剪枝: 轨迹数小于 {min_cell_trajs} 将被剪枝...")
        self.reset_muted_flags()
        total_cells_count = len(self.all_cells)

        if min_cell_trajs <= 0:
            return {"before": total_cells_count, "after": total_cells_count, "muted": 0}

        # 1. 数据完整性记录，用于验证
        initial_tids = {tid for cell in self.all_cells.values() for tid in cell.trajectories}
        initial_total_count = sum(len(cell.trajectories) for cell in self.all_cells.values())

        # 2. 预分组：按层级组织单元格以支持自底向上合并
        cells_by_level = self.get_cells_by_level()

        # 3. 执行核心剪枝：自底向上合并稀疏节点
        stats = self._perform_bottom_up_merge(cells_by_level, min_cell_trajs)

        # 4. 索引重构：更新轨迹 ID 到新单元格的映射关系
        self._rebuild_trajectory_index()

        # 5. 验证：确保合并过程中没有丢失轨迹数据
        self._validate_merge_results(initial_tids, initial_total_count, min_cell_trajs)

        active_count = sum(1 for cell in self.all_cells.values() if not cell.muted)
        stats.update({
            "before": total_cells_count,
            "after": active_count,
            "muted": total_cells_count - active_count
        })

        self.logger.info(f"[Prune] 结构剪枝完成。活跃单元格: {active_count}/{total_cells_count}")
        return stats

    def compute_signatures(self, enable_optimize: bool = True) -> Dict[str, int]:
        """在剪枝后的活跃节点上，优化 Alpha/Beta 参数并重新计算签名。
        
        参数:
            enable_optimize: 是否启用参数优化
            
        返回:
            包含优化统计信息的字典
        """
        cells_by_level = self.get_cells_by_level()
        opt_stats = {
            "shrunk_alpha": 0,
            "shrunk_beta": 0,
            "both_shrunk": 0
        }

        self.logger.info("[Signatures] 开始计算活跃单元格的签名...")

        # 针对每个活跃 Cell 寻找最小包含矩形的局部最优 Alpha/Beta
        self._process_cell_signatures(cells_by_level, opt_stats, enable_optimize=enable_optimize)

        if enable_optimize:
            self.logger.info(
                f"[Optimize] 统计: Alpha收缩={opt_stats['shrunk_alpha']}, "
                f"Beta收缩={opt_stats['shrunk_beta']}, 同时收缩={opt_stats['both_shrunk']}"
            )

        return opt_stats

    def assign_trajectory(self, traj_id: int, points: List[Tuple[float, float]]) -> None:
        """分配轨迹并验证 EE 包含性。"""
        if not points:
            return

        if self._storage.has_trajectory(traj_id):
            self._validate_reassignment(traj_id, points)
            return

        # 使用存储后端保存轨迹
        self._storage.store_trajectory(traj_id, points)

        traj_bbox = compute_trajectory_bounding_box(points)
        target_lvl = self.compute_target_level(traj_bbox)
        cell = self.get_cell_at(traj_bbox.min_x, traj_bbox.min_y, target_lvl)

        if cell:
            ee_bbox = cell.get_enlarged_element_bbox(self.alpha, self.beta)
            if not ee_bbox.contains(traj_bbox):
                print(f"[DEBUG] Traj {traj_id} EE containment failed:")
                print(f"  traj_bbox: ({traj_bbox.min_x}, {traj_bbox.min_y}) - ({traj_bbox.max_x}, {traj_bbox.max_y})")
                print(f"  cell_bbox: ({cell.bbox.min_x}, {cell.bbox.min_y}) - ({cell.bbox.max_x}, {cell.bbox.max_y})")
                print(f"  ee_bbox: ({ee_bbox.min_x}, {ee_bbox.min_y}) - ({ee_bbox.max_x}, {ee_bbox.max_y})")
                print(f"  alpha={self.alpha}, beta={self.beta}, target_lvl={target_lvl}")
                print(f"  global: ({self.bbox.min_x}, {self.bbox.min_y}) - ({self.bbox.max_x}, {self.bbox.max_y})")
            assert ee_bbox.contains(traj_bbox), f"EE 包含性验证失败: Traj {traj_id}"

            cell.trajectories.add(traj_id)
            cell.trajectory_mbrs[traj_id] = traj_bbox

            from src.utils.signature import compute_traj_signature
            cell.signatures[traj_id] = compute_traj_signature(self.alpha, self.beta, cell, points)

            self.trajectory_to_cells[traj_id] = cell

    def compute_target_level(self, bbox: SpatialBoundingBox) -> int:
        """计算轨迹对应层级。
        
        参数:
            bbox: 轨迹边界框
            
        返回:
            目标层级
        """
        x1, y1, x2, y2 = bbox.min_x, bbox.min_y, bbox.max_x, bbox.max_y
        eps = self.GEOMETRY_EPSILON

        max_dim_rel = max((x2 - x1) / (self.alpha * self.width),
                          (y2 - y1) / (self.beta * self.height))

        l_suggested = self.max_level if max_dim_rel <= 0 else int(math.floor(-math.log2(max_dim_rel)))
        l_suggested = max(0, min(l_suggested, self.max_level))

        def check_containment(lvl):
            w = self.width * (0.5 ** lvl)
            h = self.height * (0.5 ** lvl)
            gx = math.floor((x1 - self.bbox.min_x + eps) / w) * w + self.bbox.min_x
            gy = math.floor((y1 - self.bbox.min_y + eps) / h) * h + self.bbox.min_y
            return (gx + self.alpha * w >= x2 - eps) and (gy + self.beta * h >= y2 - eps)

        target = l_suggested if check_containment(l_suggested) else l_suggested - 1
        return max(0, min(target, self.max_level))

    def trajectory_intersects_cell(self, trajectory_id: int, cell: QuadTreeCell) -> bool:
        """基于 MBR 快速判断轨迹与单元格相交。"""
        traj_mbr = self._storage.get_trajectory_mbr(trajectory_id)
        return cell.bbox.intersects(traj_mbr) if traj_mbr else False

    def get_quadtree_stats(self) -> Dict[str, int]:
        all_cells = list(self.all_cells.values())
        return {
            "total_cells": len(all_cells),
            "active_cells": len([c for c in all_cells if not c.muted]),
            "muted_cells": len([c for c in all_cells if c.muted]),
            "max_level": self.max_level
        }

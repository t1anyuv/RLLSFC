"""四叉树签名处理器。"""
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Union, Optional

import numpy as np

from src.indexing.quadtree_cell import QuadTreeCell
from src.training.signature_optimizer import SignatureOptimizer
from src.utils.signature import compute_signature_vectorized


# 类型别名：可以接受字典或可调用对象
TrajectoryPointsAccessor = Union[Dict[int, List], Callable[[int], List]]


class SignatureProcessor:
    """四叉树签名处理器。
    
    负责计算单元格的轨迹签名，并可选地优化 Alpha/Beta 参数。支持并行计算以加速处理。
    """

    def __init__(self, global_alpha: int, global_beta: int, max_workers: Optional[int] = None):
        self.global_alpha = global_alpha
        self.global_beta = global_beta
        self.optimizer = SignatureOptimizer(alpha_range=(2, 6), beta_range=(2, 6))
        self.max_workers = max_workers
        self.logger = logging.getLogger(self.__class__.__name__)

    def process_cell_signatures(
        self,
        cells_by_level: List[List[QuadTreeCell]],
        trajectory_points: TrajectoryPointsAccessor,
        max_level: int,
        enable_optimize: bool = True,
        parallel: bool = False,
    ) -> Dict[str, int]:
        """执行签名计算与自适应优化（可选）。
        
        参数:
            cells_by_level: 按层级组织的单元格列表
            trajectory_points: 轨迹点数据，可以是字典或函数(tid)->points
            max_level: 四叉树最大层级
            enable_optimize: 是否启用参数优化
            parallel: 是否使用并行计算
            
        返回:
            统计信息字典
        """
        # 统一访问接口
        def get_points(tid: int):
            if callable(trajectory_points):
                return trajectory_points(tid)
            return trajectory_points.get(tid)
        
        if parallel and self.max_workers and self.max_workers > 1:
            return self._process_parallel(cells_by_level, get_points, max_level, enable_optimize)
        
        return self._process_serial(cells_by_level, get_points, max_level, enable_optimize)
    
    def _process_serial(
        self,
        cells_by_level: List[List[QuadTreeCell]],
        get_points: Callable[[int], List],
        max_level: int,
        enable_optimize: bool
    ) -> Dict[str, int]:
        """串行处理签名计算。"""
        stats = {"shrunk_alpha": 0, "shrunk_beta": 0, "both_shrunk": 0}
        current_max_shapes = 0

        for level in range(0, max_level + 1):
            for cell in cells_by_level[level]:
                result = self._process_single_cell(cell, get_points, enable_optimize)
                stats["shrunk_alpha"] += result["shrunk_alpha"]
                stats["shrunk_beta"] += result["shrunk_beta"]
                stats["both_shrunk"] += result["both_shrunk"]
                current_max_shapes = max(current_max_shapes, result["shapes"])

        return {
            **stats,
            "max_shape_num": int(current_max_shapes * 1.5)
        }

    
    def _process_parallel(
        self,
        cells_by_level: List[List[QuadTreeCell]],
        get_points: Callable[[int], List],
        max_level: int,
        enable_optimize: bool
    ) -> Dict[str, int]:
        """使用线程池并行处理签名计算。"""
        # 收集所有需要处理的单元格
        all_cells = []
        for level in range(0, max_level + 1):
            for cell in cells_by_level[level]:
                if not cell.muted and cell.trajectories:
                    all_cells.append(cell)
        
        if not all_cells:
            return {"shrunk_alpha": 0, "shrunk_beta": 0, "both_shrunk": 0, "max_shape_num": 0}
        
        self.logger.info(f"并行计算 {len(all_cells)} 个单元格的签名 (workers={self.max_workers})")
        
        # 使用线程池并行处理
        max_workers = min(self.max_workers, len(all_cells))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(
                lambda cell: self._process_single_cell(cell, get_points, enable_optimize),
                all_cells
            ))
        
        # 合并统计信息
        stats = {"shrunk_alpha": 0, "shrunk_beta": 0, "both_shrunk": 0}
        current_max_shapes = 0
        for result in results:
            stats["shrunk_alpha"] += result["shrunk_alpha"]
            stats["shrunk_beta"] += result["shrunk_beta"]
            stats["both_shrunk"] += result["both_shrunk"]
            current_max_shapes = max(current_max_shapes, result["shapes"])
        
        return {
            **stats,
            "max_shape_num": int(current_max_shapes * 1.5)
        }

    
    def _process_single_cell(
        self,
        cell: QuadTreeCell,
        get_points: Callable[[int], List],
        enable_optimize: bool
    ) -> Dict[str, int]:
        """处理单个单元格的签名计算。
        
        返回:
            包含 shrunk_alpha, shrunk_beta, both_shrunk, shapes 的字典
        """
        stats = {"shrunk_alpha": 0, "shrunk_beta": 0, "both_shrunk": 0}
        
        # 1. 跳过已屏蔽或无轨迹的 Cell
        if cell.muted or not cell.trajectories:
            cell.alpha, cell.beta = self.global_alpha, self.global_beta
            return {**stats, "shapes": 0}
        
        # 2. 准备点数据
        traj_pts_list = []
        valid_tids = []
        for tid in cell.trajectories:
            pts = get_points(tid)
            if pts is not None:
                traj_pts_list.append(np.array(pts))
                valid_tids.append(tid)
        
        if not traj_pts_list:
            cell.alpha, cell.beta = self.global_alpha, self.global_beta
            return {**stats, "shapes": 0}
        
        # 3. 参数优化
        if enable_optimize:
            best_alpha, best_beta = self.optimizer.find_best_config(
                self.global_alpha, self.global_beta, traj_pts_list, cell
            )
            if best_alpha < self.global_alpha:
                stats["shrunk_alpha"] = 1
            if best_beta < self.global_beta:
                stats["shrunk_beta"] = 1
            if best_alpha < self.global_alpha and best_beta < self.global_beta:
                stats["both_shrunk"] = 1
            
            cell.alpha, cell.beta = best_alpha, best_beta
        else:
            cell.alpha, cell.beta = self.global_alpha, self.global_beta
        
        # 4. 计算签名
        ee_bbox = cell.get_enlarged_element_bbox(self.global_alpha, self.global_beta)
        for tid, pts_np in zip(valid_tids, traj_pts_list):
            cell.signatures[tid] = compute_signature_vectorized(
                cell.alpha, cell.beta, pts_np, ee_bbox
            )
        
        # 5. 统计该 Cell 内去重后的形状数量
        unique_shapes = len(set(cell.signatures.values()))
        return {**stats, "shapes": unique_shapes}
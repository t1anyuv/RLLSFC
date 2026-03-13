"""四叉树数据一致性验证器。"""
import logging
from typing import Dict, List, Set, Tuple

from src.indexing.quadtree_cell import QuadTreeCell
from src.utils.trajectory_geometry import compute_trajectory_bounding_box


class QuadTreeValidator:
    """四叉树数据一致性验证器。
    
    负责验证四叉树操作（如剪枝、轨迹分配）后的数据完整性和一致性。
    """

    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)

    def validate_reassignment(
        self,
        traj_id: int,
        points: List[Tuple[float, float]],
        old_cell: QuadTreeCell,
        old_points: List[Tuple[float, float]],
        quadtree
    ) -> None:
        """验证轨迹重新分配的一致性。
        
        参数:
            traj_id: 轨迹ID
            points: 新的轨迹点
            old_cell: 原单元格
            old_points: 原轨迹点
            quadtree: 四叉树索引
            
        异常:
            AssertionError: 如果检测到几何漂移
        """
        self.logger.info(f"[DIAGNOSE] Detected Re-assignment for TID: {traj_id}")

        # 1. 验证轨迹点是否一致
        points_match = (len(points) == len(old_points)) and all(
            p1 == p2 for p1, p2 in zip(points, old_points)
        )
        self.logger.info(f"  - Points Match: {points_match}")
        if not points_match:
            self.logger.warning(
                f"    Warning: Points content changed! Old len: {len(old_points)}, New len: {len(points)}"
            )

        # 2. 验证计算出的 Cell 信息是否一致
        new_bbox = compute_trajectory_bounding_box(points)
        new_lvl = quadtree.compute_target_level(new_bbox)
        new_cell = quadtree.get_cell_at(new_bbox.min_x, new_bbox.min_y, new_lvl)

        self.logger.info(
            f"  - Old Cell: Level {old_cell.level}, Path: {getattr(old_cell, 'quadrant_sequence', 'N/A')}"
        )
        self.logger.info(
            f"  - New Cell: Level {new_lvl}, Path: {getattr(new_cell, 'quadrant_sequence', 'N/A')}"
        )

        cell_match = (old_cell == new_cell)
        self.logger.info(f"  - Cell Consistency Match: {cell_match}")

        if not cell_match:
            self.logger.error(f"  [!] CRITICAL: Geometry drift detected for TID {traj_id}!")
            self.logger.error(f"    MBR: {new_bbox}")
            raise AssertionError("Reassignment Trajectory!")

    def validate_merge_results(
        self,
        initial_tids: Set[int],
        initial_count: int,
        min_threshold: int,
        active_cells: List[QuadTreeCell],
        trajectory_to_cell: Dict[int, QuadTreeCell],
        root_cell: QuadTreeCell
    ) -> None:
        """验证合并后数据的一致性、完整性及唯一性。
        
        参数:
            initial_tids: 初始轨迹ID集合
            initial_count: 初始轨迹总数
            min_threshold: 最小轨迹数阈值
            active_cells: 活跃单元格列表
            trajectory_to_cell: 轨迹到单元格的映射
            root_cell: 根节点
            
        异常:
            AssertionError: 如果验证失败
        """
        # 1. 基础集合验证：轨迹 TID 和总数量必须完全吻合
        final_tids = {tid for cell in active_cells for tid in cell.trajectories}
        final_total_count = sum(len(cell.trajectories) for cell in active_cells)

        assert final_tids == initial_tids, \
            f"轨迹ID不一致！缺失: {initial_tids - final_tids}, 多出: {final_tids - initial_tids}"
        assert final_total_count == initial_count, \
            f"轨迹总数变化！期望 {initial_count}, 实际 {final_total_count}"

        # 2. 节点合法性验证：非根节点的活跃 Cell 轨迹数必须满足阈值
        for cell in active_cells:
            if cell != root_cell:
                assert len(cell.trajectories) >= min_threshold, \
                    f"Cell {cell.code} (Level {cell.level}) 轨迹数 {len(cell.trajectories)} 低于阈值 {min_threshold}"

        # 3. 实时映射表验证：TID -> Cell 的双向绑定必须正确
        for tid in initial_tids:
            mapped_cell = trajectory_to_cell.get(tid)
            # 验证轨迹是否在映射表中
            assert mapped_cell is not None, f"轨迹 {tid} 在 trajectory_to_cell 映射表中丢失"
            # 验证映射的 Cell 是否为活跃状态
            assert not mapped_cell.muted, f"轨迹 {tid} 映射到了已屏蔽的 Cell {mapped_cell.code}"
            # 验证 Cell 内部是否真的持有该轨迹
            assert tid in mapped_cell.trajectories, \
                f"映射表冲突：轨迹 {tid} 指向 Cell {mapped_cell.code}，但该 Cell 的集合中不包含此 TID"

        # 4. 唯一映射验证：确保一条轨迹在整个索引中只属于一个活跃 Cell
        tid_appearance_counts = {}
        for cell in active_cells:
            for tid in cell.trajectories:
                tid_appearance_counts[tid] = tid_appearance_counts.get(tid, 0) + 1

        has_duplicates = False
        for tid, count in tid_appearance_counts.items():
            if count > 1:
                has_duplicates = True
                self.logger.error(f"\n[ERROR] 发现唯一性破坏！轨迹 TID: {tid} 重复出现在 {count} 个 Cell 中:")

                # 查找所有持有该 TID 的 Cell 并打印详情
                for cell in active_cells:
                    if tid in cell.trajectories:
                        path = "->".join(map(str, cell.quadrant_sequence))
                        sig = cell.signatures.get(tid)
                        self.logger.error(
                            f"  > Level: {cell.level} | Path: [{path}] | "
                            f"Muted: {cell.muted} | Signature: {bin(sig) if sig is not None else 'None'}"
                        )

                # 额外检查映射表指向哪里
                mapped = trajectory_to_cell.get(tid)
                if mapped:
                    self.logger.error(
                        f"  > 映射表(trajectory_to_cell)当前指向: "
                        f"Level {mapped.level} [Path: {'->'.join(map(str, mapped.quadrant_sequence))}]"
                    )

        if has_duplicates:
            raise AssertionError("唯一性破坏：检测到轨迹重复分配，请检查 _mute_cell 或轨迹上移逻辑。")

        self.logger.info(
            f"[Validate] 剪枝结果验证通过: {len(active_cells)} Active Cells, {final_total_count} Trajectories."
        )

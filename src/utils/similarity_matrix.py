"""相似度矩阵预计算工具。"""
import logging
import multiprocessing
import os
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from typing import Deque, Dict, Iterator, List, Optional, Tuple, Any

import numpy as np
from tqdm import tqdm

from src.indexing.quadtree_cell import QuadTreeCell
from src.indexing.quadtree_index import QuadTreeIndex
from src.reward.cost_evaluator import TraversalCostEvaluator


_WORKER_COST_EVALUATOR: Optional[TraversalCostEvaluator] = None
_WORKER_ALL_CELLS: Optional[List[QuadTreeCell]] = None


def _init_similarity_worker(
    cost_evaluator: TraversalCostEvaluator,
    all_cells: List[QuadTreeCell]
) -> None:
    """Initialize shared read-only state once per child process."""
    global _WORKER_COST_EVALUATOR, _WORKER_ALL_CELLS
    _WORKER_COST_EVALUATOR = cost_evaluator
    _WORKER_ALL_CELLS = all_cells


def _compute_chunk_task(
    indices_pairs: List[Tuple[int, int]]
) -> List[Tuple[int, int, float]]:
    """子进程执行的任务函数：计算一批单元格对的相似度。
    
    参数:
        task_data: (索引对列表, 成本评估器, 单元格列表)
        
    返回:
        (i, j, similarity) 三元组列表
    """
    if _WORKER_COST_EVALUATOR is None or _WORKER_ALL_CELLS is None:
        raise RuntimeError("Similarity worker is not initialized")

    results = []
    for i, j in indices_pairs:
        sim = _WORKER_COST_EVALUATOR.jaccard_similarity(
            _WORKER_ALL_CELLS[i],
            _WORKER_ALL_CELLS[j],
            use_cache=False,
        )
        results.append((i, j, float(sim)))
    return results


class SimilarityMatrix:
    """
    相似度矩阵预计算工具。
    
    支持多进程并行计算和高效的矩阵存储/加载。
    """
    
    # 常量定义
    DEFAULT_CHUNK_SIZE = 500
    DEFAULT_WORKERS = 12  # 使用 CPU 核心数

    def __init__(self, quadtree: QuadTreeIndex, cost_evaluator: TraversalCostEvaluator):
        self.quadtree = quadtree
        self.cost_evaluator = cost_evaluator
        self.logger = logging.getLogger(__name__)

        self.matrix_array: Optional[np.ndarray] = None
        self.cell_to_index: Dict[QuadTreeCell, int] = {}
        self.all_cells: List[QuadTreeCell] = []
        self.is_computed: bool = False

    def _build_cell_index_mapping(self, all_cells: List[QuadTreeCell]) -> None:
        self.all_cells = all_cells
        self.cell_to_index = {cell: idx for idx, cell in enumerate(all_cells)}
        n = len(all_cells)
        self.matrix_array = np.zeros((n, n), dtype=np.float32)

    def compute(
        self,
        all_cells: List[QuadTreeCell],
        use_symmetric: bool = True,
        show_progress: bool = True,
        num_workers: Optional[int] = None,
        chunk_size: int = None
    ) -> None:
        """使用多进程分块并行预计算相似度矩阵。
        
        参数:
            all_cells: 所有单元格列表
            use_symmetric: 是否利用对称性减少计算量
            show_progress: 是否显示进度条
            num_workers: 工作进程数，None 表示使用 CPU 核心数
            chunk_size: 每个任务块的大小
        """
        n = len(all_cells)
        chunk_size = chunk_size or self.DEFAULT_CHUNK_SIZE
        num_workers = num_workers if num_workers is not None else self.DEFAULT_WORKERS
        
        self._build_cell_index_mapping(all_cells)

        # 生成任务对
        total_pairs = self._count_task_pairs(n, use_symmetric)

        self.logger.info(
            f"开始并行预计算相似度矩阵 (Workers: {num_workers}, Total: {total_pairs})"
        )
        start_time = time.perf_counter()

        if num_workers <= 1:
            self._compute_sequential(n, all_cells, use_symmetric, show_progress)
        else:
            self._compute_parallel(
                n, all_cells, use_symmetric, show_progress,
                num_workers, chunk_size, total_pairs
            )

        self.is_computed = True
        elapsed = time.perf_counter() - start_time
        self.logger.info(
            f"预计算完成！耗时: {elapsed:.2f}s | 速度: {total_pairs / elapsed:.1f} pairs/s"
        )

    def _count_task_pairs(self, n: int, use_symmetric: bool) -> int:
        """生成需要计算的单元格对。"""
        if use_symmetric:
            return n * (n + 1) // 2
        return n * n

    def _generate_task_pairs(self, n: int, use_symmetric: bool) -> Iterator[Tuple[int, int]]:
        """鐢熸垚闇€瑕佽绠楃殑鍗曞厓鏍煎銆?"""
        if use_symmetric:
            for i in range(n):
                for j in range(i, n):
                    yield (i, j)
        else:
            for i in range(n):
                for j in range(n):
                    yield (i, j)

    def _generate_task_chunks(
        self,
        n: int,
        use_symmetric: bool,
        chunk_size: int
    ) -> Iterator[List[Tuple[int, int]]]:
        """鎸夊潡鐢熸垚闇€瑕佽绠楃殑鍗曞厓鏍煎銆?"""
        chunk: List[Tuple[int, int]] = []
        for pair in self._generate_task_pairs(n, use_symmetric):
            chunk.append(pair)
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

    def _compute_sequential(
        self,
        n: int,
        all_cells: List[QuadTreeCell],
        use_symmetric: bool,
        show_progress: bool
    ) -> None:
        """顺序计算相似度矩阵。"""
        pairs = self._generate_task_pairs(n, use_symmetric)
        total_pairs = self._count_task_pairs(n, use_symmetric)
        for i, j in tqdm(pairs, total=total_pairs, disable=not show_progress, desc="顺序计算"):
            sim = self.cost_evaluator.jaccard_similarity(all_cells[i], all_cells[j], use_cache=False)
            self.matrix_array[i, j] = sim
            if use_symmetric:
                self.matrix_array[j, i] = sim

    def _compute_parallel(
        self,
        n: int,
        all_cells: List[QuadTreeCell],
        use_symmetric: bool,
        show_progress: bool,
        num_workers: int,
        chunk_size: int,
        total_pairs: int
    ) -> None:
        """并行计算相似度矩阵。"""
        mp_context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=mp_context,
            initializer=_init_similarity_worker,
            initargs=(self.cost_evaluator, all_cells),
        ) as executor:
            chunk_iter = self._generate_task_chunks(n, use_symmetric, chunk_size)
            pending: Deque = deque()

            for _ in range(max(1, num_workers * 2)):
                try:
                    chunk = next(chunk_iter)
                except StopIteration:
                    break
                pending.append(executor.submit(_compute_chunk_task, chunk))

            with tqdm(total=total_pairs, disable=not show_progress, desc="并行计算") as pbar:
                while pending:
                    done, _ = wait(list(pending), return_when=FIRST_COMPLETED)
                    for future in done:
                        pending.remove(future)
                        chunk_results = future.result()
                        for i, j, sim in chunk_results:
                            self.matrix_array[i, j] = sim
                            if use_symmetric:
                                self.matrix_array[j, i] = sim
                        pbar.update(len(chunk_results))

                        try:
                            chunk = next(chunk_iter)
                        except StopIteration:
                            continue
                        pending.append(executor.submit(_compute_chunk_task, chunk))

    def get_similarity(self, cell_a: QuadTreeCell, cell_b: QuadTreeCell) -> float:
        """获取相似度，支持 O(1) 矩阵查询。"""
        if not self.is_computed:
            return self.cost_evaluator.jaccard_similarity(cell_a, cell_b)

        idx_a = self.cell_to_index.get(cell_a)
        idx_b = self.cell_to_index.get(cell_b)

        if idx_a is None or idx_b is None:
            return self.cost_evaluator.jaccard_similarity(cell_a, cell_b)

        return float(self.matrix_array[idx_a, idx_b])

    def save(self, filepath: str) -> None:
        """保存高效的 .npz 格式数据。
        
        参数:
            filepath: 保存路径
            
        异常:
            RuntimeError: 如果矩阵尚未计算
        """
        if not self.is_computed or self.matrix_array is None:
            raise RuntimeError("矩阵尚未计算")

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # 提取单元格唯一标识用于验证
        cell_ids = np.array([
            ",".join(map(str, c.quadrant_sequence))
            for c in self.all_cells
        ], dtype=str)

        np.savez_compressed(
            filepath,
            matrix=self.matrix_array,
            cell_ids=cell_ids
        )
        
        size_mb = self.matrix_array.nbytes / 1024 ** 2
        self.logger.info(f"矩阵已压缩保存至: {filepath} ({size_mb:.2f} MB)")

    def load(self, filepath: str, all_cells: List[QuadTreeCell]) -> bool:
        """从文件加载并执行严格的单元格顺序校验。
        
        参数:
            filepath: 加载路径
            all_cells: 当前单元格列表
            
        返回:
            加载是否成功
        """
        if not os.path.exists(filepath):
            return False

        try:
            data = np.load(filepath, allow_pickle=True)
            loaded_matrix = data['matrix']
            loaded_ids = data['cell_ids']

            # 验证维度
            if len(all_cells) != len(loaded_ids):
                self.logger.warning("加载失败：单元格数量不匹配")
                return False

            # 验证序列一致性
            current_ids = [
                ",".join(map(str, c.quadrant_sequence))
                for c in all_cells
            ]
            if not np.array_equal(loaded_ids, current_ids):
                self.logger.warning("加载失败：单元格标识顺序不匹配")
                return False

            self.all_cells = all_cells
            self.cell_to_index = {cell: idx for idx, cell in enumerate(all_cells)}
            self.matrix_array = loaded_matrix
            self.is_computed = True
            return True
            
        except Exception as e:
            self.logger.error(f"加载矩阵异常: {e}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """利用 NumPy 高效计算统计信息。"""
        if not self.is_computed or self.matrix_array is None:
            return {"status": "Not Computed"}

        return {
            "mean": float(np.mean(self.matrix_array)),
            "std": float(np.std(self.matrix_array)),
            "max": float(np.max(self.matrix_array)),
            "sparsity": float(np.mean(self.matrix_array == 0)),
            "memory_mb": self.matrix_array.nbytes / 1024 ** 2
        }

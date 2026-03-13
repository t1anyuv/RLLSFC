from typing import List, Tuple

import numpy as np

from src.core.bounding_box import SpatialBoundingBox
from src.indexing.quadtree_cell import QuadTreeCell
from src.utils.signature import compute_signature_vectorized


def evaluate(alpha: int, beta: int, traj_points_list: List[np.ndarray],
             ee_bbox: SpatialBoundingBox, max_sample_num: int = 100) -> float:
    """
    评估函数: Score = Entropy(p) + AvgHammingDist / (alpha*beta)
    """
    if not traj_points_list:
        return -1.0

    sigs = [compute_signature_vectorized(alpha, beta, p, ee_bbox) for p in traj_points_list]
    total_bits = alpha * beta

    # 1. 计算填充率 p 和熵 H
    ones_count = sum(bin(s).count('1') for s in sigs)  # 统计所有位图中 1 的总数
    p = ones_count / (len(sigs) * total_bits + 1e-9)
    p = np.clip(p, 0.01, 0.99)
    entropy = -p * np.log2(p) - (1 - p) * np.log2(1 - p)

    # 2. 计算归一化汉明距离，比较区分度
    if len(sigs) > 1:
        # 随机采样对进行比较
        num_samples = min(max_sample_num, len(sigs) * (len(sigs) - 1) // 2)
        hamming_dists = []
        for _ in range(num_samples):
            s1, s2 = np.random.choice(sigs, 2, replace=False)
            hamming_dists.append(bin(s1 ^ s2).count('1'))
        avg_hamming = np.mean(hamming_dists) / total_bits
    else:
        avg_hamming = 0.0

    return entropy + avg_hamming


class SignatureOptimizer:
    def __init__(self, alpha_range=(2, 6), beta_range=(2, 6)):
        self.alpha_range = range(alpha_range[0], alpha_range[1] + 1)
        self.beta_range = range(beta_range[0], beta_range[1] + 1)

    def find_best_config(self, global_alpha: int, global_beta: int,
                         traj_points_list: List[np.ndarray], cell: QuadTreeCell) -> Tuple[int, int]:
        """为单个 Cell 寻找最优 alpha, beta"""

        ee = cell.get_enlarged_element_bbox(global_alpha, global_beta)
        best_cfg = (global_alpha, global_beta)
        best_score = evaluate(global_alpha, global_beta, traj_points_list, ee)

        for a in self.alpha_range:
            for b in self.beta_range:
                if a == global_alpha and b == global_beta:
                    continue
                score = evaluate(a, b, traj_points_list, ee)

                if score > best_score:
                    best_score = score
                    best_cfg = (a, b)
        return best_cfg

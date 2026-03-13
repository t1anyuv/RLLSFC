"""经验回放缓冲区，用于存储和管理强化学习交互数据。"""
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch


@dataclass
class Transition:
    """单步交互经验。"""
    state: np.ndarray
    action: int
    reward: float
    log_prob: torch.Tensor
    value: torch.Tensor
    mask: np.ndarray
    done: bool


class ReplayBuffer:
    """经验回放缓冲区，用于存储和批量获取训练数据。"""

    def __init__(self):
        self.transitions: List[Transition] = []

    def store(self, transition: Transition) -> None:
        """存储单步经验。"""
        self.transitions.append(transition)

    def get_all(self) -> List[Transition]:
        """获取所有存储的经验。"""
        return self.transitions

    def clear(self) -> None:
        """清空缓冲区。"""
        self.transitions.clear()

    def __len__(self) -> int:
        return len(self.transitions)

    def is_empty(self) -> bool:
        return len(self.transitions) == 0

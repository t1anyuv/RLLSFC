from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from src.rl.replay_buffer import ReplayBuffer, Transition
from src.rl.ppo_updater import PPOUpdater


class TraversalActorNetwork(nn.Module):
    """
    策略网络（Actor），用于根据输入状态生成每个动作的概率分布（未归一化的 logits）。

    架构说明：
        输入：状态向量 (state)
        输出：动作 logits，用于计算 softmax 概率分布。

    参数：
        state_dim : int
            状态特征维度。
        action_dim : int
            可用动作数量。
        hidden_dims : List[int], optional
            隐藏层神经元数量列表（默认：[256, 256]）。
        dropout_rate : float, optional
            Dropout 比率（默认：0.1）。
    """

    # 默认超参数
    DEFAULT_DROPOUT_RATE = 0.1
    MASKED_LOGIT_PENALTY = 1e9

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = None,
        dropout_rate: float = None
    ):
        super().__init__()
        hidden_dims = hidden_dims or [256, 256]
        dropout_rate = dropout_rate if dropout_rate is not None else self.DEFAULT_DROPOUT_RATE
        
        layers: List[nn.Module] = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, action_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向传播，输出每个动作的 logits。

        参数：
            state : torch.Tensor
                状态输入张量，形状为 (batch_size, state_dim)。

        返回：
            torch.Tensor
                动作 logits，形状为 (batch_size, action_dim)。
        """
        return self.network(state)

    def select_action(self, state: torch.Tensor, action_mask: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """
        从策略分布中采样一个合法动作。

        逻辑说明：
            1. 对输入状态计算 logits；
            2. 根据动作掩码屏蔽非法动作；
            3. 对剩余动作进行 softmax 得到概率分布；
            4. 从分布中采样动作。

        参数：
            state : torch.Tensor
                当前状态张量，形状 (1, state_dim)。
            action_mask : torch.Tensor
                动作掩码（1 表示可选动作，0 表示禁用动作）。

        返回：
            Tuple[int, torch.Tensor]
                - 采样得到的动作索引；
                - 对应的对数概率（log probability）。
        """
        logits = self.forward(state)
        masked_logits = logits + (action_mask - 1.0) * self.MASKED_LOGIT_PENALTY
        probabilities = F.softmax(masked_logits, dim=-1)

        distribution = torch.distributions.Categorical(probabilities)
        action = distribution.sample()

        return action.item(), distribution.log_prob(action)


class TraversalCriticNetwork(nn.Module):
    """
    价值网络（Critic），用于估计输入状态的期望回报（即 V(s)）。

    参数：
        state_dim : int
            状态特征维度。
        hidden_dims : List[int], optional
            隐藏层神经元数量列表（默认：[256, 256]）。
    """

    def __init__(self, state_dim: int, hidden_dims: List[int] = None):
        super().__init__()
        hidden_dims = hidden_dims or [256, 256]

        layers: List[nn.Module] = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向传播，输出当前状态的价值估计。

        参数：
            state : torch.Tensor
                状态输入张量，形状为 (batch_size, state_dim)。

        返回：
            torch.Tensor
                状态价值估计，形状为 (batch_size, 1)。
        """
        return self.network(state)


class TraversalPolicyAgent:
    """
    Actor–Critic agent responsible for learning traversal policies.

    基于 PPO (Proximal Policy Optimization) 算法的智能体，用于在空间遍历任务中学习最优动作策略。

    模块结构：
        - Actor：策略网络，用于输出动作分布；
        - Critic：价值网络，用于评估状态价值；
        - Optimizer：分别对 actor / critic 使用独立的 Adam 优化器。
        - ReplayBuffer：经验回放缓冲区。
        - PPOUpdater：PPO 更新逻辑封装（支持 GAE）。

    参数：
        state_dim : int
            状态特征维度。
        action_dim : int
            动作空间维度。
        lr_actor : float, default = 3e-4
            策略网络（Actor）的学习率。
        lr_critic : float, default = 3e-4
            价值网络（Critic）的学习率。
        eps_clip: float, default = 0.2
            PPO 裁剪阈值。
        k_epochs: int, default = 4
            每轮交互数据重复学习的次数。
        gamma : float, default = 0.99
            折扣因子。
        entropy_coef: float, default = 0.01
            熵系数，用于鼓励探索。
        gae_lambda: float, default = 0.95
            GAE lambda 参数，用于平衡偏差和方差。
        gradient_clip_norm: float, default = 1.0
            梯度裁剪阈值。
        device : str, default = "cpu"
            计算设备（"cpu" 或 "cuda"）。
        hidden_dims : List[int], optional
            隐藏层结构（默认：[256, 256]）。
        dropout_rate : float, optional
            Dropout 比率（默认：0.1）。
    """

    # 学习率调度器常量
    LR_SCHEDULER_T_MAX = 100
    LR_SCHEDULER_ETA_MIN = 1e-6

    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            lr_actor: float = 3e-4,
            lr_critic: float = 3e-4,
            eps_clip: float = 0.2,
            k_epochs: int = 4,
            gamma: float = 0.99,
            entropy_coef: float = 0.01,
            gae_lambda: float = 0.95,
            gradient_clip_norm: float = 1.0,
            device: str = "cpu",
            hidden_dims: List[int] = None,
            dropout_rate: float = None,
    ):
        self.device = torch.device(device)
        self.action_dim = action_dim

        # 1. 初始化网络架构
        hidden_dims = hidden_dims or [256, 256]
        self.actor = TraversalActorNetwork(
            state_dim, action_dim, hidden_dims, dropout_rate
        ).to(self.device)
        self.critic = TraversalCriticNetwork(state_dim, hidden_dims).to(self.device)

        # 2. 独立优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        # 3. 学习率调度器
        self.actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.actor_optimizer, T_max=self.LR_SCHEDULER_T_MAX, eta_min=self.LR_SCHEDULER_ETA_MIN
        )
        self.critic_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.critic_optimizer, T_max=self.LR_SCHEDULER_T_MAX, eta_min=self.LR_SCHEDULER_ETA_MIN
        )

        # 4. 经验缓冲区
        self.replay_buffer = ReplayBuffer()

        # 5. PPO 更新器
        self.ppo_updater = PPOUpdater(
            actor=self.actor,
            critic=self.critic,
            actor_optimizer=self.actor_optimizer,
            critic_optimizer=self.critic_optimizer,
            gamma=gamma,
            eps_clip=eps_clip,
            k_epochs=k_epochs,
            entropy_coef=entropy_coef,
            device=self.device,
            gae_lambda=gae_lambda,
            gradient_clip_norm=gradient_clip_norm
        )

    def select_action(self, state: np.ndarray, action_mask: np.ndarray) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        根据当前状态选择动作，并返回动作、对数概率与状态价值。

        参数：
            state : np.ndarray
                当前环境状态向量。
            action_mask : np.ndarray
                合法动作掩码（非法动作位置为 0）。

        返回：
            Tuple[int, torch.Tensor, torch.Tensor]
                - action: 选定的动作索引；
                - log_prob: 该动作对应的对数概率；
                - value: 当前状态的价值估计。
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mask_tensor = torch.FloatTensor(action_mask).unsqueeze(0).to(self.device)

        with torch.no_grad():
            value = self.critic(state_tensor)
            action, log_prob = self.actor.select_action(state_tensor, mask_tensor)
        return action, log_prob, value

    def store_transition(
            self,
            state: np.ndarray,
            action: int,
            reward: float,
            log_prob: torch.Tensor,
            value: torch.Tensor,
            mask: np.ndarray,
            done: bool,
    ) -> None:
        """
        将一次交互的经验（状态、动作、奖励等）存入缓存，用于后续更新。

        参数：
            state : np.ndarray
                当前状态向量。
            action : int
                当前执行的动作索引。
            reward : float
                执行动作后获得的奖励。
            log_prob : torch.Tensor
                该动作的对数概率，用于策略梯度计算。
            value : torch.Tensor
                当前状态的价值估计。
            mask : np.ndarray
                动作掩码。
            done : bool
                是否为终止状态。
        """
        transition = Transition(
            state=state,
            action=action,
            reward=reward,
            log_prob=log_prob,
            value=value,
            mask=mask,
            done=done
        )
        self.replay_buffer.store(transition)

    def update(self) -> Dict[str, float]:
        """
        使用存储的交互序列更新 Actor-Critic 网络。
        
        返回:
            包含损失信息的字典
        """
        if self.replay_buffer.is_empty():
            return {}

        # 执行 PPO 更新
        loss_info = self.ppo_updater.update(self.replay_buffer.get_all())

        # 更新学习率
        self.actor_scheduler.step()
        self.critic_scheduler.step()

        # 清空缓冲区
        self.replay_buffer.clear()

        return loss_info

    def save(self, filepath: str) -> None:
        """保存模型状态、优化器及调度器。"""
        torch.save({
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_opt": self.actor_optimizer.state_dict(),
            "critic_opt": self.critic_optimizer.state_dict(),
            "actor_sched": self.actor_scheduler.state_dict(),
            "critic_sched": self.critic_scheduler.state_dict()
        }, filepath)

    def load(self, filepath: str) -> None:
        """从指定文件加载模型与优化器参数。"""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_opt"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_opt"])
        self.actor_scheduler.load_state_dict(checkpoint["actor_sched"])
        self.critic_scheduler.load_state_dict(checkpoint["critic_sched"])

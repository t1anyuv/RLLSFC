"""PPO 算法更新逻辑封装。"""
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from src.rl.replay_buffer import Transition


class PPOUpdater:
    """PPO (Proximal Policy Optimization) 更新器。
    
    封装 PPO 算法的核心更新逻辑，包括回报计算、优势函数计算和策略更新。
    支持 GAE (Generalized Advantage Estimation) 用于更稳定的优势估计。
    """

    # 常量定义
    ADVANTAGE_EPSILON = 1e-8

    def __init__(
        self,
        actor,
        critic,
        actor_optimizer,
        critic_optimizer,
        gamma: float,
        eps_clip: float,
        k_epochs: int,
        entropy_coef: float,
        device: torch.device,
        gae_lambda: float = 0.95,
        gradient_clip_norm: float = 1.0
    ):
        self.actor = actor
        self.critic = critic
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.entropy_coef = entropy_coef
        self.device = device
        self.gae_lambda = gae_lambda
        self.gradient_clip_norm = gradient_clip_norm

    def compute_gae_advantages(
        self,
        rewards: List[float],
        values: torch.Tensor,
        dones: List[bool]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """使用 GAE (Generalized Advantage Estimation) 计算优势函数。
        
        GAE 通过 lambda 参数平衡偏差和方差：
        - lambda=0: 高偏差，低方差（TD(0)）
        - lambda=1: 低偏差，高方差（Monte Carlo）
        - lambda=0.95: 通常的最佳平衡点
        
        参数:
            rewards: 奖励序列
            values: 状态价值估计序列
            dones: 终止标志序列
            
        返回:
            (advantages, returns) 元组
            - advantages: 标准化后的优势函数
            - returns: 折扣回报（用于训练 Critic）
        """
        advantages = []
        gae = 0.0
        
        values_list = values.cpu().detach().numpy().tolist()
        
        # 从后向前计算 GAE
        for t in reversed(range(len(rewards))):
            # 确定下一个状态的价值
            if t == len(rewards) - 1:
                # 最后一步：如果done=True则next_value=0，否则用当前critic估计
                # 注意：在你的训练循环中，每个episode结束后才update，所以最后一步必然done=True
                next_value = 0.0 if dones[t] else values_list[t]
            else:
                next_value = values_list[t + 1]
            
            # TD误差: δ_t = r_t + γ * V(s_{t+1}) * (1 - done_t) - V(s_t)
            # 当done=True时，(1-done)=0，所以next_value被清零
            delta = rewards[t] + self.gamma * next_value * (1.0 - float(dones[t])) - values_list[t]
            
            # GAE: A_t = δ_t + (γλ) * (1 - done_t) * A_{t+1}
            # 当done=True时，GAE链条被截断，不会传播到前面的步骤
            gae = delta + self.gamma * self.gae_lambda * (1.0 - float(dones[t])) * gae
            advantages.insert(0, gae)
        
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # 标准化优势函数（减少方差）
        advantages = (advantages - advantages.mean()) / (advantages.std() + self.ADVANTAGE_EPSILON)
        
        # 计算回报: R_t = A_t + V(s_t)
        returns = advantages + values
        
        return advantages, returns

    def update(self, transitions: List[Transition]) -> Dict[str, float]:
        """执行 PPO 更新。
        
        参数:
            transitions: 经验回放缓冲区中的交互数据
            
        返回:
            包含损失信息的字典
        """
        if not transitions:
            return {}

        # 1. 数据转换
        old_states = torch.FloatTensor(np.array([t.state for t in transitions])).to(self.device)
        old_actions = torch.LongTensor([t.action for t in transitions]).to(self.device)
        old_log_probs = torch.stack([t.log_prob for t in transitions]).detach().to(self.device)
        old_values = torch.stack([t.value for t in transitions]).squeeze().detach().to(self.device)
        old_masks = torch.FloatTensor(np.array([t.mask for t in transitions])).to(self.device)

        # 2. 使用 GAE 计算优势函数和回报
        rewards = [t.reward for t in transitions]
        dones = [t.done for t in transitions]
        advantages, returns = self.compute_gae_advantages(rewards, old_values, dones)

        # 3. PPO 多轮更新
        total_actor_loss = 0.0
        total_critic_loss = 0.0

        for _ in range(self.k_epochs):
            # 重新计算动作概率
            logits = self.actor(old_states)
            masked_logits = logits + (old_masks - 1.0) * 1e9
            new_probs = F.softmax(masked_logits, dim=-1)

            dist = Categorical(new_probs)
            new_log_probs = dist.log_prob(old_actions)
            entropy = dist.entropy().mean()

            # 获取新的状态价值
            new_values = self.critic(old_states).squeeze()

            # 计算重要性采样比率
            ratios = torch.exp(new_log_probs - old_log_probs)

            # PPO 裁剪目标
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # Actor 损失：策略梯度 + 熵正则化
            actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy

            # Critic 损失：均方误差
            critic_loss = F.mse_loss(new_values, returns)

            # 更新 Actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.gradient_clip_norm)
            self.actor_optimizer.step()

            # 更新 Critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.gradient_clip_norm)
            self.critic_optimizer.step()

            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()

        return {
            "loss": (total_actor_loss + total_critic_loss) / self.k_epochs,
            "actor_loss": total_actor_loss / self.k_epochs,
            "critic_loss": total_critic_loss / self.k_epochs
        }

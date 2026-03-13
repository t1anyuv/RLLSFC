"""训练状态管理。"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TrainingState:
    """训练过程状态跟踪。
    
    集中管理训练过程中的所有状态变量，避免在主类中分散管理。
    """
    
    # 训练指标
    episode_rewards: List[float] = field(default_factory=list)
    """每个 episode 的总奖励"""
    
    episode_lengths: List[int] = field(default_factory=list)
    """每个 episode 的步数"""
    
    # 评估指标（扩展为三类）
    train_improvement_history: List[float] = field(default_factory=list)
    """每次评估的训练集改进率历史记录"""
    
    val_improvement_history: List[float] = field(default_factory=list)
    """每次评估的验证集改进率历史记录"""
    
    test_improvement_history: List[float] = field(default_factory=list)
    """每次评估的测试集改进率历史记录"""
    
    loss_history: List[float] = field(default_factory=list)
    """每个 episode 更新时的损失函数值"""
    
    improvement_episodes: List[int] = field(default_factory=list)
    """对应每次评估发生时的 episode 编号"""
    
    # 早停相关
    best_improvement: float = float('-inf')
    """训练过程中达到的最高改进率百分比"""
    
    patience_counter: int = 0
    """当前的耐心消耗计数"""
    
    negative_streak_counter: int = 0
    """负收益率最大持续周期"""
    
    early_stop_episode: Optional[int] = None
    """若触发早停，记录触发时的 episode 编号；否则为 None"""
    
    def record_episode(self, reward: float, length: int, loss: Optional[float] = None) -> None:
        """记录单个 episode 的结果。
        
        参数:
            reward: episode 总奖励
            length: episode 步数
            loss: 损失值（可选）
        """
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        if loss is not None:
            self.loss_history.append(loss)
    
    def record_evaluation(self, episode: int, 
                          train_improvement: float,
                          val_improvement: float,
                          test_improvement: float) -> None:
        """记录评估结果。
        
        参数:
            episode: 当前 episode 编号
            train_improvement: 训练集改进率
            val_improvement: 验证集改进率
            test_improvement: 测试集改进率
        """
        self.improvement_episodes.append(episode)
        
        # 记录三类数据集改进率
        self.train_improvement_history.append(train_improvement if train_improvement is not None else 0.0)
        self.val_improvement_history.append(val_improvement if val_improvement is not None else 0.0)
        self.test_improvement_history.append(test_improvement if test_improvement is not None else 0.0)
    
    def update_best_improvement(self, improvement: float) -> bool:
        """更新最佳改进率。
        
        参数:
            improvement: 当前改进率
            
        返回:
            是否刷新了最佳记录
        """
        if improvement > self.best_improvement:
            self.best_improvement = improvement
            self.patience_counter = 0
            return True
        else:
            self.patience_counter += 1
            return False
    
    def update_negative_streak(self, improvement: float) -> None:
        """更新负收益连续计数。
        
        参数:
            improvement: 当前改进率
        """
        if improvement < 0:
            self.negative_streak_counter += 1
        else:
            self.negative_streak_counter = 0
    
    def get_recent_avg_reward(self, window: int = 10) -> float:
        """获取最近N轮的平均奖励。
        
        参数:
            window: 窗口大小
            
        返回:
            平均奖励
        """
        if not self.episode_rewards:
            return 0.0
        recent = self.episode_rewards[-window:]
        return sum(recent) / len(recent)

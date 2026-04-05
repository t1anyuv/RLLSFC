import torch

from src.rl.ppo_updater import PPOUpdater


class _TinyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


def _build_updater() -> PPOUpdater:
    actor = _TinyNet()
    critic = _TinyNet()
    actor_opt = torch.optim.Adam(actor.parameters(), lr=1e-3)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=1e-3)
    return PPOUpdater(
        actor=actor,
        critic=critic,
        actor_optimizer=actor_opt,
        critic_optimizer=critic_opt,
        gamma=0.99,
        eps_clip=0.2,
        k_epochs=1,
        entropy_coef=0.0,
        device=torch.device("cpu"),
        gae_lambda=0.95,
    )


def test_compute_gae_advantages_keeps_returns_unscaled():
    updater = _build_updater()
    rewards = [1.0, 2.0]
    values = torch.tensor([0.5, 0.25], dtype=torch.float32)
    dones = [False, True]

    advantages, returns = updater.compute_gae_advantages(rewards, values, dones)

    expected_returns = torch.tensor([2.893375, 2.0], dtype=torch.float32)

    assert torch.allclose(returns, expected_returns, atol=1e-5)
    assert abs(float(advantages.mean())) < 1e-6
    assert float(advantages.std()) > 0

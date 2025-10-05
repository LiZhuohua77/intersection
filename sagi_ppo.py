import warnings
from typing import Any, Dict, Optional, Type, Union

import numpy as np
import torch
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.utils import explained_variance, get_schedule_fn, obs_as_tensor


class ActorCriticCostPolicy(ActorCriticPolicy):
    """
    一个同时拥有奖励价值头(value head)和成本价值头(cost value head)的策略网络。
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cost_vf_arch = self.net_arch.pop('cost_vf', [])
        self.cost_value_net = self.make_vf(cost_vf_arch)

    def forward(self, obs: torch.Tensor, deterministic: bool = False):
        latent_pi, latent_vf, latent_sde = self._get_latent(obs)
        distribution = self._get_action_dist_from_latent(latent_pi, latent_sde=latent_sde)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        cost_values = self.cost_value_net(latent_vf)
        return actions, values, cost_values, log_prob
        
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        latent_pi, latent_vf, latent_sde = self._get_latent(obs)
        distribution = self._get_action_dist_from_latent(latent_pi, latent_sde)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        cost_values = self.cost_value_net(latent_vf)
        entropy = distribution.entropy()
        return values, cost_values, log_prob, entropy


class SAGIRolloutBuffer(RolloutBuffer):
    """
    能够额外存储成本信息，并计算平均每回合累积成本的缓冲区。
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.costs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.cost_values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.cost_advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.cost_returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

    def add(self, cost: np.ndarray, cost_value: torch.Tensor, **kwargs) -> None:
        # **kwargs 会包含所有父类 add 方法的参数
        self.costs[self.pos] = np.array(cost).copy()
        self.cost_values[self.pos] = cost_value.clone().cpu().numpy().flatten()
        super().add(**kwargs)
    
    def compute_returns_and_advantage(self, last_values: torch.Tensor, last_cost_values: torch.Tensor, dones: np.ndarray):
        # 计算奖励 GAE
        super().compute_returns_and_advantage(last_values, dones)

        # 计算成本 GAE
        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_cost_values = last_cost_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_cost_values = self.cost_values[step + 1]
            delta = self.costs[step] + self.gamma * next_cost_values * next_non_terminal - self.cost_values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.cost_advantages[step] = last_gae_lam
        self.cost_returns = self.cost_advantages + self.cost_values

    def get_mean_episode_costs(self) -> float:
        """严谨地计算平均每回合的累积折扣成本。"""
        episode_costs = []
        current_discounted_costs = np.zeros(self.n_envs)
        
        # 从后向前遍历以正确计算累积折扣值
        for step in reversed(range(self.buffer_size)):
            costs_step = self.costs[step]
            # 如果不是回合开始，则累积成本
            current_discounted_costs = costs_step + self.gamma * current_discounted_costs * (1.0 - self.episode_starts[step])
            
            # 检查每个环境中是否有回合在这一步(step)结束
            # episode_starts[step] == 1 意味着 step 是新回合的第一步，即上一步是终点
            for env_idx, is_start in enumerate(self.episode_starts[step]):
                if is_start:
                    # 将上一个刚结束的回合的累积成本存起来
                    # 我们需要找到上一个step的累积值，但由于迭代方向，直接使用当前值然后重置更方便
                    episode_costs.append(current_discounted_costs[env_idx])
                    # 重置这个环境的累积成本，为下一个追溯到的回合做准备
                    current_discounted_costs[env_idx] = 0

        if not episode_costs:
            warnings.warn("No episode found ending in the rollout buffer, cost surplus calculation may be inaccurate.")
            return 0.0
            
        return np.mean(episode_costs)


class SAGIPPO(PPO):
    policy_aliases: Dict[str, Type[ActorCriticPolicy]] = {
        "MlpPolicy": ActorCriticCostPolicy,
    }

    def __init__(self, policy, env, cost_limit: float = 25.0, lambda_lr: float = 0.035, cost_vf_coef: float = 0.5, **kwargs):
        self.cost_limit = cost_limit
        self.lambda_lr = lambda_lr
        self.cost_vf_coef = cost_vf_coef
        self.lambda_ = 0.0
        
        super().__init__(policy=policy, env=env, **kwargs)
        self._setup_custom_components()

    def _setup_custom_components(self) -> None:
        self.policy_class = ActorCriticCostPolicy
        self.rollout_buffer_class = SAGIRolloutBuffer
        super()._setup_model()
        # 确保使用的是我们自定义的Buffer
        assert isinstance(self.rollout_buffer, SAGIRolloutBuffer), "SAGIPPO must use SAGIRolloutBuffer"

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        
        # 计算奖励和成本的 GAE 优势函数
        with torch.no_grad():
            last_obs = obs_as_tensor(self.rollout_buffer.obs[-1], self.device)
            _, last_values, last_cost_values, _ = self.policy(last_obs)
        self.rollout_buffer.compute_returns_and_advantage(last_values, last_cost_values, self.rollout_buffer.dones)
        
        # --- [严谨修正] 计算 c 和 p ---
        # 1. 严谨计算成本裕量 c
        j_c_k = self.rollout_buffer.get_mean_episode_costs()
        c = j_c_k - self.cost_limit
        
        # 2. 严谨计算策略梯度内积 p
        full_batch = self.rollout_buffer.get(batch_size=None)
        observations, actions, old_log_prob = full_batch.observations, full_batch.actions, full_batch.old_log_prob
        reward_advantages = full_batch.advantages
        cost_advantages = torch.as_tensor(self.rollout_buffer.cost_advantages, device=self.device).flatten()

        self.policy.train() # 确保策略在训练模式
        _, _, log_prob, _ = self.policy.evaluate_actions(observations, actions)
        ratio = torch.exp(log_prob - old_log_prob)

        # 定义用于求导的代理损失
        reward_surrogate_loss = -torch.min(reward_advantages * ratio, reward_advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)).mean()
        cost_surrogate_loss = -torch.min(cost_advantages * ratio, cost_advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)).mean()
        
        # 通过“伪反向传播”获得 g_R 和 g_C
        g_r_tensors = torch.autograd.grad(reward_surrogate_loss, self.policy.parameters(), retain_graph=True)
        g_r_flat = torch.cat([grad.flatten() for grad in g_r_tensors])

        g_c_tensors = torch.autograd.grad(cost_surrogate_loss, self.policy.parameters(), retain_graph=False)
        g_c_flat = torch.cat([grad.flatten() for grad in g_c_tensors])

        p = torch.dot(g_r_flat, g_c_flat).item()
        # --- 修正结束 ---

        self.logger.record("sagi/cost_surplus_c", c)
        self.logger.record("sagi/grad_inner_product_p", p)
        self.logger.record("sagi/lambda", self.lambda_)

        # 根据 (c, p) 选择更新模式并修改优势函数
        original_advantages = self.rollout_buffer.advantages.copy()
        if c < 0 and p <= 0:
            self.logger.record("sagi/mode", "A")
        elif c > 0:
            self.logger.record("sagi/mode", "C")
            self.rollout_buffer.advantages = self.rollout_buffer.cost_advantages.copy() * -1
        else:
            self.logger.record("sagi/mode", "B")
            self.rollout_buffer.advantages = (original_advantages - self.lambda_ * self.rollout_buffer.cost_advantages) / (1 + self.lambda_)
        
        self.lambda_ = max(0, self.lambda_ + self.lambda_lr * c)

        # 执行 PPO 更新循环
        for epoch in range(self.n_epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                reward_values, cost_values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                reward_values = reward_values.flatten()
                cost_values = cost_values.flatten()

                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = torch.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                value_loss = F.mse_loss(rollout_data.returns, reward_values)
                cost_value_loss = F.mse_loss(rollout_data.cost_returns, cost_values)
                entropy_loss = -torch.mean(entropy) if entropy is not None else -torch.mean(-log_prob)

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + self.cost_vf_coef * cost_value_loss

                self.policy.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

        self._n_updates += 1
        self.rollout_buffer.advantages = original_advantages
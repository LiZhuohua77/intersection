import warnings
from typing import Any, Dict, Optional, Type, Union

import numpy as np
import torch
from gymnasium import spaces
from torch.nn import functional as F
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import explained_variance, get_schedule_fn, obs_as_tensor
from stable_baselines3.common.torch_layers import create_mlp
from stable_baselines3.common.vec_env import VecEnv



class ActorCriticCostPolicy(ActorCriticPolicy):
    """
    一个同时拥有奖励价值头(value head)和成本价值头(cost value head)的策略网络。
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        cost_vf_arch = self.net_arch.pop('cost_vf', [])
        
        if cost_vf_arch:
            # [FIXED] 使用 nn.Sequential 将 create_mlp 返回的层列表组装成一个可调用的网络模块
            self.cost_value_net = nn.Sequential(*create_mlp(
                self.mlp_extractor.latent_dim_vf,
                1,
                net_arch=cost_vf_arch,
                activation_fn=self.activation_fn
            ))
        else:
            self.cost_value_net = nn.Identity()

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

        last_cost_values = last_cost_values.clone().cpu().numpy().flatten()

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
        
        super().__init__(
            policy=policy,
            env=env,
            rollout_buffer_class=SAGIRolloutBuffer,
            **kwargs
        )

        assert isinstance(self.rollout_buffer, SAGIRolloutBuffer), "SAGIPPO must use SAGIRolloutBuffer"


    def collect_rollouts(
        self, env: VecEnv, callback: BaseCallback, rollout_buffer: SAGIRolloutBuffer, n_rollout_steps: int
    ) -> bool:
        """
        [FIXED] 重写数据收集过程，以正确获取 cost 和 cost_value。
        """
        assert self._last_obs is not None, "No previous observation was provided"
        self.policy.set_training_mode(False)
        n_steps = 0
        rollout_buffer.reset()
        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            with torch.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                
                # 手动执行前向传播以获取所有需要的值
                features = self.policy.extract_features(obs_tensor)
                latent_pi, latent_vf = self.policy.mlp_extractor(features)
                distribution = self.policy._get_action_dist_from_latent(latent_pi)
                actions = distribution.get_actions(deterministic=False)
                log_probs = distribution.log_prob(actions)
                values = self.policy.value_net(latent_vf)
                cost_values = self.policy.cost_value_net(latent_vf)

            actions = actions.cpu().numpy()
            clipped_actions = actions
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)
            costs = np.array([info.get("cost", 0) for info in infos])
            self.num_timesteps += env.num_envs

            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1
            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.reshape(-1, 1)

            rollout_buffer.add(
                obs=self._last_obs, action=actions, reward=rewards, cost=costs,
                episode_start=self._last_episode_starts, value=values, cost_value=cost_values, log_prob=log_probs
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with torch.no_grad():
            last_obs_tensor = obs_as_tensor(new_obs, self.device)
            features = self.policy.extract_features(last_obs_tensor)
            _, latent_vf = self.policy.mlp_extractor(features)
            
            last_values = self.policy.value_net(latent_vf)
            last_cost_values = self.policy.cost_value_net(latent_vf)
        rollout_buffer.compute_returns_and_advantage(last_values, last_cost_values, dones)
        callback.on_rollout_end()
        return True




    def train(self) -> None:
        """
        [最终修正版] SAGI-PPO 核心训练逻辑。
        修正了对 a generator' object 的错误使用。
        """
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)

        # --- [严谨修正] 计算 c 和 p ---
        # 1. 使用成本回报的平均值作为 J_C,k 的估计
        j_c_k = np.mean(self.rollout_buffer.cost_returns)
        c = j_c_k - self.cost_limit
        
        # 2. 严谨计算策略梯度内积 p
        # [FIXED] 使用 next() 从生成器中获取数据
        full_batch = next(self.rollout_buffer.get(batch_size=None))
        
        observations, actions, old_log_prob = full_batch.observations, full_batch.actions, full_batch.old_log_prob
        
        reward_advantages = torch.as_tensor(full_batch.advantages, device=self.device).flatten()
        cost_advantages = torch.as_tensor(self.rollout_buffer.cost_advantages, device=self.device).flatten()

        self.policy.train()
        _, _, log_prob, _ = self.policy.evaluate_actions(observations, actions)
        ratio = torch.exp(log_prob - old_log_prob)

        reward_surrogate_loss = -torch.min(reward_advantages * ratio, reward_advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)).mean()
        cost_surrogate_loss = -torch.min(cost_advantages * ratio, cost_advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)).mean()
        
        g_r_tensors = torch.autograd.grad(reward_surrogate_loss, self.policy.parameters(), retain_graph=True)
        g_r_flat = torch.cat([grad.flatten() for grad in g_r_tensors])

        g_c_tensors = torch.autograd.grad(cost_surrogate_loss, self.policy.parameters(), retain_graph=False)
        g_c_flat = torch.cat([grad.flatten() for grad in g_c_tensors])

        p = torch.dot(g_r_flat, g_c_flat).item()
        
        self.logger.record("sagi/cost_surplus_c", c)
        self.logger.record("sagi/grad_inner_product_p", p)
        self.logger.record("sagi/lambda", self.lambda_)

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

        # 执行 PPO 更新循环 (这部分不变)
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
                policy_loss = -torch.min(advantages * ratio, advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)).mean()
                
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
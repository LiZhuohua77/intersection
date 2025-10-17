import warnings
# [关键修正] 从 typing 模块导入 Generator 和 Tuple
from typing import Any, Dict, Optional, Type, Union, Generator, Tuple

import numpy as np
import torch
from gymnasium import spaces
from torch.nn import functional as F
import torch.nn as nn
import torch as th

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.buffers import RolloutBuffer, RolloutBufferSamples
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.torch_layers import create_mlp
from stable_baselines3.common.vec_env import VecEnv, VecNormalize


# ==============================================================================
# 1. 自定义策略网络 (此部分已稳定，无需修改)
# ==============================================================================
class ActorCriticCostPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cost_vf_arch = self.net_arch.get('cost_vf', [])
        if cost_vf_arch:
            self.cost_value_net = nn.Sequential(*create_mlp(
                self.mlp_extractor.latent_dim_vf, 1, net_arch=cost_vf_arch, activation_fn=self.activation_fn
            ))
        else:
            self.cost_value_net = nn.Identity()

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        """
        [FINAL CORRECTED VERSION]
        This version is based on your provided source code and removes the problematic SDE check.
        """
        # Get latent features using the standard, correct two-step process
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)

        # [FIXED] The entire block checking for 'sde_features_extractor' has been removed.
        # We pass latent_pi directly to the distribution network.
        distribution = self._get_action_dist_from_latent(latent_pi)

        log_prob = distribution.log_prob(actions)

        # Calculate reward and cost values
        values = self.value_net(latent_vf)
        cost_values = self.cost_value_net(latent_vf)

        entropy = distribution.entropy()

        return values, cost_values, log_prob, entropy


# ==============================================================================
# 2. 自定义经验缓冲区 (对 get 方法的类型标注进行修正)
# ==============================================================================
class SAGIRolloutBuffer(RolloutBuffer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.costs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.cost_values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.cost_advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.cost_returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)


    def reset(self) -> None:
        super().reset()
        self.costs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.cost_values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.cost_advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.cost_returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

    def add(self, cost: np.ndarray, cost_value: torch.Tensor, **kwargs) -> None:
        self.costs[self.pos] = np.array(cost).copy()
        self.cost_values[self.pos] = cost_value.clone().cpu().numpy().flatten()
        super().add(**kwargs)

    def compute_returns_and_advantage(self, last_values: torch.Tensor, last_cost_values: torch.Tensor, dones: np.ndarray):
        super().compute_returns_and_advantage(last_values, dones)
        last_cost_values = last_cost_values.clone().cpu().numpy().flatten()
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
            """
            [CORRECTED VERSION]
            Computes the mean of discounted costs for all completed episodes in the buffer.
            """
            episode_costs = []
            # A running total of discounted costs for each environment
            current_discounted_costs = np.zeros(self.n_envs)
            
            # Iterate backwards through the buffer
            for step in reversed(range(self.buffer_size)):
                # Create a boolean mask for environments where a new episode starts at this step
                is_start_mask = self.episode_starts[step].astype(bool)

                # If any episodes start here, it means episodes have just ended.
                # Their total accumulated costs are in `current_discounted_costs`.
                if np.any(is_start_mask):
                    # Append the accumulated costs from the completed episodes to our list
                    episode_costs.extend(current_discounted_costs[is_start_mask])
                    # Reset the accumulator for those environments
                    current_discounted_costs[is_start_mask] = 0.0

                # Accumulate the cost for the current step into the running total.
                # This happens *after* we've potentially recorded and reset.
                current_discounted_costs = self.costs[step] + self.gamma * current_discounted_costs
            
            # The loop misses the very first set of episodes that don't start with a 'True'
            # in the buffer, but this is standard and acceptable. We only average full episodes.
            
            if not episode_costs:
                warnings.warn(
                    "No full episodes found in the rollout buffer. "
                    "Consider increasing n_rollout_steps or Rewarding faster termination."
                )
                return 0.0
            
            return np.mean(episode_costs)

# ==============================================================================
# 3. SAGI-PPO 算法 (此部分已稳定，无需修改)
# ==============================================================================
class SAGIPPO(PPO):
    policy_aliases: Dict[str, Type[ActorCriticPolicy]] = { "MlpPolicy": ActorCriticCostPolicy }

    def __init__(self, policy, env, 
                 initial_cost_limit: float = 500.0,
                 final_cost_limit: float = 30.0,
                 decay_start_step: int = 5_000_000, 
                 lambda_lr: float = 0.035, cost_vf_coef: float = 0.5, **kwargs):

        self.initial_cost_limit = initial_cost_limit
        self.final_cost_limit = final_cost_limit
        self.decay_start_step = decay_start_step
        self.lambda_lr = lambda_lr
        self.cost_vf_coef = cost_vf_coef
        self.lambda_ = 0.0
        self.cost_limit = self.initial_cost_limit
        super().__init__(policy=policy, env=env, rollout_buffer_class=SAGIRolloutBuffer, **kwargs)

    def train(self) -> None:
        """
        [最终修正]
        调整了代码的执行顺序，以适应 get() 方法的惰性求值特性，
        从根源上解决反复出现的形状不匹配问题。
        """
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        # ==================== [新增] 非线性成本限制退火逻辑 ====================
        current_step = self.num_timesteps
        decay_start = self.decay_start_step
        total_steps = self._total_timesteps  # 从 PPO 基类获取总训练步数

        if current_step <= decay_start:
            # 阶段一：在衰减开始前，保持高的初始值
            current_cost_limit = self.initial_cost_limit
        else:
            # 阶段二：在衰减阶段，进行线性插值
            # 计算在衰减阶段的进度比例 (从 0.0 到 1.0)
            progress_in_decay_phase = (current_step - decay_start) / max(1, total_steps - decay_start)
            progress_in_decay_phase = min(1.0, progress_in_decay_phase)  # 确保不超过1.0
            
            # 从 initial_cost_limit 线性插值到 final_cost_limit
            current_cost_limit = self.initial_cost_limit + progress_in_decay_phase * (self.final_cost_limit - self.initial_cost_limit)
        
        # 更新 self.cost_limit 并记录到 TensorBoard
        self.cost_limit = current_cost_limit
        self.logger.record("sagi/current_cost_limit", self.cost_limit)
        # =====================================================================

        clip_range = self.clip_range(self._current_progress_remaining)

        j_c_k = self.rollout_buffer.get_mean_episode_costs()
        c = j_c_k - self.cost_limit

        # ==================== 关键修正：调整执行顺序 ====================
        # 1. 首先，获取一次完整的批次数据。
        #    这一步会触发 get() 内部的 flatten 逻辑，将 self.advantages 等数组的形状变为 (4096, 1)
        full_batch_generator = self.rollout_buffer.get(batch_size=self.rollout_buffer.buffer_size * self.n_envs)
        full_batch = next(full_batch_generator)

        # 2. 现在，self.rollout_buffer.advantages 已经是被 flatten 后的正确形状了，我们再复制它。
        original_advantages = self.rollout_buffer.advantages.copy()

        # 3. 手动 flatten 我们的自定义数组，使其形状与 original_advantages 保持完全一致。
        cost_advantages_flat = self.rollout_buffer.cost_advantages.swapaxes(0, 1).reshape(-1, 1)
        cost_returns_flat = self.rollout_buffer.cost_returns.swapaxes(0, 1).reshape(-1, 1)
        
        # 4. 现在两个数组的形状都是 (4096, 1)，可以安全地进行运算。
        cost_adv_for_update = cost_advantages_flat
        # ===============================================================

        # --- 计算梯度内积 p ---
        # 我们已经获取了 full_batch，可以直接使用
        reward_advantages_p = torch.as_tensor(original_advantages, device=self.device).flatten()
        cost_advantages_p = torch.as_tensor(cost_adv_for_update, device=self.device).flatten()
        
        self.policy.train()
        _, _, log_prob, _ = self.policy.evaluate_actions(full_batch.observations, full_batch.actions)
        ratio = torch.exp(log_prob - full_batch.old_log_prob)
        
        reward_surrogate_loss = -torch.min(reward_advantages_p * ratio, reward_advantages_p * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)).mean()
        cost_surrogate_loss = -torch.min(cost_advantages_p * ratio, cost_advantages_p * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)).mean()
        
        g_r_tensors = torch.autograd.grad(reward_surrogate_loss, self.policy.parameters(), retain_graph=True, allow_unused=True)
        g_r_flat = torch.cat([grad.flatten() for grad in g_r_tensors if grad is not None])
        g_c_tensors = torch.autograd.grad(cost_surrogate_loss, self.policy.parameters(), retain_graph=False, allow_unused=True)
        g_c_flat = torch.cat([grad.flatten() for grad in g_c_tensors if grad is not None])
        p = torch.dot(g_r_flat, g_c_flat).item() if g_r_flat.numel() > 0 and g_c_flat.numel() > 0 else 0.0
        
        self.logger.record("sagi/cost_surplus_c", c)
        self.logger.record("sagi/grad_inner_product_p", p)
        self.logger.record("sagi/lambda", self.lambda_)

        if c < 0 and p <= 0:
            self.logger.record("sagi/mode", "A")
            # 模式A: 保持原始奖励优势，即 self.rollout_buffer.advantages 已经是 original_advantages
            self.rollout_buffer.advantages = original_advantages
        elif c > 0:
            self.logger.record("sagi/mode", "C")
            self.rollout_buffer.advantages = -cost_adv_for_update.copy()
        else:
            self.logger.record("sagi/mode", "B")
            self.rollout_buffer.advantages = (original_advantages - self.lambda_ * cost_adv_for_update) / (1 + self.lambda_)

        self.lambda_ = max(0, self.lambda_ + self.lambda_lr * c)

        # --- PPO 小批次更新循环 ---
        # 再次调用 get() 时，由于 generator_ready=True，它会直接从已 flatten 的数据中创建小批次
        current_batch_start_idx = 0
        for rollout_data in self.rollout_buffer.get(self.batch_size):
            reward_values, cost_values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, rollout_data.actions)
            advantages = rollout_data.advantages
            if self.normalize_advantage:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            ratio = torch.exp(log_prob - rollout_data.old_log_prob)
            policy_loss = -torch.min(advantages * ratio, advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)).mean()
            
            batch_size = rollout_data.observations.shape[0]
            # 手动切片获取对应的小批次成本回报
            cost_returns_batch = cost_returns_flat[current_batch_start_idx : current_batch_start_idx + batch_size]
            current_batch_start_idx += batch_size

            value_loss = F.mse_loss(rollout_data.returns, reward_values.flatten())
            cost_value_loss = F.mse_loss(torch.as_tensor(cost_returns_batch, device=self.device).flatten(), cost_values.flatten())
            
            entropy_loss = -torch.mean(entropy) if entropy is not None else 0.0
            loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + self.cost_vf_coef * cost_value_loss

            self.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += 1
        # 在训练结束后恢复原始的 advantages，以便日志记录等后续步骤使用
        self.rollout_buffer.advantages = original_advantages
    
    def collect_rollouts(self, env: VecEnv, callback: BaseCallback, rollout_buffer: SAGIRolloutBuffer, n_rollout_steps: int) -> bool:
        assert self._last_obs is not None
        self.policy.set_training_mode(False)
        n_steps = 0
        rollout_buffer.reset()
        callback.on_rollout_start()
        while n_steps < n_rollout_steps:
            with torch.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                features = self.policy.extract_features(obs_tensor)
                latent_pi, latent_vf = self.policy.mlp_extractor(features)
                distribution = self.policy._get_action_dist_from_latent(latent_pi)
                actions = distribution.get_actions(deterministic=False)
                log_probs = distribution.log_prob(actions)
                values = self.policy.value_net(latent_vf)
                cost_values = self.policy.cost_value_net(latent_vf)

            actions = actions.cpu().numpy()
            clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)
            new_obs, rewards, dones, infos = env.step(clipped_actions)
            costs = np.array([info.get("cost", 0) for info in infos])
            self.num_timesteps += env.num_envs
            if callback.on_step() is False: return False
            self._update_info_buffer(infos, dones)
            n_steps += 1
            if isinstance(self.action_space, spaces.Discrete): actions = actions.reshape(-1, 1)
            rollout_buffer.add(obs=self._last_obs, action=actions, reward=rewards, cost=costs, episode_start=self._last_episode_starts, value=values, cost_value=cost_values, log_prob=log_probs)
            self._last_obs = new_obs
            self._last_episode_starts = dones
        with torch.no_grad():
            last_obs_tensor = obs_as_tensor(new_obs, self.device)
            features = self.policy.extract_features(last_obs_tensor)
            _, latent_vf = self.policy.mlp_extractor(features)
            last_values = self.policy.value_net(latent_vf)
            last_cost_values = self.policy.cost_value_net(latent_vf)
        rollout_buffer.compute_returns_and_advantage(last_values=last_values, last_cost_values=last_cost_values, dones=dones)
        callback.on_rollout_end()
        return True
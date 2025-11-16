# [关键修正] 从 typing 模块导入 Generator 和 Tuple
from typing import Any, Dict, Optional, Type, Union, Generator, Tuple

import numpy as np
import torch
from torch.nn import functional as F
import torch as th

from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import VecEnv, VecNormalize
from stable_baselines3.common.type_aliases import GymEnv

# 关键：从您现有的 SAGI-PPO 文件中导入基类
# 我们假设您的 SAGIPPO 类在 'sagi_ppo.py' 文件中
from sagi_ppo import SAGIPPO

class PPOLagrangian(SAGIPPO):
    """
    PPO-Lagrangian 算法 (作为 SAGI-PPO 的基线)。

    该类继承自 SAGIPPO, 从而确保了：
    1. 共享相同的 ActorCriticCostPolicy (策略网络结构)
    2. 共享相同的 SAGIRolloutBuffer (经验回放缓冲区)
    3. 共享相同的 collect_rollouts 方法
    4. 共享所有相同的超参数 (cost_limit, lambda_lr, cost_vf_coef 等)

    这使得本算法与 SAGI-PPO 的对比 (Ablation Study) 绝对公平。

    唯一的区别是 *重写* (override) 了 `train()` 方法:
    - 移除了 SAGI-PPO 的 (c, p) KKT 诊断 (Case A 和 Case C)。
    - 移除了昂贵的梯度内积 `p` 的计算。
    - *始终* 执行 Case B (PPO-Lagrangian) 的更新逻辑。
    """

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        # --- 复制 SAGIPPO 的所有参数以确保签名一致 ---
        initial_cost_limit: float = 500.0,
        final_cost_limit: float = 30.0,
        decay_start_step: int = 5_000_000, 
        lambda_lr: float = 0.035, 
        cost_vf_coef: float = 0.5,
        **kwargs,
    ):
        # 初始化父类 (SAGIPPO)
        # 这将自动设置好 cost_vf, lambda_k (self.lambda_), cost_limit, 
        # SAGIRolloutBuffer, ActorCriticCostPolicy 等
        super().__init__(
            policy=policy,
            env=env,
            initial_cost_limit=initial_cost_limit,
            final_cost_limit=final_cost_limit,
            decay_start_step=decay_start_step,
            lambda_lr=lambda_lr,
            cost_vf_coef=cost_vf_coef,
            **kwargs,
        )

    def train(self) -> None:
        """
        重写的训练方法 (PPO-Lagrangian 逻辑)。
        
        该方法移除了 Case A (奖励) 和 Case C (安全) 的诊断,
        强制执行标准的 PPO-Lagrangian (Case B) 更新。
        """
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        # --- 1. [PPO-L] 更新成本限制 (与 SAGI-PPO 相同) ---
        current_step = self.num_timesteps
        decay_start = self.decay_start_step
        total_steps = self._total_timesteps

        if current_step <= decay_start:
            current_cost_limit = self.initial_cost_limit
        else:
            progress_in_decay_phase = (current_step - decay_start) / max(1, total_steps - decay_start)
            progress_in_decay_phase = min(1.0, progress_in_decay_phase)
            current_cost_limit = self.initial_cost_limit + progress_in_decay_phase * (self.final_cost_limit - self.initial_cost_limit)
        
        self.cost_limit = current_cost_limit
        # (日志记录前缀改为 'train/' 以便与 SAGI-PPO 的 'sagi/' 区分)
        self.logger.record("train/current_cost_limit", self.cost_limit)

        clip_range = self.clip_range(self._current_progress_remaining)

        # --- 2. [PPO-L] 计算成本盈余 'c' (与 SAGI-PPO 相同) ---
        j_c_k = self.rollout_buffer.get_mean_episode_costs()
        c = j_c_k - self.cost_limit

        # --- 3. [PPO-L] 更新 Lambda (原对偶梯度上升) ---
        # (这在 SAGI-PPO 中是在 A-B-C 逻辑之后做的)
        self.lambda_ = max(0, self.lambda_ + self.lambda_lr * c)
        
        self.logger.record("train/cost_surplus_c", c)
        self.logger.record("train/lambda", self.lambda_)
        
        # --- 4. [PPO-L] 准备数据缓冲区 (与 SAGI-PPO 相同) ---
        # (复制您在 SAGI-PPO.train() 中的形状修正逻辑)
        full_batch_generator = self.rollout_buffer.get(batch_size=self.rollout_buffer.buffer_size * self.n_envs)
        full_batch = next(full_batch_generator)
        original_advantages = self.rollout_buffer.advantages.copy() # (A_R)
        cost_advantages_flat = self.rollout_buffer.cost_advantages.swapaxes(0, 1).reshape(-1, 1) # (A_C)
        cost_returns_flat = self.rollout_buffer.cost_returns.swapaxes(0, 1).reshape(-1, 1) # (V_C_target)

        # --- 5. [PPO-L] 计算拉格朗日优势 (强制 Case B) ---
        # (移除了 SAGI-PPO 的 'p' 计算和 'if/elif/else' 逻辑)
        # (这就是 PPO-Lagrangian 和 SAGI-PPO 的唯一区别)
        self.rollout_buffer.advantages = (original_advantages - self.lambda_ * cost_advantages_flat) / (1 + self.lambda_)

        # --- 6. [PPO-L] PPO 小批次更新循环 (与 SAGI-PPO 相同) ---
        current_batch_start_idx = 0
        for rollout_data in self.rollout_buffer.get(self.batch_size):
            # (以下代码从 SAGI-PPO.train() 复制而来, 确保更新逻辑一致)
            reward_values, cost_values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, rollout_data.actions)
            
            # 'advantages' 此时是我们在步骤5中计算的拉格朗日优势
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
            
            # (总损失与 SAGI-PPO 相同)
            loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + self.cost_vf_coef * cost_value_loss

            self.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += 1
        # 恢复原始奖励优势 (与 SAGI-PPO 相同)
        self.rollout_buffer.advantages = original_advantages
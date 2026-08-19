import warnings
# [关键修正] 从 typing 模块导入 Generator 和 Tuple
from typing import Any, Dict, Optional, Type, Union, Generator, NamedTuple, Tuple

import numpy as np
import torch
from gymnasium import spaces
from torch.nn import functional as F
import torch.nn as nn
import torch as th

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import explained_variance, obs_as_tensor
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
class SAGIRolloutBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    cost_advantages: torch.Tensor
    cost_returns: torch.Tensor


class SAGIRolloutBuffer(RolloutBuffer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.costs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.cost_values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.cost_advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.cost_returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.last_dones = np.zeros(self.n_envs, dtype=bool)


    def reset(self) -> None:
        super().reset()
        self.costs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.cost_values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.cost_advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.cost_returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.last_dones = np.zeros(self.n_envs, dtype=bool)

    def add(self, cost: np.ndarray, cost_value: torch.Tensor, **kwargs) -> None:
        self.costs[self.pos] = np.array(cost).copy()
        self.cost_values[self.pos] = cost_value.clone().cpu().numpy().flatten()
        super().add(**kwargs)

    def compute_returns_and_advantage(self, last_values: torch.Tensor, last_cost_values: torch.Tensor, dones: np.ndarray):
        super().compute_returns_and_advantage(last_values, dones)
        self.last_dones = np.asarray(dones, dtype=bool).copy()
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

    def prepare_for_sampling(self) -> None:
        """Flatten reward and cost arrays once using the same env-major order."""
        if self.generator_ready:
            return

        tensor_names = [
            "observations",
            "actions",
            "values",
            "log_probs",
            "advantages",
            "returns",
            "cost_advantages",
            "cost_returns",
        ]
        for tensor_name in tensor_names:
            self.__dict__[tensor_name] = self.swap_and_flatten(self.__dict__[tensor_name])
        self.generator_ready = True

    def get(self, batch_size: Optional[int] = None) -> Generator[SAGIRolloutBufferSamples, None, None]:
        """Yield aligned reward/cost samples without moving the full rollout to the device."""
        assert self.full, ""
        total_samples = self.buffer_size * self.n_envs
        indices = np.random.permutation(total_samples)
        self.prepare_for_sampling()

        if batch_size is None:
            batch_size = total_samples

        start_idx = 0
        while start_idx < total_samples:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> SAGIRolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            self.cost_advantages[batch_inds].flatten(),
            self.cost_returns[batch_inds].flatten(),
        )
        return SAGIRolloutBufferSamples(*tuple(map(self.to_torch, data)))

    def get_mean_episode_costs(self) -> float:
        """Return the mean discounted cost of complete episodes in this rollout.

        ``episode_starts[t, env]`` marks the first transition of a new episode.
        A forward pass therefore closes the previous episode *before* adding the
        cost at ``t``.  Prefixes that started before the rollout and unfinished
        suffixes at the end of the rollout are excluded.  Episodes ending on the
        final rollout transition are included through ``last_dones``.
        """
        episode_costs = []
        running_costs = np.zeros(self.n_envs, dtype=np.float64)
        discount_weights = np.ones(self.n_envs, dtype=np.float64)
        has_observed_start = np.zeros(self.n_envs, dtype=bool)

        for step in range(self.buffer_size):
            start_mask = self.episode_starts[step].astype(bool)
            completed_mask = start_mask & has_observed_start

            if np.any(completed_mask):
                episode_costs.extend(running_costs[completed_mask].tolist())

            if np.any(start_mask):
                running_costs[start_mask] = 0.0
                discount_weights[start_mask] = 1.0
                has_observed_start[start_mask] = True

            running_costs += discount_weights * self.costs[step]
            discount_weights *= self.gamma

        completed_at_rollout_end = self.last_dones & has_observed_start
        if np.any(completed_at_rollout_end):
            episode_costs.extend(running_costs[completed_at_rollout_end].tolist())

        if not episode_costs:
            raise RuntimeError(
                "No complete episode was observed in the rollout buffer. "
                "Increase n_steps so the expected episode length fits in one rollout."
            )

        return float(np.mean(episode_costs))

# ==============================================================================
# 3. SAGI-PPO 算法 (此部分已稳定，无需修改)
# ==============================================================================
class SAGIPPO(PPO):
    policy_aliases: Dict[str, Type[ActorCriticPolicy]] = { "MlpPolicy": ActorCriticCostPolicy }

    def __init__(self, policy, env, 
                 initial_cost_limit: float = 500.0,
                 final_cost_limit: float = 30.0,
                 decay_start_step: Optional[int] = None,
                 cost_warmup_fraction: float = 0.10,
                 cost_anneal_fraction: float = 0.40,
                 lambda_lr: float = 0.035, cost_vf_coef: float = 0.5, **kwargs):

        self.initial_cost_limit = initial_cost_limit
        self.final_cost_limit = final_cost_limit
        # Retained only so older checkpoints and the out-of-scope CPO class load.
        self.decay_start_step = decay_start_step
        self.cost_warmup_fraction = cost_warmup_fraction
        self.cost_anneal_fraction = cost_anneal_fraction
        self._validate_cost_schedule()
        self.lambda_lr = lambda_lr
        self.cost_vf_coef = cost_vf_coef
        self.lambda_ = 0.0
        self.cost_limit = self.initial_cost_limit
        super().__init__(policy=policy, env=env, rollout_buffer_class=SAGIRolloutBuffer, **kwargs)

    def _validate_cost_schedule(self) -> None:
        if not 0.0 <= self.cost_warmup_fraction < 1.0:
            raise ValueError("cost_warmup_fraction must be in [0, 1).")
        if not 0.0 < self.cost_anneal_fraction <= 1.0:
            raise ValueError("cost_anneal_fraction must be in (0, 1].")
        if self.cost_warmup_fraction + self.cost_anneal_fraction > 1.0:
            raise ValueError(
                "cost_warmup_fraction + cost_anneal_fraction must not exceed 1."
            )

    def get_cost_limit(self, current_step: int, total_steps: int) -> Tuple[float, int]:
        """Return the three-stage cost limit and its phase at ``current_step``.

        Phase 0 keeps the initial limit during warm-up, phase 1 linearly
        anneals to the final limit, and phase 2 holds the final limit.
        """
        if total_steps <= 0:
            raise ValueError("total_steps must be positive.")

        progress = float(np.clip(current_step / total_steps, 0.0, 1.0))
        anneal_end = self.cost_warmup_fraction + self.cost_anneal_fraction

        if progress <= self.cost_warmup_fraction:
            return float(self.initial_cost_limit), 0
        if progress < anneal_end:
            anneal_progress = (
                (progress - self.cost_warmup_fraction) / self.cost_anneal_fraction
            )
            cost_limit = self.initial_cost_limit + anneal_progress * (
                self.final_cost_limit - self.initial_cost_limit
            )
            return float(cost_limit), 1
        return float(self.final_cost_limit), 2

    def _compute_gradient_inner_product(self, clip_range: float) -> Tuple[float, int]:
        """Compute the full-rollout reward/cost gradient dot product in bounded chunks.

        Each chunk gradient is weighted by its share of rollout samples, so the
        accumulated gradients equal the gradients of the original full-batch
        mean losses without constructing a full-batch CUDA tensor.
        """
        total_samples = self.rollout_buffer.buffer_size * self.n_envs
        gradient_batch_size = min(total_samples, max(self.batch_size, 1024))
        parameters = tuple(
            parameter for parameter in self.policy.parameters() if parameter.requires_grad
        )
        reward_gradients = [torch.zeros_like(parameter) for parameter in parameters]
        cost_gradients = [torch.zeros_like(parameter) for parameter in parameters]

        for rollout_data in self.rollout_buffer.get(gradient_batch_size):
            _, _, log_prob, _ = self.policy.evaluate_actions(
                rollout_data.observations, rollout_data.actions
            )
            ratio = torch.exp(log_prob - rollout_data.old_log_prob)
            clipped_ratio = torch.clamp(ratio, 1 - clip_range, 1 + clip_range)

            reward_advantages = rollout_data.advantages.flatten()
            cost_advantages = rollout_data.cost_advantages.flatten()
            reward_surrogate_loss = -torch.min(
                reward_advantages * ratio,
                reward_advantages * clipped_ratio,
            ).mean()
            cost_surrogate_loss = -torch.min(
                cost_advantages * ratio,
                cost_advantages * clipped_ratio,
            ).mean()

            reward_chunk_gradients = torch.autograd.grad(
                reward_surrogate_loss,
                parameters,
                retain_graph=True,
                allow_unused=True,
            )
            cost_chunk_gradients = torch.autograd.grad(
                cost_surrogate_loss,
                parameters,
                allow_unused=True,
            )
            chunk_weight = reward_advantages.numel() / total_samples

            for accumulated, gradient in zip(reward_gradients, reward_chunk_gradients):
                if gradient is not None:
                    accumulated.add_(gradient.detach(), alpha=chunk_weight)
            for accumulated, gradient in zip(cost_gradients, cost_chunk_gradients):
                if gradient is not None:
                    accumulated.add_(gradient.detach(), alpha=chunk_weight)

        inner_product = sum(
            torch.sum(reward_gradient * cost_gradient)
            for reward_gradient, cost_gradient in zip(reward_gradients, cost_gradients)
        )
        return float(inner_product.item()), gradient_batch_size

    def _optimize_policy(self, clip_range: float) -> None:
        """Run PPO minibatch optimization for the configured number of epochs.

        SAGI-PPO and PPO-Lagrangian both call this method so that their optimizer
        behavior stays identical.  The previous implementation made only one
        pass over the rollout buffer and silently ignored ``self.n_epochs``.
        """
        entropy_losses = []
        policy_losses = []
        value_losses = []
        cost_value_losses = []
        clip_fractions = []
        approx_kl_divs = []
        continue_training = True
        epochs_completed = 0
        final_loss = None

        for epoch in range(self.n_epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = actions.long().flatten()

                reward_values, cost_values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )

                ratio = torch.exp(log_prob - rollout_data.old_log_prob)
                policy_loss = -torch.min(
                    advantages * ratio,
                    advantages * torch.clamp(
                        ratio, 1 - clip_range, 1 + clip_range
                    ),
                ).mean()
                value_loss = F.mse_loss(
                    rollout_data.returns, reward_values.flatten()
                )
                cost_value_loss = F.mse_loss(
                    rollout_data.cost_returns, cost_values.flatten()
                )

                if entropy is None:
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)

                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                    + self.cost_vf_coef * cost_value_loss
                )

                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl = torch.mean(
                        (torch.exp(log_ratio) - 1) - log_ratio
                    ).item()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                cost_value_losses.append(cost_value_loss.item())
                entropy_losses.append(entropy_loss.item())
                clip_fractions.append(
                    torch.mean((torch.abs(ratio - 1) > clip_range).float()).item()
                )
                approx_kl_divs.append(approx_kl)

                if (
                    self.target_kl is not None
                    and approx_kl > 1.5 * self.target_kl
                ):
                    continue_training = False
                    if self.verbose >= 1:
                        print(
                            f"Early stopping at epoch {epoch + 1} because "
                            f"approx_kl={approx_kl:.4f} exceeded "
                            f"1.5 * target_kl={1.5 * self.target_kl:.4f}."
                        )
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.policy.optimizer.step()
                final_loss = loss.item()

            self._n_updates += 1
            epochs_completed += 1
            if not continue_training:
                break

        self.logger.record("train/epochs_completed", epochs_completed)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if policy_losses:
            self.logger.record("train/policy_gradient_loss", np.mean(policy_losses))
            self.logger.record("train/value_loss", np.mean(value_losses))
            self.logger.record("train/cost_value_loss", np.mean(cost_value_losses))
            self.logger.record("train/entropy_loss", np.mean(entropy_losses))
            self.logger.record("train/clip_fraction", np.mean(clip_fractions))
            self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        if final_loss is not None:
            self.logger.record("train/loss", final_loss)
        self.logger.record(
            "train/explained_variance",
            explained_variance(
                self.rollout_buffer.values.flatten(),
                self.rollout_buffer.returns.flatten(),
            ),
        )

    def train(self) -> None:
        """
        [最终修正]
        调整了代码的执行顺序，以适应 get() 方法的惰性求值特性，
        从根源上解决反复出现的形状不匹配问题。
        """
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        # Three-stage cost curriculum: 10% warm-up, 40% linear annealing,
        # followed by 50% training at the final cost limit.
        current_step = self.num_timesteps
        total_steps = self._total_timesteps
        current_cost_limit, schedule_phase = self.get_cost_limit(current_step, total_steps)

        self.cost_limit = current_cost_limit
        self.logger.record("sagi/current_cost_limit", self.cost_limit)
        self.logger.record("sagi/cost_schedule_phase", schedule_phase)

        clip_range = self.clip_range(self._current_progress_remaining)

        j_c_k = self.rollout_buffer.get_mean_episode_costs()
        c = j_c_k - self.cost_limit
        self.logger.record("sagi/empirical_discounted_cost", j_c_k)

        # Flatten reward and cost data together so every later permutation stays aligned.
        self.rollout_buffer.prepare_for_sampling()
        original_advantages = self.rollout_buffer.advantages.copy()
        cost_adv_for_update = self.rollout_buffer.cost_advantages

        # Compute p over the complete rollout, but keep CUDA allocations bounded.
        self.policy.train()
        p, gradient_batch_size = self._compute_gradient_inner_product(clip_range)
        self.logger.record("sagi/gradient_batch_size", gradient_batch_size)
        
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

        try:
            self._optimize_policy(clip_range)
        finally:
            # Restore reward advantages for logging and any later buffer users.
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

# cpo.py
#
# 这是一个基于 OmniSafe CPO 代码的实现, 
# 适配到 SAGIPPO 基类上, 以便共享缓冲区和策略网络。

import torch
import torch as th
import numpy as np
from typing import Any, Dict, Optional, Type, Union, Generator, Tuple

from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.utils import obs_as_tensor, explained_variance, get_schedule_fn
from stable_baselines3.common import logger

# 关键：从您现有的 SAGI-PPO 文件中导入基类
from sagi_ppo import SAGIPPO

# --- [新] OmniSafe CPO 所需的辅助函数 ---
# (我们必须从 OmniSafe 中复制这些)
from omnisafe.utils import distributed
from omnisafe.utils.math import conjugate_gradients
from omnisafe.utils.tools import (
    get_flat_gradients_from,
    get_flat_params_from,
    set_param_values_to_model,
)
# ---------------------------------------


class CPO(SAGIPPO):
    """
    Constrained Policy Optimization (CPO) 算法。
    
    该实现移植自 OmniSafe 的 CPO, 并适配到 SAGIPPO 基类上。
    它共享 ActorCriticCostPolicy 和 SAGIRolloutBuffer,
    但 *完全重写* 了 train() 方法, 以使用共轭梯度 (CG) 和
    信任域 (Trust Region) 约束求解器, 而不是 PPO 的裁剪 (Clipping)
    和 Adam 优化器 (仅用于策略)。
    """

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        # --- [新] CPO 特有的参数 ---
        target_kl: float = 0.01,
        cg_iters: int = 10,
        cg_damping: float = 0.1,
        fvp_sample_freq: int = 1, # (OmniSafe FVP 采样频率, 1=使用所有)
        line_search_decay: float = 0.8,
        line_search_total_steps: int = 15,
        # --- 复制 SAGIPPO 的所有参数 ---
        initial_cost_limit: float = 500.0,
        final_cost_limit: float = 30.0,
        decay_start_step: int = 5_000_000, 
        lambda_lr: float = 0.035, # (CPO 不使用此参数)
        cost_vf_coef: float = 0.5,
        # (确保 n_epochs=1, 因为 CPO 每次只更新一次)
        n_epochs: int = 1,
        batch_size: int = 2048, # (CPO 必须使用 full batch, 确保 batch_size >= n_steps)
        **kwargs,
    ):
        
        # [关键] CPO 每次 rollout 只更新一次
        if n_epochs != 1:
            warnings.warn(f"CPO 每次 rollout 只更新一次。将 n_epochs 从 {n_epochs} 强制设为 1。")
            n_epochs = 1
        
        # [关键] CPO 必须在 full-batch 上操作
        # (确保 batch_size 足够大以包含整个 rollout buffer)
        # (我们将在 train() 中手动处理 full-batch, 
        #  因此 SB3 的 batch_size 参数作用不大, 
        #  但 n_steps (rollout buffer size) 很重要)

        super().__init__(
            policy=policy,
            env=env,
            initial_cost_limit=initial_cost_limit,
            final_cost_limit=final_cost_limit,
            decay_start_step=decay_start_step,
            lambda_lr=lambda_lr,
            cost_vf_coef=cost_vf_coef,
            n_epochs=n_epochs,
            batch_size=batch_size, 
            **kwargs,
        )
        
        # CPO 不使用 lambda_, 将其永久设为 0
        self.lambda_ = 0.0
        
        # 存储 CPO 特有的超参数
        self.target_kl = target_kl
        self.cg_iters = cg_iters
        self.cg_damping = cg_damping
        self.fvp_sample_freq = fvp_sample_freq
        self.line_search_decay = line_search_decay
        self.line_search_total_steps = line_search_total_steps
        
        # 用于 FVP (Fisher-Vector Product) 的采样观测
        self._fvp_obs = None
    
    # --- [新] CPO 辅助方法 (从 OmniSafe 移植) ---

    def _fvp(self, vector: torch.Tensor) -> torch.Tensor:
        """
        计算 Fisher-Vector Product (FVP), Hx。
        这是 CPO/TRPO 计算信任域的关键。
        """
        self.policy.actor.zero_grad()
        
        # 使用采样的观测数据
        obs = self._fvp_obs 
        if obs is None:
            # (安全回退, 尽管不应该发生)
            obs = self.rollout_buffer.observations
            obs = obs.swapaxes(0, 1).reshape(-1, *obs.shape[2:])
            obs = obs_as_tensor(obs, self.device)
            
        with torch.no_grad():
            p_dist = self.policy.actor(obs)

        q_dist = self.policy.actor(obs)
        kl = torch.distributions.kl.kl_divergence(p_dist, q_dist).mean()

        grads = torch.autograd.grad(kl, self.policy.actor.parameters(), create_graph=True)
        flat_grad_kl = get_flat_gradients_from(grads)

        kl_v = (flat_grad_kl * vector).sum()
        grads = torch.autograd.grad(kl_v, self.policy.actor.parameters(), retain_graph=False)
        flat_grad_grad_kl = get_flat_gradients_from(grads)

        return flat_grad_grad_kl + vector * self.cg_damping

    def _loss_pi(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算奖励的代理损失 (Policy Loss for Reward)
        (移植自 OmniSafe CPO._loss_pi)
        """
        # (注意：CPO/TRPO 不使用 PPO 的裁剪 'clip_range')
        self.policy.actor(obs) # (确保前向传播已执行)
        logp_ = self.policy.actor.log_prob(act)
        ratio = torch.exp(logp_ - logp)
        return -(ratio * adv).mean()

    def _loss_pi_cost(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv_c: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算成本的代理损失 (Policy Loss for Cost)
        (移植自 OmniSafe CPO._loss_pi_cost)
        """
        self.policy.actor(obs) # (确保前向传播已执行)
        logp_ = self.policy.actor.log_prob(act)
        ratio = torch.exp(logp_ - logp)
        return (ratio * adv_c).mean()
    
    # (从 OmniSafe 移植: _determine_case)
    def _determine_case(
        self,
        b_grads: torch.Tensor, # 成本梯度
        ep_costs: float,      # 当前成本 - 成本限制 (c)
        q: torch.Tensor,      # g_R^T H^{-1} g_R
        r: torch.Tensor,      # g_R^T H^{-1} g_C
        s: torch.Tensor,      # g_C^T H^{-1} g_C
    ) -> tuple[int, torch.Tensor, torch.Tensor]:

        if b_grads.dot(b_grads) <= 1e-6 and ep_costs < 0:
            A = torch.zeros(1)
            B = torch.zeros(1)
            optim_case = 4
        else:
            assert torch.isfinite(r).all(), 'r is not finite'
            assert torch.isfinite(s).all(), 's is not finite'

            A = q - r**2 / (s + 1e-8)
            B = 2 * self.target_kl - ep_costs**2 / (s + 1e-8)

            if ep_costs < 0 and B < 0:
                optim_case = 3
            elif ep_costs < 0 <= B:
                optim_case = 2
            elif ep_costs >= 0 and B >= 0:
                optim_case = 1
                logger.log('Alert! Attempting feasible recovery!', 'yellow')
            else:
                optim_case = 0
                logger.log('Alert! Attempting infeasible recovery!', 'red')

        return optim_case, A, B

    # (从 OmniSafe 移植: _step_direction)
    def _step_direction(
        self,
        optim_case: int,
        xHx: torch.Tensor,
        g_r: torch.Tensor, # 奖励梯度 (g)
        g_c: torch.Tensor, # 成本梯度 (b)
        A: torch.Tensor,
        B: torch.Tensor,
        q: torch.Tensor,   # g_r^T H^{-1} g_r
        p: torch.Tensor,   # H^{-1} g_c
        r: torch.Tensor,   # g_r^T H^{-1} g_c
        s: torch.Tensor,   # g_c^T H^{-1} g_c
        ep_costs: float,   # (c)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        # x 是 H^{-1} g_r (奖励更新方向)
        x = conjugate_gradients(self._fvp, g_r, self.cg_iters)

        if optim_case in (3, 4):
            alpha = torch.sqrt(2 * self.target_kl / (xHx + 1e-8))
            nu_star = torch.zeros(1)
            lambda_star = 1 / (alpha + 1e-8)
            step_direction = alpha * x

        elif optim_case in (1, 2):

            def project(data: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
                return torch.clamp(data, low, high)

            lambda_a = torch.sqrt(A / B)
            lambda_b = torch.sqrt(q / (2 * self.target_kl))
            
            r_num = r.item()
            eps_cost = ep_costs + 1e-8
            if ep_costs < 0:
                lambda_a_star = project(lambda_a, torch.as_tensor(0.0), r_num / eps_cost)
                lambda_b_star = project(lambda_b, r_num / eps_cost, torch.as_tensor(torch.inf))
            else:
                lambda_a_star = project(lambda_a, r_num / eps_cost, torch.as_tensor(torch.inf))
                lambda_b_star = project(lambda_b, torch.as_tensor(0.0), r_num / eps_cost)

            def f_a(lam: torch.Tensor) -> torch.Tensor:
                return -0.5 * (A / (lam + 1e-8) + B * lam) - r * ep_costs / (s + 1e-8)
            def f_b(lam: torch.Tensor) -> torch.Tensor:
                return -0.5 * (q / (lam + 1e-8) + 2 * self.target_kl * lam)

            lambda_star = (
                lambda_a_star if f_a(lambda_a_star) >= f_b(lambda_b_star) else lambda_b_star
            )
            nu_star = torch.clamp(lambda_star * ep_costs - r, min=0) / (s + 1e-8)
            step_direction = 1.0 / (lambda_star + 1e-8) * (x - nu_star * p)

        else: # case == 0
            lambda_star = torch.zeros(1)
            nu_star = torch.sqrt(2 * self.target_kl / (s + 1e-8))
            step_direction = -nu_star * p

        return step_direction, lambda_star, nu_star

    # (从 OmniSafe 移植: _cpo_search_step, 即 Line Search)
    def _cpo_search_step(
        self,
        step_direction: torch.Tensor,
        g_r: torch.Tensor, # 奖励梯度
        p_dist: torch.distributions.Distribution,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv_r: torch.Tensor,
        adv_c: torch.Tensor,
        loss_reward_before: torch.Tensor,
        loss_cost_before: torch.Tensor,
        violation_c: float, # (c)
        optim_case: int,
    ) -> tuple[torch.Tensor, int]:
        
        step_frac = 1.0
        theta_old = get_flat_params_from(self.policy.actor)
        expected_reward_improve = g_r.dot(step_direction)

        kl = torch.zeros(1)
        for step in range(self.line_search_total_steps):
            new_theta = theta_old + step_frac * step_direction
            set_param_values_to_model(self.policy.actor, new_theta)
            acceptance_step = step + 1

            with torch.no_grad():
                try:
                    loss_reward = self._loss_pi(obs=obs, act=act, logp=logp, adv=adv_r)
                except ValueError:
                    step_frac *= self.line_search_decay
                    continue
                loss_cost = self._loss_pi_cost(obs=obs, act=act, logp=logp, adv_c=adv_c)
                q_dist = self.policy.actor(obs)
                kl = torch.distributions.kl.kl_divergence(p_dist, q_dist).mean()
            
            loss_reward_improve = loss_reward_before - loss_reward
            loss_cost_diff = loss_cost - loss_cost_before
            
            # (在 CPO 中, 我们假设分布式=False, 移除 dist_avg)
            
            if not torch.isfinite(loss_reward) and not torch.isfinite(loss_cost):
                logger.log('WARNING: loss_pi not finite')
            if not torch.isfinite(kl):
                logger.log('WARNING: KL not finite')
                continue
            
            if loss_reward_improve < 0 if optim_case > 1 else False:
                logger.log('INFO: did not improve improve <0')
            elif loss_cost_diff > max(-violation_c, 0):
                logger.log(f'INFO: no improve {loss_cost_diff} > {max(-violation_c, 0)}')
            elif kl > self.target_kl:
                logger.log(f'INFO: violated KL constraint {kl} at step {step + 1}.')
            else:
                logger.log(f'Accept step at i={step + 1}')
                break
            step_frac *= self.line_search_decay
        else:
            logger.log('INFO: no suitable step found...')
            step_direction = torch.zeros_like(step_direction)
            acceptance_step = 0

        logger.record('train/KL', kl.item())
        
        # [关键] 恢复旧参数, 因为步进是在 learn() 中完成的
        set_param_values_to_model(self.policy.actor, theta_old) 
        
        return step_frac * step_direction, acceptance_step

    # --- [关键] 重写的 train() 方法 ---

    def train(self) -> None:
        """
        CPO 的核心训练逻辑。
        
        这 *完全替换* 了 PPO 的 train() 方法。
        它不使用 PPO epoch 或 batch_size。
        它在整个缓冲区上执行一次复杂的信任域更新。
        """
        # [与 SAGI-PPO 相同: 更新成本限制]
        current_step = self.num_timesteps
        decay_start = self.decay_start_step
        total_steps = self._total_timesteps
        if current_step <= decay_start:
            self.cost_limit = self.initial_cost_limit
        else:
            progress = (current_step - decay_start) / max(1, total_steps - decay_start)
            progress = min(1.0, progress)
            self.cost_limit = self.initial_cost_limit + progress * (self.final_cost_limit - self.initial_cost_limit)
        
        self.logger.record("train/current_cost_limit", self.cost_limit)
        
        # --- 1. [CPO] 准备 Full Batch 数据 ---
        # CPO/TRPO 必须在整个缓冲区上操作
        self.rollout_buffer.swap_and_flatten()
        
        # (我们必须从缓冲区获取所有数据, 而不是使用 'get()')
        obs = obs_as_tensor(self.rollout_buffer.observations, self.device)
        act = obs_as_tensor(self.rollout_buffer.actions, self.device)
        logp = obs_as_tensor(self.rollout_buffer.old_log_probs, self.device)
        adv_r = obs_as_tensor(self.rollout_buffer.advantages, self.device)
        adv_c = obs_as_tensor(self.rollout_buffer.cost_advantages, self.device)
        
        # [CPO] 标准化优势 (在 OmniSafe/TRPO 中是标准做法)
        adv_r = (adv_r - adv_r.mean()) / (adv_r.std() + 1e-8)
        adv_c = (adv_c - adv_c.mean()) / (adv_c.std() + 1e-8)

        # --- 2. [CPO] 更新价值网络 (V_R 和 V_C) ---
        # (CPO 仍然使用 Adam 来更新价值网络, 
        #  因此我们保留 PPO 的 n_epochs 和 batch_size 逻辑,
        #  *仅用于* 价值函数)
        
        # (我们必须手动获取 V_target 和 V_C_target)
        returns = obs_as_tensor(self.rollout_buffer.returns, self.device)
        cost_returns = obs_as_tensor(self.rollout_buffer.cost_returns, self.device)
        
        self.policy.set_training_mode(True)
        # (使用 PPO 的学习率更新器, 但只用于 V 和 V_C)
        self._update_learning_rate(self.policy.optimizer) 
        
        # [CPO] 价值网络更新循环
        for epoch in range(self.n_epochs):
            # (我们使用 SAGIRolloutBuffer 的 get() 来获取 *小批次* 数据, 
            #  但只用于 V 和 V_C 的训练)
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                # (只评估价值函数)
                values, cost_values, _, _ = self.policy.evaluate_actions(
                    rollout_data.observations, rollout_data.actions
                )
                values = values.flatten()
                cost_values = cost_values.flatten()
                
                # (计算 V_R 和 V_C 的损失)
                value_loss = F.mse_loss(rollout_data.returns, values)
                cost_value_loss = F.mse_loss(rollout_data.cost_returns, cost_values)
                
                loss = (self.vf_coef * value_loss 
                        + self.cost_vf_coef * cost_value_loss)
                
                # (只优化 V 和 V_C, 不触及 Actor)
                self.policy.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.policy.optimizer.step()
        
        # --- 3. [CPO] 更新策略网络 (Actor) ---
        # (这是 OmniSafe CPO._update_actor 的逻辑)
        
        self.policy.set_training_mode(False) # (用于 FVP 和 Line Search)

        # [CPO] 准备 FVP 采样
        self._fvp_obs = obs[::self.fvp_sample_freq]
        
        # [CPO] 获取旧参数和损失
        theta_old = get_flat_params_from(self.policy.actor)
        self.policy.actor.zero_grad()
        with torch.no_grad():
            loss_reward = self._loss_pi(obs, act, logp, adv_r)
            loss_cost = self._loss_pi_cost(obs, act, logp, adv_c)
            p_dist = self.policy.actor(obs) # (获取旧分布)

        loss_reward_before = loss_reward
        loss_cost_before = loss_cost

        # [CPO] 计算奖励梯度 g_r (g)
        loss_reward.backward(retain_graph=True) # (需要 retain_graph 以便计算 g_c)
        g_r = -get_flat_gradients_from(self.policy.actor) # (g = -g_L)
        
        # [CPO] 计算成本梯度 g_c (b)
        self.policy.actor.zero_grad()
        loss_cost.backward()
        g_c = get_flat_gradients_from(self.policy.actor) # (b = g_L_C)
        
        # [CPO] 计算 H^{-1}g_r (x), H^{-1}g_c (p) 和二次项
        x = conjugate_gradients(self._fvp, g_r, self.cg_iters) # H^{-1} g_r
        p = conjugate_gradients(self._fvp, g_c, self.cg_iters) # H^{-1} g_c
        
        xHx = x.dot(self._fvp(x)) # (q)
        r = g_r.dot(p)           # (r)
        s = g_c.dot(p)           # (s)
        q = xHx
        
        # [CPO] 获取当前成本盈余 c
        # (我们必须使用与 PPO-L 相同的逻辑来获取 c)
        ep_costs_c = self.rollout_buffer.get_mean_episode_costs() - self.cost_limit
        
        # [CPO] 求解 QP (确定 Case 和方向)
        optim_case, A, B = self._determine_case(
            b_grads=g_c,
            ep_costs=ep_costs_c,
            q=q, r=r, s=s,
        )

        step_direction, lambda_star, nu_star = self._step_direction(
            optim_case=optim_case,
            xHx=xHx,
            g_r=g_r, g_c=g_c, # (传递 g_r 和 g_c)
            A=A, B=B, q=q, p=p, r=r, s=s,
            ep_costs=ep_costs_c,
        )

        # [CPO] 线性搜索 (Line Search)
        step_direction, accept_step = self._cpo_search_step(
            step_direction=step_direction,
            g_r=g_r,
            p_dist=p_dist,
            obs=obs,
            act=act,
            logp=logp,
            adv_r=adv_r,
            adv_c=adv_c,
            loss_reward_before=loss_reward_before,
            loss_cost_before=loss_cost_before,
            violation_c=ep_costs_c,
            optim_case=optim_case,
        )

        # [CPO] 手动更新策略网络
        theta_new = theta_old + step_direction
        set_param_values_to_model(self.policy.actor, theta_new)

        self._n_updates += 1
        
        # --- 4. [CPO] 记录日志 (从 OmniSafe 移植) ---
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), 
                                           self.rollout_buffer.returns.flatten())
        cost_explained_var = explained_variance(self.rollout_buffer.cost_values.flatten(),
                                                self.rollout_buffer.cost_returns.flatten())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/value_loss", value_loss.item())
        self.logger.record("train/cost_value_loss", cost_value_loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/cost_explained_variance", cost_explained_var)

        self.logger.record('train/AcceptanceStep', accept_step)
        self.logger.record('train/FinalStepNorm', step_direction.norm().mean().item())
        self.logger.record('train/xHx', xHx.mean().item())
        self.logger.record('train/H_inv_g_R', x.norm().item())
        self.logger.record('train/g_R_norm', torch.norm(g_r).mean().item())
        self.logger.record('train/g_C_norm', torch.norm(g_c).mean().item())
        self.logger.record('train/Lambda_star', lambda_star.item())
        self.logger.record('train/Nu_star', nu_star.item())
        self.logger.record('train/OptimCase', int(optim_case))
        self.logger.record('train/q_CPO', q.item())
        self.logger.record('train/r_CPO', r.item())
        self.logger.record('train/s_CPO', s.item())
"""
@file: ppo.py
@description: 实现基于近端策略优化(Proximal Policy Optimization, PPO)的强化学习算法。
该文件包含以下主要组件:
1. Actor网络 - 生成动作策略分布
2. Critic网络 - 估计状态价值函数
3. RolloutBuffer - 存储和处理训练数据
4. PPOAgent - 封装PPO算法逻辑

PPO是一种流行的策略梯度算法，通过限制策略更新的步长来提高训练稳定性和样本效率。
该实现使用了GAE(广义优势估计)来计算优势函数，并使用策略比率裁剪来限制更新步长。
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Normal
import numpy as np

def init_weights(m):
    """
    初始化神经网络权重的辅助函数
    
    Args:
        m: 神经网络模块
        
    Notes:
        对线性层使用正交初始化(orthogonal initialization)，这在强化学习中被证明是有效的
        偏置项初始化为0
    """
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
        nn.init.constant_(m.bias, 0.0)



class RolloutBuffer:
    """
    轨迹数据缓冲区，用于存储和处理训练数据
    
    存储交互过程中的状态、动作、奖励等信息，并计算优势函数用于策略更新
    """
    def __init__(self, buffer_size, state_dim, action_dim, gamma, gae_lambda):
        """
        初始化轨迹缓冲区
        
        Args:
            buffer_size (int): 缓冲区大小（时间步数量）
            state_dim (int): 状态空间维度
            action_dim (int): 动作空间维度
            gamma (float): 折扣因子，用于计算折扣回报
            gae_lambda (float): GAE lambda参数，用于平衡偏差和方差
        """
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        self.ptr = 0
        self.path_start_idx = 0
        self.next_value = 0 
    
    def store(self, state, action, reward, done, value, log_prob):
        """
        存储单个时间步的交互数据
        
        Args:
            state (np.ndarray): 观察到的状态
            action (np.ndarray): 执行的动作
            reward (float): 获得的奖励
            done (bool): 回合是否结束
            value (float): Critic估计的状态价值
            log_prob (float): 执行动作的对数概率
        """
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.ptr += 1

    def finish_path(self, last_value):
        """
        处理轨迹结束时的数据，保存最终状态的价值估计
        
        Args:
            last_value (float): 最终状态的价值估计，用于计算优势函数
        
        Notes:
            对于回合结束的情况，需要正确处理最终价值估计
        """
        if self.ptr > 0:  # 确保buffer不为空
            self.next_value = last_value
        
        # 添加path_start_idx初始化和更新
        if not hasattr(self, 'path_start_idx'):
            self.path_start_idx = 0
        self.path_start_idx = self.ptr

    def calculate_advantages(self):
        """
        使用广义优势估计(GAE)计算每个时间步的优势函数
        
        Returns:
            np.ndarray: 计算得到的优势函数数组
            
        Notes:
            GAE通过混合n步回报来平衡偏差和方差，lambda参数控制混合程度
        """
        advantages = np.zeros_like(self.rewards)
        last_gae_lam = 0
        for t in reversed(range(self.buffer_size)):
            if t == self.buffer_size - 1:
                next_non_terminal = 1.0 - self.dones[t]
                next_values = self.next_value if hasattr(self, 'next_value') else 0
            else:
                next_non_terminal = 1.0 - self.dones[t]
                next_values = self.values[t + 1]
            delta = self.rewards[t] + self.gamma * next_values * next_non_terminal - self.values[t]
            advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
        return advantages

    def get(self):
        """
        获取处理后的训练数据
        
        Returns:
            dict: 包含状态、动作、对数概率、优势函数和目标回报的字典
            
        Notes:
            返回前会对优势函数和回报进行归一化，以提高训练稳定性
        """
        assert self.ptr == self.buffer_size
        self.ptr = 0
        
        advantages = self.calculate_advantages()
        returns = advantages + self.values
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
        
        # 归一化优势函数
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        batch = dict(
            states=torch.as_tensor(self.states, dtype=torch.float32),
            actions=torch.as_tensor(self.actions, dtype=torch.float32),
            log_probs=torch.as_tensor(self.log_probs, dtype=torch.float32),
            advantages=torch.as_tensor(advantages, dtype=torch.float32),
            returns=torch.as_tensor(returns, dtype=torch.float32)
        )
        return batch


class PPOAgent:
    """
    [重构后] PPO算法代理。
    现在使用一个统一的、能够处理复杂观测的HybridActorCritic网络。
    """
    def __init__(self, state_dim, action_dim, lr, hidden_dim, gamma, gae_lambda,
                 clip_epsilon, update_epochs, 
                 av_obs_dim, hv_obs_dim, traj_len, traj_feat_dim, rnn_hidden_dim=64):
        
        # --- 网络定义 ---
        self.ac_network = HybridActorCritic(
            av_obs_dim=av_obs_dim,
            hv_obs_dim=hv_obs_dim,
            traj_len=traj_len,
            traj_feat_dim=traj_feat_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            rnn_hidden_dim=rnn_hidden_dim
        )

        # --- 优化器 ---
        self.optimizer = Adam(self.ac_network.parameters(), lr=lr)

        # --- PPO 超参数 ---
        self.clip_epsilon = clip_epsilon
        self.update_epochs = update_epochs

    def select_action(self, state):
        state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            dist, value = self.ac_network(state) 
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(axis=-1)
        return action.numpy().flatten(), value.item(), log_prob.item()

    def get_deterministic_action(self, state):
        state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            dist, _ = self.ac_network(state)
            action = dist.mean
        return action.numpy().flatten()

    def update(self, buffer: RolloutBuffer):
        batch = buffer.get()
        
        for _ in range(self.update_epochs):
            dist, values = self.ac_network(batch['states'])
            values = values.squeeze()
            
            new_log_probs = dist.log_prob(batch['actions']).sum(axis=-1)
            ratio = torch.exp(new_log_probs - batch['log_probs'])
            
            # Actor Loss
            target_advantages = batch['advantages']
            surr1 = ratio * target_advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * target_advantages
            actor_loss = -torch.mean(torch.min(surr1, surr2))
            
            # Critic Loss
            critic_loss = nn.MSELoss()(values, batch['returns'])

            # Entropy Loss (可选，但推荐)
            entropy_loss = -dist.entropy().mean()

            # 统一优化
            # 您可以调整 critic_loss 和 entropy_loss 的权重
            total_loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss
            
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
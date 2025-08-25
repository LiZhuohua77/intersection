# ppo.py

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Normal
import numpy as np

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
        nn.init.constant_(m.bias, 0.0)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))
        self.apply(init_weights)

    def forward(self, state):
        mean = self.net(state)
        std = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        return dist

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.apply(init_weights)

    def forward(self, state):
        return self.net(state)

class RolloutBuffer:
    """常规PPO的Buffer，只存储奖励相关信息"""
    def __init__(self, buffer_size, state_dim, action_dim, gamma, gae_lambda):
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
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.ptr += 1

    def finish_path(self, last_value):
        """正确处理轨迹结束时的价值估计"""
        if self.ptr > 0:  # 确保buffer不为空
            self.next_value = last_value
        
        # 添加path_start_idx初始化和更新
        if not hasattr(self, 'path_start_idx'):
            self.path_start_idx = 0
        self.path_start_idx = self.ptr

    def calculate_advantages(self):
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
    def __init__(self, state_dim, action_dim, lr=3e-4, hidden_dim=256, gamma=0.99, gae_lambda=0.95,
                 clip_epsilon=0.2, update_epochs=10):
        # --- 网络定义 (只有一个Critic) ---
        self.actor = Actor(state_dim, action_dim, hidden_dim)
        self.critic = Critic(state_dim, hidden_dim)

        # --- 优化器 ---
        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)

        # --- PPO 超参数 ---
        self.clip_epsilon = clip_epsilon
        self.update_epochs = update_epochs

    def select_action(self, state):
        state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            dist = self.actor(state)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(axis=-1)
            value = self.critic(state)
        return action.numpy().flatten(), value.item(), log_prob.item()

    def get_deterministic_action(self, state):
        """获取确定性动作（策略分布的均值），用于评估。"""
        state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            dist = self.actor(state)
            action = dist.mean # <--- 使用均值而不是sample()
        return action.numpy().flatten()

    def update(self, buffer: RolloutBuffer, writer, global_step):
        batch = buffer.get()
        
        # --- 常规PPO更新循环 ---
        for i in range(self.update_epochs):
            # 1. Actor 更新
            dist = self.actor(batch['states'])
            new_log_probs = dist.log_prob(batch['actions']).sum(axis=-1)
            ratio = torch.exp(new_log_probs - batch['log_probs'])
            
            # 目标优势函数就是奖励优势 A_R
            target_advantages = batch['advantages']
            
            surr1 = ratio * target_advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * target_advantages
            actor_loss = -torch.mean(torch.min(surr1, surr2))
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # 2. Critic 更新
            values = self.critic(batch['states']).squeeze()
            critic_loss = nn.MSELoss()(values, batch['returns'])
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            epoch_step = global_step + (i / self.update_epochs)

            writer.add_scalar("losses/actor_loss", actor_loss.item(), epoch_step)
            writer.add_scalar("losses/critic_loss", critic_loss.item(), epoch_step)
            
            entropy = dist.entropy().mean().item()
            std_dev = self.actor.log_std.exp().mean().item()
            writer.add_scalar("policy/entropy", entropy, epoch_step)
            writer.add_scalar("policy/std_dev", std_dev, epoch_step)
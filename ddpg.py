import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import numpy as np
import random
from collections import deque

# --- 超参数 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

ENV_NAME = 'Pendulum-v1'
BUFFER_SIZE = 100000     # 经验回放池大小
BATCH_SIZE = 128         # 批处理大小
GAMMA = 0.99             # 折扣因子
TAU = 0.005              # 目标网络软更新系数
LR_ACTOR = 1e-4          # Actor 学习率
LR_CRITIC = 1e-3         # Critic 学习率
WEIGHT_DECAY = 0         # L2 正则化
MAX_EPISODES = 200       # 总训练回合数
MAX_STEPS = 500          # 每回合最大步数
NOISE = 0.1              # 动作探索噪声的标准差

# --- 经验回放池 ---
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        """
        初始化经验回放池
        :param buffer_size: int, 经验池最大容量
        :param batch_size: int, 采样批次大小
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        """向经验池中添加一条经验"""
        e = (state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """从经验池中随机采样一个批次的经验"""
        experiences = random.sample(self.memory, k=self.batch_size)
        
        # 将经验元组转换为批次张量
        states = torch.from_numpy(np.vstack([e[0] for e in experiences if e is not None])).float().to(DEVICE)
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None])).float().to(DEVICE)
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None])).float().to(DEVICE)
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences if e is not None])).float().to(DEVICE)
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)).float().to(DEVICE)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """返回当前经验池中的经验数量"""
        return len(self.memory)

# --- Actor 网络 ---
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_high):
        super(Actor, self).__init__()
        self.action_high = torch.tensor(action_high, dtype=torch.float32).to(DEVICE)
        
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # 使用 tanh 将输出缩放到 [-1, 1]，然后乘以动作范围
        action = torch.tanh(self.fc3(x)) * self.action_high
        return action

# --- Critic 网络 ---
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        
        # Q1 网络
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, state, action):
        # 将状态和动作在维度 1 上拼接
        x = torch.cat([state, action], dim=1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value

# --- DDPG Agent ---
class DDPGAgent():
    def __init__(self, state_dim, action_dim, action_high):
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)

        # 创建 Actor 网络和其目标网络
        self.actor_local = Actor(state_dim, action_dim, action_high).to(DEVICE)
        self.actor_target = Actor(state_dim, action_dim, action_high).to(DEVICE)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # 创建 Critic 网络和其目标网络
        self.critic_local = Critic(state_dim, action_dim).to(DEVICE)
        self.critic_target = Critic(state_dim, action_dim).to(DEVICE)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # 确保目标网络和本地网络的初始权重相同
        self.hard_update(self.actor_target, self.actor_local)
        self.hard_update(self.critic_target, self.critic_local)
        
    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def select_action(self, state, add_noise=True):
        """根据当前状态选择动作"""
        state = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE) # 添加unsqueeze(0)以创建批次维度
        self.actor_local.eval() # 切换到评估模式
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy().flatten() # 使用flatten()将[[a,b]]变为[a,b]
        self.actor_local.train() # 切换回训练模式

        if add_noise:
            # 添加高斯噪声以进行探索
            action += np.random.normal(0, NOISE, size=action.shape)
        
        # 从actor网络获取动作范围，而不是硬编码
        action_high = self.actor_local.action_high.cpu().numpy()
        action_low = -action_high

        # 将动作裁剪到环境允许的范围内
        return np.clip(action, action_low, action_high)

    def step(self, state, action, reward, next_state, done):
        """将经验存入回放池，并检查是否可以开始学习"""
        self.replay_buffer.add(state, action, reward, next_state, done)

        # 如果回放池中有足够的数据，则进行一次学习
        if len(self.replay_buffer) > BATCH_SIZE:
            experiences = self.replay_buffer.sample()
            self.learn(experiences)

    def learn(self, experiences):
        """
        使用一个批次的经验数据来更新 Actor 和 Critic 网络
        """
        states, actions, rewards, next_states, dones = experiences

        # --- 更新 Critic ---
        # 1. 从目标网络获取下一个状态的动作
        actions_next = self.actor_target(next_states)
        # 2. 从目标 Critic 网络计算下一个状态的 Q 值
        Q_targets_next = self.critic_target(next_states, actions_next)
        # 3. 计算当前 Q 值的目标 y_i (TD Target)
        # y_i = r_i + γ * Q_target(s_{i+1}, μ_target(s_{i+1}))
        Q_targets = rewards + (GAMMA * Q_targets_next * (1 - dones))
        
        # 4. 计算当前 Critic 网络输出的 Q 值
        Q_expected = self.critic_local(states, actions)
        
        # 5. 计算 Critic Loss
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        
        # 6. 最小化 Critic Loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1) # 可选：梯度裁剪
        self.critic_optimizer.step()

        # --- 更新 Actor ---
        # 1. 使用当前 Actor 网络计算动作
        actions_pred = self.actor_local(states)
        # 2. 计算 Actor Loss，我们希望最大化 Q 值，等价于最小化 -Q 值
        # L_actor = -E[Q(s_t, μ(s_t))]
        actor_loss = -self.critic_local(states, actions_pred).mean()
        
        # 3. 最小化 Actor Loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- 更新目标网络 (软更新) ---
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)


    def soft_update(self, local_model, target_model, tau):
        """
        软更新模型参数: θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def save_model(self, filename):
        """保存模型参数到文件"""
        torch.save({
            'actor_state_dict': self.actor_local.state_dict(),
            'critic_state_dict': self.critic_local.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict()
        }, filename)
        print(f"模型已保存到 {filename}")
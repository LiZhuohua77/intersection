"""
@file: ddpg.py
@description:
本文件提供了深度确定性策略梯度(DDPG)算法的完整PyTorch实现。
DDPG是一种用于连续动作空间的强化学习算法，结合了DQN和Actor-Critic的优点。

主要组件:

1. ReplayBuffer类:
   - 实现经验回放池，存储和采样(s,a,r,s',done)元组
   - 函数: add(), sample(), __len__()

2. Actor类:
   - 实现策略网络，将状态映射到确定性动作
   - 函数: __init__(), forward()

3. Critic类:
   - 实现价值网络，评估状态-动作对的Q值
   - 函数: __init__(), forward()

4. DDPGAgent类:
   - 实现DDPG算法的核心逻辑
   - 函数: __init__(), hard_update(), select_action(), step(), learn(),
     soft_update(), save_model(), load_model()

工作流程:
- 智能体通过Actor网络选择动作，并添加噪声以促进探索
- 经验存储在回放池中，定期从中采样批次数据进行学习
- Critic网络通过TD学习更新Q值估计
- Actor网络通过策略梯度更新，以最大化预期回报
- 目标网络通过软更新跟踪主网络，提高训练稳定性
"""
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

ENV_NAME = 'Intersection'
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
        
        参数:
            buffer_size: int, 经验池最大容量，当超过此容量时会移除最旧的经验
            batch_size: int, 每次学习采样的批次大小
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        """
        向经验池中添加一条经验
        
        参数:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否为终止状态的标志
        """
        e = (state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """
        从经验池中随机采样一个批次的经验
        
        返回:
            包含(states, actions, rewards, next_states, dones)的元组，
            每个元素都是批次大小的张量，已转移到指定设备
        """
        experiences = random.sample(self.memory, k=self.batch_size)
        
        # 将经验元组转换为批次张量
        states = torch.from_numpy(np.vstack([e[0] for e in experiences if e is not None])).float().to(DEVICE)
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None])).float().to(DEVICE)
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None])).float().to(DEVICE)
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences if e is not None])).float().to(DEVICE)
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)).float().to(DEVICE)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """
        返回当前经验池中的经验数量，用于检查是否有足够的样本开始学习
        
        返回:
            int: 当前经验池大小
        """
        return len(self.memory)

# --- Actor 网络 ---
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_high):
        """
        初始化Actor网络
        
        参数:
            state_dim: int, 状态空间维度
            action_dim: int, 动作空间维度
            action_high: ndarray, 动作空间上限值，用于缩放tanh输出
        """
        super(Actor, self).__init__()
        self.action_high = torch.tensor(action_high, dtype=torch.float32).to(DEVICE)
        
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, state):
        """
        前向传播函数，将状态映射为确定性动作
        
        参数:
            state: 输入状态张量
            
        返回:
            action: 缩放到实际动作范围的动作张量
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # 使用 tanh 将输出缩放到 [-1, 1]，然后乘以动作范围
        action = torch.tanh(self.fc3(x)) * self.action_high
        return action

# --- Critic 网络 ---
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        """
        初始化Critic网络
        
        参数:
            state_dim: int, 状态空间维度
            action_dim: int, 动作空间维度
        """
        super(Critic, self).__init__()
        
        # Q1 网络
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, state, action):
        """
        前向传播函数，计算给定状态-动作对的Q值
        
        参数:
            state: 状态张量
            action: 动作张量
            
        返回:
            q_value: 对应状态-动作对的Q值
        """
        # 将状态和动作在维度 1 上拼接
        x = torch.cat([state, action], dim=1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value

# --- DDPG Agent ---
class DDPGAgent():
    def __init__(self, state_dim, action_dim, action_high, writer=None):
        """
        初始化DDPG智能体
        
        参数:
            state_dim: int, 状态空间维度
            action_dim: int, 动作空间维度
            action_high: ndarray, 动作空间上限
            writer: SummaryWriter对象，用于TensorBoard可视化(可选)
        """
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)

        self.writer = writer
        self.learn_step_counter = 0

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
        """
        硬更新目标网络参数，直接复制源网络参数
        
        参数:
            target: 目标网络
            source: 源网络
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def select_action(self, state, add_noise=True):
        """
        根据当前状态选择动作
        
        参数:
            state: ndarray, 当前环境状态
            add_noise: bool, 是否添加噪声进行探索
            
        返回:
            ndarray: 裁剪到合法范围的动作值
        """
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
        """
        处理一步交互的经验，存入经验池并可能触发学习过程
        
        参数:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否为终止状态
        """
        self.replay_buffer.add(state, action, reward, next_state, done)

        # 如果回放池中有足够的数据，则进行一次学习
        if len(self.replay_buffer) > BATCH_SIZE:
            experiences = self.replay_buffer.sample()
            self.learn(experiences)

    def learn(self, experiences):
        """
        使用一个批次的经验数据来更新Actor和Critic网络
        
        参数:
            experiences: 包含(states, actions, rewards, next_states, dones)的元组
            
        流程:
            1. 更新Critic网络：最小化TD误差
            2. 更新Actor网络：最大化Q值期望
            3. 软更新两个目标网络
            4. 记录训练指标(如果有writer)
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

        if self.writer:
            self.writer.add_scalar('Loss/Critic_Loss', critic_loss.item(), self.learn_step_counter)
            self.writer.add_scalar('Loss/Actor_Loss', actor_loss.item(), self.learn_step_counter)
            self.writer.add_scalar('Value/Mean_Q_Value', Q_expected.mean().item(), self.learn_step_counter)
        
        self.learn_step_counter += 1 

    def soft_update(self, local_model, target_model, tau):
        """
        软更新模型参数: θ_target = τ*θ_local + (1-τ)*θ_target
        
        参数:
            local_model: 本地网络(源)
            target_model: 目标网络
            tau: float, 软更新系数，控制更新速度
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def save_model(self, filename):
        """
        保存模型参数到文件
        
        参数:
            filename: str, 保存文件的路径
        """
        torch.save({
            'actor_state_dict': self.actor_local.state_dict(),
            'critic_state_dict': self.critic_local.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict()
        }, filename)
        print(f"模型已保存到 {filename}")

    def load_model(self, filename):
        """
        从文件加载模型参数
        
        参数:
            filename: str, 模型文件的路径
            
        异常处理:
            处理文件不存在和其他加载错误
        """
        try:
            checkpoint = torch.load(filename, map_location=DEVICE)
            self.actor_local.load_state_dict(checkpoint['actor_state_dict'])
            self.critic_local.load_state_dict(checkpoint['critic_state_dict'])
            
            # 目标网络也需要加载，以确保一致性
            self.actor_target.load_state_dict(checkpoint['actor_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_state_dict'])
            
            # 优化器状态也可以选择性加载，以便继续训练
            # self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            # self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            
            print(f"模型已从 {filename} 加载")
        except FileNotFoundError:
            print(f"错误: 找不到模型文件 {filename}")
        except Exception as e:
            print(f"加载模型时发生错误: {e}")
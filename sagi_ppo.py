"""
@file: sagi_ppo.py
@description:
该文件实现了**场景感知几何引导近端策略优化**（Scenario-Aware Geometrically-Informed Proximal Policy Optimization，SAGI-PPO）
算法，这是一种针对带约束的马尔科夫决策过程（CMDP）的创新强化学习方法。SAGI-PPO旨在解决无信号交叉口环境中的安全高效决策问题，
通过几何引导的策略更新机制，在优化驾驶性能的同时确保路径跟踪的精确性。

算法核心架构:

1.  **网络结构设计 (Network Architecture):**
    - **Actor网络**：输出连续动作空间的高斯策略分布，使用双隐层前馈网络结构和tanh输出层。
    - **双Critic网络**：分别估计奖励价值（reward critic）和成本价值（cost critic），使能对
      驾驶效率和安全性的分离评估。
    - 使用正交初始化提升训练稳定性和收敛速度。

2.  **经验回放与优势估计 (Rollout Buffer & Advantage Estimation):**
    - 专门设计的`RolloutBuffer`类，同时存储状态、动作、奖励和成本轨迹。
    - 实现广义优势估计（GAE）算法，分别计算奖励优势函数A_R和成本优势函数A_C。
    - 通过优势函数归一化提升训练稳定性。

3.  **几何状态评估 (Geometric State Assessment):**
    - **成本盈余（c）**：评估当前策略相对于成本预算d的表现，c = J_C - d。
      负值表示策略当前安全，正值表示策略违反约束。
    - **梯度相关（p）**：通过计算优势函数的点积p = E[A_R·A_C]，近似估计奖励改进
      方向与成本增加方向之间的几何关系。正值表明两个目标存在冲突。

4.  **基于几何分析的动态策略更新 (Geometric Case-based Policy Update):**
    - **情景A（自由探索模式）**：当c < 0且p ≤ 0时，策略安全且奖励改进与约束不冲突，
      直接使用奖励优势A_R优化。
    - **情景C（紧急恢复模式）**：当c > 0且p > 0时，策略违反约束且冲突严重，
      使用成本优势的负值-A_C引导策略恢复安全状态。
    - **情景B（约束权衡模式）**：其他所有情况，使用拉格朗日方法平衡奖励和成本，
      优势函数为A' = (A_R - λ·A_C)/(1+λ)。

5.  **确定性策略提取 (Deterministic Action Extraction):**
    - 提供`get_deterministic_action`方法，从策略分布中提取均值作为确定性动作，
      用于评估和部署阶段。

SAGI-PPO算法相比传统的拉格朗日乘子方法，通过场景感知和几何引导的动态策略更新机制，
实现了更稳定的训练过程和更可靠的约束满足，特别适合应对无信号交叉口等高度不确定的复杂环境。
"""

# sagi_ppo.py

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Normal
import numpy as np

def init_weights(m):
    """
    初始化神经网络层的权重
    
    Args:
        m (nn.Module): 需要初始化的神经网络层
        
    Notes:
        使用正交初始化方法初始化线性层权重，并将偏置项初始化为0
        正交初始化有助于稳定深度网络的训练过程
    """
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
        nn.init.constant_(m.bias, 0.0)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        """
        初始化Actor网络
        
        Args:
            state_dim (int): 状态空间的维度
            action_dim (int): 动作空间的维度
            hidden_dim (int): 隐藏层的神经元数量
            
        Notes:
            Actor网络使用两层隐藏层的全连接网络，输出层使用tanh激活函数
            网络同时学习动作的均值和标准差对数，用于构建高斯策略分布
        """
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
        """
        前向传播计算策略分布
        
        Args:
            state (torch.Tensor): 输入状态张量
            
        Returns:
            torch.distributions.Normal: 表示动作的高斯分布，包含均值和标准差
            
        Notes:
            输出是一个多变量高斯分布，mean是tanh输出的动作均值，
            std是log_std的指数，确保标准差为正值
        """
        mean = self.net(state)
        std = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        return dist

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        """
        初始化Critic网络
        
        Args:
            state_dim (int): 状态空间的维度
            hidden_dim (int): 隐藏层的神经元数量
            
        Notes:
            Critic网络使用两层隐藏层的全连接网络，输出一个标量值
            用于估计状态价值函数（奖励价值或成本价值）
        """
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
        """
        前向传播计算状态价值
        
        Args:
            state (torch.Tensor): 输入状态张量
            
        Returns:
            torch.Tensor: 状态价值估计
        """
        return self.net(state)

class RolloutBuffer:
    def __init__(self, buffer_size, state_dim, action_dim, gamma, gae_lambda):
        """
        初始化经验回放缓冲区
        
        Args:
            buffer_size (int): 缓冲区大小，即存储的最大时间步数
            state_dim (int): 状态空间的维度
            action_dim (int): 动作空间的维度
            gamma (float): 折扣因子，用于计算折扣回报
            gae_lambda (float): GAE lambda参数，用于平衡偏差和方差
            
        Notes:
            为SAGI-PPO设计的特殊缓冲区，同时存储奖励和成本相关数据
            支持广义优势估计(GAE)计算
        """
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.costs = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.values_r = np.zeros(buffer_size, dtype=np.float32)
        self.values_c = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        self.ptr, self.path_start_idx = 0, 0
        self.next_value_r = 0
        self.next_value_c = 0        
    
    def store(self, state, action, reward, cost, done, value_r, value_c, log_prob):
        """
        存储单个交互步骤的数据
        
        Args:
            state (np.ndarray): 环境状态
            action (np.ndarray): 执行的动作
            reward (float): 获得的奖励
            cost (float): 产生的成本/安全违反值
            done (bool): 回合是否结束
            value_r (float): 奖励Critic估计的状态价值
            value_c (float): 成本Critic估计的状态价值
            log_prob (float): 动作的对数概率
        """
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.costs[self.ptr] = cost
        self.dones[self.ptr] = done
        self.values_r[self.ptr] = value_r
        self.values_c[self.ptr] = value_c
        self.log_probs[self.ptr] = log_prob
        self.ptr += 1

    def finish_path(self, last_value_r, last_value_c):
        """
        结束当前轨迹，记录最终状态的价值估计
        
        Args:
            last_value_r (float): 最终状态的奖励价值估计
            last_value_c (float): 最终状态的成本价值估计
            
        Notes:
            用于正确处理回合结束时的价值估计，这对GAE计算很重要
        """
        if self.ptr > 0:  # 确保buffer不为空
            self.next_value_r = last_value_r
            self.next_value_c = last_value_c
        
        self.path_start_idx = self.ptr

    def _calculate_advantages(self, values, rewards_or_costs, dones):
        """
        计算广义优势估计(GAE)
        
        Args:
            values (np.ndarray): 状态价值估计数组
            rewards_or_costs (np.ndarray): 奖励或成本数组
            dones (np.ndarray): 终止标志数组
            
        Returns:
            np.ndarray: 计算得到的优势函数数组
            
        Notes:
            内部方法，分别用于计算奖励优势和成本优势
            使用GAE算法，结合TD误差和折扣因子
        """
        advantages = np.zeros_like(rewards_or_costs)
        last_gae_lam = 0
        for t in reversed(range(self.buffer_size)):
            if t == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones[t]
                next_values = self.next_value_r if values is self.values_r else self.next_value_c
            else:
                next_non_terminal = 1.0 - dones[t]
                next_values = values[t + 1]
            delta = rewards_or_costs[t] + self.gamma * next_values * next_non_terminal - values[t]
            advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
        return advantages

    def get(self):
        """
        获取处理后的训练数据批次
        
        Returns:
            dict: 包含处理后训练数据的字典，键包括:
                - states: 状态张量
                - actions: 动作张量
                - log_probs: 动作对数概率
                - advantages_r: 归一化的奖励优势函数
                - advantages_c: 归一化的成本优势函数
                - returns_r: 归一化的奖励回报
                - returns_c: 归一化的成本回报
                
        Notes:
            此方法在缓冲区满时调用，计算优势函数和回报
            对优势函数和回报进行归一化以提高训练稳定性
        """
        assert self.ptr == self.buffer_size # Buffer must be full
        self.ptr = 0
        
        # --- 核心优势计算 ---
        advantages_r = self._calculate_advantages(self.values_r, self.rewards, self.dones)
        advantages_c = self._calculate_advantages(self.values_c, self.costs, self.dones)
        
        returns_r = advantages_r + self.values_r
        returns_c = advantages_c + self.values_c

        # 添加与ppo.py相同的returns归一化
        returns_r = (returns_r - np.mean(returns_r)) / (np.std(returns_r) + 1e-8)
        returns_c = (returns_c - np.mean(returns_c)) / (np.std(returns_c) + 1e-8)

        # --- 归一化优势函数 ---
        advantages_r = (advantages_r - np.mean(advantages_r)) / (np.std(advantages_r) + 1e-8)
        advantages_c = (advantages_c - np.mean(advantages_c)) / (np.std(advantages_c) + 1e-8)

        # 转换为 PyTorch Tensors
        batch = dict(
            states=torch.as_tensor(self.states, dtype=torch.float32),
            actions=torch.as_tensor(self.actions, dtype=torch.float32),
            log_probs=torch.as_tensor(self.log_probs, dtype=torch.float32),
            advantages_r=torch.as_tensor(advantages_r, dtype=torch.float32),
            advantages_c=torch.as_tensor(advantages_c, dtype=torch.float32),
            returns_r=torch.as_tensor(returns_r, dtype=torch.float32),
            returns_c=torch.as_tensor(returns_c, dtype=torch.float32)
        )
        return batch


class SAGIPPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, hidden_dim=256, gamma=0.99, gae_lambda=0.95,
                 clip_epsilon=0.2, update_epochs=10, cost_limit=5.0):
        """
        初始化SAGI-PPO代理
        
        Args:
            state_dim (int): 状态空间的维度
            action_dim (int): 动作空间的维度
            lr (float): 学习率
            hidden_dim (int): 隐藏层的神经元数量
            gamma (float): 折扣因子
            gae_lambda (float): GAE lambda参数
            clip_epsilon (float): PPO裁剪参数
            update_epochs (int): 每批数据的更新轮数
            cost_limit (float): 成本预算上限d
            
        Notes:
            SAGI-PPO代理包含一个Actor网络和两个Critic网络(奖励和成本)
            使用拉格朗日乘子方法处理约束，但根据场景动态调整更新策略
        """
        # --- 网络定义 ---
        self.actor = Actor(state_dim, action_dim, hidden_dim)
        self.critic_r = Critic(state_dim, hidden_dim) # 奖励Critic
        self.critic_c = Critic(state_dim, hidden_dim) # 成本Critic

        # --- 优化器 ---
        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr)
        self.critic_r_optimizer = Adam(self.critic_r.parameters(), lr=lr)
        self.critic_c_optimizer = Adam(self.critic_c.parameters(), lr=lr)

        # --- PPO 超参数 ---
        self.clip_epsilon = clip_epsilon
        self.update_epochs = update_epochs

        # --- SAGI-PPO 核心参数 ---
        self.cost_limit = cost_limit  # 成本预算 d
        self.lambda_lr = 0.01         # 拉格朗日乘子学习率 alpha_lambda
        self.lambda_val = 1.0         # 拉格朗日乘子 lambda (初始值)

    def select_action(self, state):
        """
        根据当前状态和策略选择动作
        
        Args:
            state (np.ndarray): 环境状态
            
        Returns:
            tuple: (动作数组, 奖励价值估计, 成本价值估计, 动作对数概率)
            
        Notes:
            用于训练过程中的探索，从策略分布中采样动作
            同时返回奖励和成本价值估计，用于后续的优势计算
        """
        state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            dist = self.actor(state)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(axis=-1)
            value_r = self.critic_r(state)
            value_c = self.critic_c(state)
        return action.numpy().flatten(), value_r.item(), value_c.item(), log_prob.item()

    def get_deterministic_action(self, state):
        """
        获取确定性动作(策略分布的均值)
        
        Args:
            state (np.ndarray): 环境状态
            
        Returns:
            np.ndarray: 确定性动作数组
            
        Notes:
            用于评估和部署阶段，直接使用策略分布的均值作为动作
            无需探索，因此不进行随机采样
        """
        state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            dist = self.actor(state)
            action = dist.mean # <--- 使用均值而不是sample()
        return action.numpy().flatten()

    def update(self, buffer: RolloutBuffer, writer, global_step):
        """
        使用收集的轨迹数据更新网络参数
        
        Args:
            buffer (RolloutBuffer): 包含训练数据的缓冲区
            writer: TensorBoard日志记录器，用于记录训练指标
            global_step (int): 全局训练步数
            
        Notes:
            SAGI-PPO的核心更新逻辑，包括:
            1. 计算成本盈余c和梯度方向点积p
            2. 根据c和p的值确定更新模式(A/B/C三种情况)
            3. 根据不同模式使用不同的目标优势函数更新Actor
            4. 分别更新奖励Critic和成本Critic
            5. 更新拉格朗日乘子lambda
            
            三种更新模式:
            - A模式(自由探索): 当前安全且无冲突，使用奖励优势
            - C模式(紧急恢复): 当前危险且有冲突，使用负成本优势
            - B模式(约束权衡): 其他情况，使用加权组合优势
        """
        batch = buffer.get()
        
        # --- 核心要素计算 ---
        # 1. 计算当前批次的平均成本 J_C,k 和成本盈余 c
        current_cost = torch.mean(batch['returns_c']).item()
        cost_surplus = current_cost - self.cost_limit # c = J_C - d
        
        # 2. 计算梯度方向的点积 p (用优势函数的相关性来近似)
        adv_dot_product = torch.mean(batch['advantages_r'] * batch['advantages_c']).item() # p = E[A_R * A_C]
        
        # --- SAGI-PPO 分类决策 ---
        # 根据当前的安全形势，动态选择本次更新的目标优势函数 A'
        writer.add_scalar("sagi/cost_surplus_c", cost_surplus, global_step)
        writer.add_scalar("sagi/adv_dot_product_p", adv_dot_product, global_step)
        writer.add_scalar("sagi/lambda_val", self.lambda_val, global_step)
        print(f"Updating... Cost Surplus(c): {cost_surplus:.3f}, Adv Dot(p): {adv_dot_product:.3f}, Lambda: {self.lambda_val:.3f}")

        update_mode = -1
        if cost_surplus < 0 and adv_dot_product <= 0:
            # Case A: 自由探索模式 (当前安全，且提升奖励不增加成本)
            print("Mode: [A] Free Exploration")
            target_advantages = batch['advantages_r']
            update_mode = 0
        elif cost_surplus > 0 and adv_dot_product > 0:
            # Case C: 紧急恢复模式 (当前危险，且提升奖励会加剧危险)
            print("Mode: [C] Emergency Recovery")
            target_advantages = -batch['advantages_c'] # 最大化成本优势的相反数，即最小化成本
            update_mode = 2
        else:
            # Case B: 约束权衡模式 (其他所有需要权衡的情况)
            print("Mode: [B] Constrained Trade-off")
            target_advantages = (batch['advantages_r'] - self.lambda_val * batch['advantages_c']) / (1 + self.lambda_val)
            update_mode = 1

        writer.add_scalar("sagi/update_mode", update_mode, global_step)

        # --- PPO 核心更新循环 ---
        for i in range(self.update_epochs):
            # 1. Actor 更新
            dist = self.actor(batch['states'])
            new_log_probs = dist.log_prob(batch['actions']).sum(axis=-1)
            ratio = torch.exp(new_log_probs - batch['log_probs'])
            
            surr1 = ratio * target_advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * target_advantages
            actor_loss = -torch.mean(torch.min(surr1, surr2))
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # 2. Critic R 更新
            values_r = self.critic_r(batch['states']).squeeze()
            critic_r_loss = nn.MSELoss()(values_r, batch['returns_r'])
            
            self.critic_r_optimizer.zero_grad()
            critic_r_loss.backward()
            self.critic_r_optimizer.step()
            
            # 3. Critic C 更新
            values_c = self.critic_c(batch['states']).squeeze()
            critic_c_loss = nn.MSELoss()(values_c, batch['returns_c'])

            self.critic_c_optimizer.zero_grad()
            critic_c_loss.backward()
            self.critic_c_optimizer.step()

            epoch_step = global_step + (i / self.update_epochs) # 映射到全局步数
            
            writer.add_scalar("losses/actor_loss", actor_loss.item(), epoch_step)
            writer.add_scalar("losses/critic_r_loss", critic_r_loss.item(), epoch_step)
            writer.add_scalar("losses/critic_c_loss", critic_c_loss.item(), epoch_step)
            
            # 记录策略的熵和标准差，以监控探索程度
            entropy = dist.entropy().mean().item()
            std_dev = self.actor.log_std.exp().mean().item()
            writer.add_scalar("policy/entropy", entropy, epoch_step)
            writer.add_scalar("policy/std_dev", std_dev, epoch_step)

        # --- 拉格朗日乘子 Lambda 更新 ---
        # 注意：只有在Case B中才会用到lambda，但我们始终更新它，使其能平滑地跟踪成本变化
        self.lambda_val = max(0, self.lambda_val + self.lambda_lr * cost_surplus)
"""
@file: train.py
@description:
该文件是强化学习智能体的**主训练脚本**。
它负责编排整个训练流程：初始化环境和DDPG智能体，然后在多个回合（episodes）中
进行交互、学习和更新，最后保存训练好的模型并可视化训练结果。
此脚本通常在无图形界面的服务器或后台运行，以最大化训练效率。

核心流程:

1.  **初始化 (Initialization):**
    - 设置 `TensorBoard` 的 `SummaryWriter`，用于记录和可视化训练过程中的关键指标
      （如奖励、损失函数等）。
    - 在**无渲染模式**下创建 `TrafficEnv` 环境，以加速训练。
    - 根据环境的观测和动作空间维度，初始化 `DDPGAgent`。

2.  **训练主循环 (Main Training Loop):**
    - 包含一个外层的回合循环（episodes）和一个内层的步数循环（steps）。
    - 在循环中执行标准的强化学习交互流程：
        a. 智能体根据当前状态选择一个带探索噪声的动作 (`agent.select_action`)。
        b. 环境执行该动作并返回结果 (`env.step`)。
        c. 将这次交互的完整经验 `(s, a, r, s', done)` 存入智能体的经验回放池并
           触发学习过程 (`agent.step`)。

3.  **日志与监控 (Logging and Monitoring):**
    - 每个回合结束后，将该回合的总奖励记录到 `TensorBoard` 中。
    - 可以在终端通过 `tensorboard --logdir=runs` 命令来实时监控智能体的学习曲线，
      判断训练是否收敛。

4.  **模型保存 (Model Checkpointing):**
    - 训练过程中会周期性地（例如每50个回合）调用 `agent.save_model()` 来保存当前
      的模型权重。这对于断点续训和后续的模型评估至关重要。

5.  **结果可视化 (Result Visualization):**
    - 全部训练结束后，脚本会使用 `matplotlib` 绘制整个训练过程中的奖励变化曲线，
      并将其保存为图片，为最终的训练效果提供一个直观的总结。
"""

import gymnasium as gym
import numpy as np
import torch # 假设您使用PyTorch实现DDPG
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from traffic_env import TrafficEnv
from ddpg import DDPGAgent 

def main():
    # --- 1. 初始化 ---
    writer = SummaryWriter('runs/ddpg_traffic_v1')  # 初始化 TensorBoard
    # 创建 TrafficEnv 环境，注意这里我们不需要渲染，训练会更快
    env = TrafficEnv()
    
    # 设置超参数
    MAX_EPISODES = 500       # 总共训练的回合数
    MAX_STEPS_PER_EPISODE = 1000 # 每个回合的最大步数
    
    # 从环境中获取状态和动作空间的维度
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_high = env.action_space.high 
    
    # 初始化您的DDPG Agent
    # 您需要根据您自己类的定义来传入参数
    agent = DDPGAgent(state_dim=state_dim, action_dim=action_dim, action_high=action_high, writer=writer)

    # 用于记录每个回合的奖励，方便后续绘图
    episode_rewards = []
    
    print("--- 训练开始 ---")
    print(f"状态空间维度: {state_dim}, 动作空间维度: {action_dim}")

    # --- 2. 主训练循环 ---
    
    for episode in range(MAX_EPISODES):
        # 重置环境，获取初始状态
        state, info = env.reset(options={'scenario': 'agent_only'})
        episode_reward = 0
        
        for step in range(MAX_STEPS_PER_EPISODE):
            # a. Agent选择动作 (需要加入探索噪声)
            action = agent.select_action(state)
            
            # b. 环境执行动作
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # 判断回合是否结束
            done = terminated or truncated
            
            # c. 将经验存入Replay Buffer
            agent.step(state, action, reward, next_state, done)
            
            # d. 更新当前状态
            state = next_state
            episode_reward += reward
            
            # f. 如果回合结束，则跳出
            if done:
                break
        
        # --- 回合结束后的处理 ---
        episode_rewards.append(episode_reward)
        writer.add_scalar('Reward/Episode_Reward', episode_reward, episode)
        print(f"Episode: {episode+1}, Steps: {step+1}, Total Reward: {episode_reward:.2f}")

        # (可选) 每隔一定回合数保存一次模型
        if (episode + 1) % 50 == 0:
            agent.save_model(f'ddpg_episode_{episode+1}.pth')
            print(f"--- 模型已在第 {episode+1} 回合保存 ---")

    # --- 3. 训练结束 ---
    print("--- 训练结束 ---")
    env.close()

    # 绘制奖励曲线
    plt.figure()
    plt.plot(episode_rewards)
    plt.title("Episode Rewards over Time")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.savefig("ddpg_training_rewards.png")
    plt.show()

if __name__ == '__main__':
    main()
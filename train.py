# train.py

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
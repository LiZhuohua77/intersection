"""
@file: train.py
@description:
该文件是用于**训练和保存**强化学习智能体的主脚本。
它支持两种强化学习算法：标准的PPO（近端策略优化）和SAGI-PPO（安全增强型PPO），
通过命令行参数灵活配置训练过程，并提供完整的日志记录、模型保存和训练状态可视化功能。

核心流程:

1.  **命令行参数配置 (Command Line Configuration):**
    - 提供丰富的命令行参数选项，包括算法选择、训练轮次、缓冲区大小、学习率等
      超参数，以及随机种子设置，确保实验的可重复性和可配置性。
    - 支持从已有检查点恢复训练的功能，便于长时间训练任务的中断和继续。

2.  **环境初始化与日志设置 (Environment Setup & Logging):**
    - 初始化交通仿真环境`TrafficEnv`，统一设置随机种子以确保实验可重复。
    - 配置TensorBoard日志记录器，自动创建时间戳命名的运行目录，记录训练过程中的
      各项指标，便于后续分析和可视化。
    - 创建专门的日志文件夹，记录训练配置和详细的训练数据。

3.  **算法初始化 (Algorithm Initialization):**
    - 根据命令行参数动态选择并初始化对应的智能体类型（PPOAgent或SAGIPPOAgent）。
    - 为不同算法配置相应的经验回放缓冲区，PPO使用标准Buffer，SAGI-PPO使用带有
      成本评估功能的Buffer。
    - 支持从已保存的模型文件中加载网络权重，实现训练的延续性。

4.  **训练主循环 (Training Loop):**
    - 采用基于回合(episode)的训练方式，每个回合通过与环境交互收集轨迹数据。
    - 实现经验采样与策略更新的分离：当Buffer满时触发网络参数的批量更新。
    - 全面记录每个回合的奖励、成本和长度等关键指标，并周期性保存到CSV文件。
    - 定期保存训练检查点，确保长时间训练过程的容错性和可恢复性。

5.  **模型保存与结果统计 (Model Saving & Statistics):**
    - 在训练结束时保存最终模型，保持与检查点相同的命名规范。
    - 生成完整的训练统计数据文件，便于后续分析和可视化。
    - 为不同算法保存相应的网络组件：PPO保存actor和critic网络，SAGI-PPO则额外
      保存成本评估网络。

本脚本与evaluate.py脚本构成完整的训练-评估工作流，前者负责模型的训练和保存，
后者负责加载训练好的模型并进行可视化评估和性能分析。
"""

import argparse
import torch
import time
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
from traffic_env import TrafficEnv
import numpy as np
import random

# --- 动态导入算法 ---
from sagi_ppo import SAGIPPOAgent, RolloutBuffer as SAGIRolloutBuffer
from ppo import PPOAgent, RolloutBuffer as PPORolloutBuffer

import pandas as pd
import os

def set_seed(seed):
    """设置所有可能的随机种子，确保实验可重复"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Train an agent for the traffic intersection environment.")
    
    # --- 实验与算法选择 ---
    parser.add_argument("--algo", type=str, default="ppo", choices=["sagi_ppo", "ppo"], help="The reinforcement learning algorithm to use.")
    
    # --- 训练过程参数 ---
    parser.add_argument("--total-episodes", type=int, default=100000, help="Total episodes to train the agent.")
    parser.add_argument("--buffer-size", type=int, default=2048, help="Size of the rollout buffer.")
    parser.add_argument("--update-epochs", type=int, default=2, help="Number of epochs to update the policy per rollout.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    # --- 算法超参数 ---
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate for the optimizers.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor gamma.")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="Lambda for the GAE advantage calculation.")
    parser.add_argument("--clip-epsilon", type=float, default=0.2, help="Clipping parameter epsilon for PPO.")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Dimension of the hidden layers.")

    # --- SAGI-PPO 专属参数 ---
    parser.add_argument("--cost-limit", type=float, default=30.0, help="Cost limit 'd' for SAGI-PPO.")
    
    # --- 继续训练参数 ---
    parser.add_argument("--resume", action="store_true", help="Whether to resume training from a checkpoint.")
    parser.add_argument("--model-path", type=str, default="D:\Code\intersection\models\ppo_20250825-160710", help="Path to the model directory for resuming training.")
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    # --- 设置随机种子 ---
    set_seed(args.seed)
    print(f"--- Setting random seed to {args.seed} ---")
    
    # --- 设置TensorBoard日志 ---
    run_name = f"{args.algo}_{time.strftime('%Y%m%d-%H%M%S')}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    log_dir = f"logs/{run_name}"  # <--- 定义日志文件夹路径
    os.makedirs(log_dir, exist_ok=True) # <--- 创建文件夹

    # --- 创建环境 ---
    env = TrafficEnv()
    # 为环境设置种子
    env.reset(seed=args.seed)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # --- 根据参数选择并初始化算法、Buffer ---
    if args.algo == "sagi_ppo":
        print(f"--- Using SAGI-PPO ---")
        agent = SAGIPPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            lr=args.lr,
            hidden_dim=args.hidden_dim,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_epsilon=args.clip_epsilon,
            update_epochs=args.update_epochs,
            cost_limit=args.cost_limit
        )
        buffer = SAGIRolloutBuffer(args.buffer_size, state_dim, action_dim, args.gamma, gae_lambda=0.95)
    
    elif args.algo == "ppo":
        print(f"--- Using Standard PPO (Baseline) ---")
        agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            lr=args.lr,
            hidden_dim=args.hidden_dim,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_epsilon=args.clip_epsilon,
            update_epochs=args.update_epochs
        )
        buffer = PPORolloutBuffer(args.buffer_size, state_dim, action_dim, args.gamma, gae_lambda=0.95)

    # --- 如果需要，从现有模型加载参数 ---
    if args.resume and args.model_path is not None:
        print(f"--- Resuming training from {args.model_path} ---")
        if args.algo == "sagi_ppo":
            actor_path = os.path.join(args.model_path, "sagi_ppo_actor.pth")
            critic_r_path = os.path.join(args.model_path, "sagi_ppo_critic_r.pth")
            critic_c_path = os.path.join(args.model_path, "sagi_ppo_critic_c.pth")
            
            if os.path.exists(actor_path) and os.path.exists(critic_r_path) and os.path.exists(critic_c_path):
                agent.actor.load_state_dict(torch.load(actor_path))
                agent.critic_r.load_state_dict(torch.load(critic_r_path))
                agent.critic_c.load_state_dict(torch.load(critic_c_path))
                print("Successfully loaded SAGI-PPO model")
            else:
                print("Warning: Could not find model files, starting with fresh model")
        
        elif args.algo == "ppo":
            actor_path = os.path.join(args.model_path, "ppo_actor.pth")
            critic_path = os.path.join(args.model_path, "ppo_critic.pth")
            
            if os.path.exists(actor_path) and os.path.exists(critic_path):
                agent.actor.load_state_dict(torch.load(actor_path))
                agent.critic.load_state_dict(torch.load(critic_path))
                print("Successfully loaded PPO model")
            else:
                print("Warning: Could not find model files, starting with fresh model")

    # --- 保存配置信息 ---
    config_path = os.path.join(log_dir, "config.txt")
    with open(config_path, 'w') as f:
        f.write(f"Algorithm: {args.algo}\n")
        f.write(f"Seed: {args.seed}\n")
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")
    print(f"Configuration saved to {config_path}")
    
    # --- 训练主循环（基于episode） ---
    total_timesteps = 0
    episode_count = 0

    # 创建一个字典用于记录训练指标
    training_stats = {
        'episode_rewards': [],
        'episode_costs': [],
        'episode_lengths': []
    }
    
    for episode in range(args.total_episodes):
        state, _ = env.reset(options={'scenario': 'agent_only', 'algo': args.algo})
        episode_reward = 0
        episode_cost = 0
        episode_len = 0
        done = False
        
        while not done:
            # 选择动作
            if args.algo == "sagi_ppo":
                action, value_r, value_c, log_prob = agent.select_action(state)
            elif args.algo == "ppo":
                action, value, log_prob = agent.select_action(state)
            
            # 执行动作
            next_state, reward, terminated, truncated, info = env.step(action)
            cost = info.get('cost', 0)
            done = terminated or truncated
            
            # 存储经验
            if args.algo == "sagi_ppo":
                buffer.store(state, action, reward, cost, done, value_r, value_c, log_prob)
            elif args.algo == "ppo":
                buffer.store(state, action, reward, done, value, log_prob)
            
            # 更新状态和记录
            state = next_state
            episode_reward += reward
            episode_cost += cost
            episode_len += 1
            total_timesteps += 1
            
            # 如果buffer满了，进行一次更新
            if buffer.ptr == args.buffer_size:
                agent.update(buffer, writer, total_timesteps)
        
        # 一个episode结束后记录信息
        episode_count += 1
        print(f"Episode: {episode+1}/{args.total_episodes}, Algo: {args.algo}, EpReward: {episode_reward:.2f}, EpCost: {episode_cost:.2f}, EpLen: {episode_len}")
        writer.add_scalar("charts/episode_reward", episode_reward, episode)
        writer.add_scalar("charts/episode_cost", episode_cost, episode)
        writer.add_scalar("charts/episode_length", episode_len, episode)
        writer.add_scalar("charts/total_timesteps", total_timesteps, episode)
        
        # 保存训练统计信息
        training_stats['episode_rewards'].append(episode_reward)
        training_stats['episode_costs'].append(episode_cost)
        training_stats['episode_lengths'].append(episode_len)
        # 保存前1个episode的详细日志
        if episode < 1:
            if 'episode_log' in info:
                log_data = info['episode_log']
                df = pd.DataFrame(log_data)
                log_filename = os.path.join(log_dir, f"episode_{episode+1}_log.csv")
                df.to_csv(log_filename, index=False)
                print(f"成功保存回合 {episode+1} 的日志到: {log_filename}")

        # 每100个episode保存一次中间模型
        if (episode + 1) % 10000 == 0:
            model_save_dir = f"models/{run_name}/checkpoints"
            os.makedirs(model_save_dir, exist_ok=True)
            
            if args.algo == "sagi_ppo":
                torch.save(agent.actor.state_dict(), os.path.join(model_save_dir, f"sagi_ppo_actor_ep{episode+1}.pth"))
                torch.save(agent.critic_r.state_dict(), os.path.join(model_save_dir, f"sagi_ppo_critic_r_ep{episode+1}.pth"))
                torch.save(agent.critic_c.state_dict(), os.path.join(model_save_dir, f"sagi_ppo_critic_c_ep{episode+1}.pth"))
            elif args.algo == "ppo":
                torch.save(agent.actor.state_dict(), os.path.join(model_save_dir, f"ppo_actor_ep{episode+1}.pth"))
                torch.save(agent.critic.state_dict(), os.path.join(model_save_dir, f"ppo_critic_ep{episode+1}.pth"))
            
            print(f"Saved checkpoint at episode {episode+1}")
    
    env.close()
    writer.close()
    
    # 保存训练统计数据
    stats_df = pd.DataFrame(training_stats)
    stats_df.to_csv(os.path.join(log_dir, "training_stats.csv"), index=False)
    
    # --- 保存最终训练好的模型 ---
    print("--- Saving trained model ---")
    model_save_dir = f"models/{run_name}"
    os.makedirs(model_save_dir, exist_ok=True)

    if args.algo == "sagi_ppo":
        torch.save(agent.actor.state_dict(), os.path.join(model_save_dir, "sagi_ppo_actor.pth"))
        torch.save(agent.critic_r.state_dict(), os.path.join(model_save_dir, "sagi_ppo_critic_r.pth"))
        torch.save(agent.critic_c.state_dict(), os.path.join(model_save_dir, "sagi_ppo_critic_c.pth"))
    elif args.algo == "ppo":
        torch.save(agent.actor.state_dict(), os.path.join(model_save_dir, "ppo_actor.pth"))
        torch.save(agent.critic.state_dict(), os.path.join(model_save_dir, "ppo_critic.pth"))

    print(f"Model saved to: {model_save_dir}")
    print("--- Training finished ---")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Total episodes: {episode_count}")

if __name__ == "__main__":
    main()
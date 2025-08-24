import argparse
import torch
import time
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
from traffic_env import TrafficEnv

# --- 动态导入算法 ---
from sagi_ppo import SAGIPPOAgent, RolloutBuffer as SAGIRolloutBuffer
from ppo import PPOAgent, RolloutBuffer as PPORolloutBuffer

import pandas as pd
import os

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Train an agent for the traffic intersection environment.")
    
    # --- 实验与算法选择 ---
    parser.add_argument("--algo", type=str, default="ppo", choices=["sagi_ppo", "ppo"], help="The reinforcement learning algorithm to use.")
    
    # --- 训练过程参数 ---
    parser.add_argument("--total-episodes", type=int, default=300, help="Total episodes to train the agent.")
    parser.add_argument("--buffer-size", type=int, default=2048, help="Size of the rollout buffer.")
    parser.add_argument("--update-epochs", type=int, default=2, help="Number of epochs to update the policy per rollout.")

    # --- 算法超参数 ---
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the optimizers.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor gamma.")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="Lambda for the GAE advantage calculation.")
    parser.add_argument("--clip-epsilon", type=float, default=0.2, help="Clipping parameter epsilon for PPO.")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Dimension of the hidden layers.")

    # --- SAGI-PPO 专属参数 ---
    parser.add_argument("--cost-limit", type=float, default=90.0, help="Cost limit 'd' for SAGI-PPO.")
    
    # --- 继续训练参数 ---
    parser.add_argument("--resume", action="store_true", help="Whether to resume training from a checkpoint.")
    parser.add_argument("--model-path", type=str, default="D:\Code\intersection\models\ppo_20250824-095015", help="Path to the model directory for resuming training.")
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
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
        
        # 保存前5个episode的详细日志
        if episode < 5:
            if 'episode_log' in info:
                log_data = info['episode_log']
                df = pd.DataFrame(log_data)
                log_filename = os.path.join(log_dir, f"episode_{episode+1}_log.csv")
                df.to_csv(log_filename, index=False)
                print(f"成功保存回合 {episode+1} 的日志到: {log_filename}")

        # 每100个episode保存一次中间模型
        if (episode + 1) % 100 == 0:
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
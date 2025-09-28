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
from agent import HybridFeaturesExtractor 
from config import *
import numpy as np
import random

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

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
    parser.add_argument("--n-envs", type=int, default=1, help="Number of parallel environments to use for training.")
    
    # --- 训练过程参数 ---
    parser.add_argument("--total-timesteps", type=int, default=200000, help="Total timesteps to train the agent.")
    parser.add_argument("--save-freq", type=int, default=50000, help="Save a checkpoint every N timesteps.")

    parser.add_argument("--buffer-size", type=int, default=2048, help="Size of the rollout buffer.")
    parser.add_argument("--update-epochs", type=int, default=2, help="Number of epochs to update the policy per rollout.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    # --- 算法超参数 ---
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate for the optimizers.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor gamma.")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="Lambda for the GAE advantage calculation.")
    parser.add_argument("--clip-epsilon", type=float, default=0.2, help="Clipping parameter epsilon for PPO.")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Dimension of the hidden layers.")
    parser.add_argument("--rnn-hidden-dim", type=int, default=64, help="Dimension of the GRU hidden layers for trajectory encoding.")


    # --- SAGI-PPO 专属参数 ---
    parser.add_argument("--cost-limit", type=float, default=30.0, help="Cost limit 'd' for SAGI-PPO.")
    
    # --- 继续训练参数 ---
    parser.add_argument("--resume-from", type=str, default=None, help="Path to a .zip model file to resume training from.")
    parser.add_argument("--model-path", type=str, default="D:\Code\intersection\models\ppo_20250829-083302", help="Path to the model directory for resuming training.")
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    set_seed(args.seed)
    
    run_name = f"{args.algo}_{time.strftime('%Y%m%d-%H%M%S')}"
    log_dir = f"logs/{run_name}"
    model_save_dir = f"models/{run_name}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)

    # --- 1. [核心改进] 创建并行化的 Gym 环境 ---
    # 这是提升训练速度的关键
    env = make_vec_env(TrafficEnv, n_envs=args.n_envs, vec_env_cls=SubprocVecEnv,
                       env_kwargs=dict(scenario='head_on_conflict'))

    # --- 2. [核心改进] 定义自定义策略网络参数 ---
    policy_kwargs = dict(
        features_extractor_class=HybridFeaturesExtractor,
        features_extractor_kwargs=dict(
            av_obs_dim=AV_OBS_DIM,
            hv_obs_dim=HV_OBS_DIM,
            traj_len=PREDICTION_HORIZON,
            traj_feat_dim=FEATURES_PER_STEP,
            rnn_hidden_dim=args.rnn_hidden_dim
        ),
        net_arch=dict(pi=[args.hidden_dim, args.hidden_dim], vf=[args.hidden_dim, args.hidden_dim])
    )

    # --- 3. [核心改进] 初始化或加载SB3模型 ---
    if args.resume_from:
        print(f"--- Resuming training from {args.resume_from} ---")
        model = PPO.load(args.resume_from, env=env, tensorboard_log=f"runs/{run_name}")
    else:
        print(f"--- Starting a new training run ---")
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=f"runs/{run_name}",
            # 可以直接在这里设置PPO的超参数
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            learning_rate=3e-4
        )
        
    # --- 4. [核心改进] 使用SB3的回调函数来保存检查点 ---
    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.save_freq // args.n_envs, 1),
        save_path=os.path.join(model_save_dir, "checkpoints"),
        name_prefix="ppo_model"
    )

    # --- 5. [核心改进] 开始训练 ---
    # SB3的learn方法封装了所有复杂的训练循环
    print("--- Starting training ---")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=checkpoint_callback,
        reset_num_timesteps=(not args.resume_from) # 如果是新训练，则重置时间步计数
    )

    # --- 6. [核心改进] 保存最终模型 ---
    final_model_path = os.path.join(model_save_dir, "final_model.zip")
    model.save(final_model_path)
    
    print("--- Training finished ---")
    print(f"Final model saved to: {final_model_path}")
    
    env.close()

if __name__ == "__main__":
    main()
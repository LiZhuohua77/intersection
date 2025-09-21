"""
@file: evaluate.py
@description:
该文件用于评估和可视化已训练好的强化学习智能体在交通场景中的表现。
支持加载PPO和SAGI-PPO算法的模型，在可视化仿真环境中运行并评估智能体性能，
生成量化指标并记录详细日志数据。

主要函数:
- set_seed(seed): 设置所有随机数生成器的种子，确保实验可重复性
- parse_args(): 解析命令行参数，支持配置评估算法类型、模型路径、评估回合数等
- main(): 主函数，执行完整评估流程，包括模型加载、环境初始化、评估循环和结果统计

核心流程:
1. 命令行参数解析：配置评估参数如算法类型、模型路径等
2. 环境与可视化引擎初始化：创建仿真环境和实时可视化界面
3. 模型加载与配置：根据算法类型加载相应网络模型
4. 评估日志设置：创建日志目录结构并记录配置信息
5. 评估执行循环：使用确定性策略评估智能体在不同场景的表现
6. 结果统计与可视化：生成性能指标统计和速度曲线等可视化图表
"""

import pygame
import torch
import time
import os
import argparse
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from game_engine import GameEngine
from traffic_env import TrafficEnv
from sagi_ppo import SAGIPPOAgent
from ppo import PPOAgent

def set_seed(seed):
    """
    设置所有随机数生成器的种子，确保实验结果的可重复性
    
    参数:
        seed (int): 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def parse_args():
    """
    解析命令行参数，配置评估过程
    
    返回:
        argparse.Namespace: 包含所有解析后参数的命名空间对象
        
    参数说明:
        --algo: 选择评估的算法类型 (ppo 或 sagi_ppo)
        --model-dir: 预训练模型文件所在的目录路径
        --num-episodes: 要运行的评估回合数
        --seed: 随机种子，用于确保实验可重复性
    """
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO/SAGI-PPO agent.")
    parser.add_argument("--algo", type=str, default="ppo", choices=["sagi_ppo", "ppo"],
                        help="The algorithm of the trained agent to evaluate.")
    parser.add_argument("--model-dir", type=str,  default="models/ppo_scenario",
                        help="Path to the directory containing the saved model files (e.g., 'models/sagi_ppo_YYYYMMDD-HHMMSS').")
    parser.add_argument("--num-episodes", type=int, default=1, 
                        help="Number of episodes to run for evaluation.")
    parser.add_argument("--seed", type=int, default=8491,
                        help="Random seed for reproducibility.")#8491是激进的 233是保守的
    return parser.parse_args()

def main():
    """
    主函数，执行完整的智能体评估流程
    
    功能:
    1. 加载命令行参数并设置随机种子
    2. 初始化交通环境和可视化引擎
    3. 根据算法类型加载相应的预训练模型
    4. 创建评估日志目录和配置记录
    5. 执行评估回合，使用确定性策略在不同场景中测试智能体
    6. 记录详细的评估数据，包括状态、动作、奖励等轨迹信息
    7. 生成速度曲线等可视化图表
    8. 汇总并输出评估结果，包括成功率、碰撞率等关键指标
    """
    args = parse_args()
    
    # --- 设置随机种子 ---
    set_seed(args.seed)
    print(f"--- Setting random seed to {args.seed} ---")
    
    # --- 2. 创建环境和游戏引擎 ---
    env = TrafficEnv()
    # 为环境设置种子
    env.reset(seed=args.seed)
    
    game_engine = GameEngine(width=800, height=800) # 可以调大窗口以便观察

    # --- 3. 根据算法选择，创建Agent并加载模型 ---
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent = None
    if args.algo == "sagi_ppo":
        print("--- Loading SAGI-PPO Agent ---")
        agent = SAGIPPOAgent(state_dim=state_dim, action_dim=action_dim)
        actor_path = os.path.join(args.model_dir, "sagi_ppo_actor.pth")
        critic_r_path = os.path.join(args.model_dir, "sagi_ppo_critic_r.pth")
        critic_c_path = os.path.join(args.model_dir, "sagi_ppo_critic_c.pth")

        agent.actor.load_state_dict(torch.load(actor_path))
        agent.critic_r.load_state_dict(torch.load(critic_r_path))
        agent.critic_c.load_state_dict(torch.load(critic_c_path))
        
    elif args.algo == "ppo":
        print("--- Loading PPO Agent ---")
        agent = PPOAgent(state_dim=state_dim, action_dim=action_dim)
        actor_path = os.path.join(args.model_dir, "ppo_actor.pth")
        critic_path = os.path.join(args.model_dir, "ppo_critic.pth")

        agent.actor.load_state_dict(torch.load(actor_path))
        agent.critic.load_state_dict(torch.load(critic_path))

    # 将网络设置为评估模式
    agent.actor.eval()
    if isinstance(agent, SAGIPPOAgent):
        agent.critic_r.eval()
        agent.critic_c.eval()
    else:
        agent.critic.eval()

    model_name = os.path.basename(args.model_dir) # 从模型路径获取名字
    log_save_dir = os.path.join("evaluation_logs", model_name)
    os.makedirs(log_save_dir, exist_ok=True)
    print(f"评估日志将保存在: {log_save_dir}")

    # --- 记录评估的配置信息 ---
    eval_config_path = os.path.join(log_save_dir, "eval_config.txt")
    with open(eval_config_path, 'w') as f:
        f.write(f"Evaluation date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Algorithm: {args.algo}\n")
        f.write(f"Model path: {args.model_dir}\n")
        f.write(f"Number of episodes: {args.num_episodes}\n")
        f.write(f"Seed: {args.seed}\n")
    print(f"Evaluation configuration saved to {eval_config_path}")

    # --- 4. 评估主循环 ---
    eval_stats = {
        "rewards": [], "costs": [], "lengths": [],
        "success": 0, "collision": 0, "timeout": 0
    }
    
    scenarios = ['head_on_conflict']

    print(f"--- 开始评估, 运行 {args.num_episodes} 个回合 ---")
    try:
        for i in range(args.num_episodes):
            current_scenario = scenarios[i % len(scenarios)]
            state, info = env.reset(options={'scenario': current_scenario, 'algo': args.algo})
            
            episode_reward, episode_cost, episode_len = 0, 0, 0
            
            print(f"\n--- [回合 {i+1}/{args.num_episodes}, 场景: {current_scenario}] ---")

            while game_engine.is_running():
                game_engine.handle_events(env)
                
                if not game_engine.input_handler.paused:
                    # --- 5. 使用确定性动作 ---
                    action = agent.get_deterministic_action(state)
                    
                    next_state, reward, terminated, truncated, info = env.step(action)
                    state = next_state
                    
                    episode_reward += reward
                    episode_cost += info.get('cost', 0)
                    episode_len += 1
                    done = terminated or truncated
                    if done:
                        # 记录回合结果
                        eval_stats["rewards"].append(episode_reward)
                        eval_stats["costs"].append(episode_cost)
                        eval_stats["lengths"].append(episode_len)
                        
                        if info.get('failure') == 'collision':
                            eval_stats["collision"] += 1
                            print("结果: 碰撞 (Collision)")
                        elif truncated:
                            eval_stats["timeout"] += 1
                            print("结果: 超时 (Timeout)")
                        else: # terminated and not collision
                            eval_stats["success"] += 1
                            print("结果: 成功 (Success)")
                        # --- 回合结束，保存该回合的详细日志 ---
                        if "episode_log" in info and info["episode_log"]:
                            log_df = pd.DataFrame(info["episode_log"])

                            # 2. 定义文件名
                            outcome = "success"
                            if info.get('failure') == 'collision': outcome = "collision"
                            elif truncated: outcome = "timeout"
                            log_filename = f"episode_{i+1}_{current_scenario}_{outcome}.csv"

                            # 3. 保存文件
                            log_filepath = os.path.join(log_save_dir, log_filename)
                            log_df.to_csv(log_filepath, index=False)
                            print(f"日志已保存到: {log_filepath}")
                        
                        # --- 绘制并显示该回合的速度曲线图 ---

                        # 图1: RL Agent 的速度曲线
                        if hasattr(env, 'rl_agent') and env.rl_agent:
                            speed_curve = env.rl_agent.get_speed_history()
                            if speed_curve:
                                plt.figure(figsize=(12, 6))
                                plt.plot(speed_curve, 'r-', linewidth=2.5, label="RL Agent")
                                plt.title(f"RL Agent Speed Curve)")
                                plt.xlabel("Time Step")
                                plt.ylabel("Speed (m/s)")
                                plt.grid(True)
                                plt.xlim(0, len(speed_curve))
                                plt.ylim(0, 15)
                                plt.legend()
                                plt.tight_layout()

                        # 图2: 背景车辆的速度曲线
                        all_npc_data = []
                        # 添加已完成的车辆
                        if env.traffic_manager.completed_vehicles_data:
                            all_npc_data.extend(env.traffic_manager.completed_vehicles_data)
                        # 添加仍在活动的车辆
                        for vehicle in env.traffic_manager.vehicles:
                            if not getattr(vehicle, 'is_rl_agent', False):
                                all_npc_data.append({
                                    'id': vehicle.vehicle_id,
                                    'history': vehicle.get_speed_history()
                                })
                        
                        if all_npc_data:
                            plt.figure(figsize=(12, 6))
                            max_len_npc = 0
                            has_npc_plot = False
                            for vehicle_data in all_npc_data:
                                vehicle_history = vehicle_data['history']
                                if vehicle_history:
                                    plt.plot(vehicle_history, label=f"NPC #{vehicle_data['id']}")
                                    max_len_npc = max(max_len_npc, len(vehicle_history))
                                    has_npc_plot = True
                            
                            if has_npc_plot:
                                plt.title(f"NPC Vehicles Speed Curves")
                                plt.xlabel("Time Step")
                                plt.ylabel("Speed (m/s)")
                                plt.grid(True)
                                if max_len_npc > 0:
                                    plt.xlim(0, max_len_npc)
                                plt.ylim(0, 15)
                                plt.legend(loc='best', fontsize='small')
                                plt.tight_layout()

                        # 显示所有创建的图表
                        plt.show()

                        break # 结束当前回合的循环
                
                game_engine.render(env)
                game_engine.tick()

    except KeyboardInterrupt:
        print("\n评估被用户中断")
    finally:
        game_engine.quit()

    # --- 6. 打印最终的量化评估结果 ---
    print("\n\n--- 评估结果汇总 ---")
    num_episodes = len(eval_stats["rewards"])
    if num_episodes > 0:
        print(f"总回合数: {num_episodes}")
        print(f"平均奖励: {np.mean(eval_stats['rewards']):.2f} ± {np.std(eval_stats['rewards']):.2f}")
        print(f"平均成本: {np.mean(eval_stats['costs']):.2f} ± {np.std(eval_stats['costs']):.2f}")
        print(f"平均长度: {np.mean(eval_stats['lengths']):.2f}")
        print("-" * 20)
        print(f"成功率: {eval_stats['success'] / num_episodes:.2%}")
        print(f"碰撞率: {eval_stats['collision'] / num_episodes:.2%}")
        print(f"超时率: {eval_stats['timeout'] / num_episodes:.2%}")
    else:
        print("没有完成任何回合的评估。")


if __name__ == "__main__":
    main()
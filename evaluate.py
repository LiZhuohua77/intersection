"""
@file: evaluate.py
@description:
该文件用于评估和可视化已训练好的强化学习智能体在交通场景中的表现。
支持加载PPO和SAGI-PPO算法的模型，在可视化仿真环境中运行并评估智能体性能，
生成量化指标并记录详细日志数据。
[MODIFIED] 增加了对不同HV（人类驾驶车辆）驾驶员行为的系统性评估功能，
并修复了matplotlib绘图导致的内存泄漏问题。

主要函数:
- set_seed(seed): 设置所有随机数生成器的种子，确保实验可重复性
- parse_args(): 解析命令行参数，支持配置评估算法类型、模型路径、评估回合数等
- main(): 主函数，执行完整评估流程，包括模型加载、环境初始化、评估循环和结果统计

核心流程:
1. 命令行参数解析：配置评估参数如算法类型、模型路径等
2. 环境与可视化引擎初始化：创建仿真环境和实时可视化界面
3. 模型加载与配置：根据算法类型加载相应网络模型
4. 评估日志设置：创建日志目录结构并记录配置信息
5. [MODIFIED] 评估执行循环：为每种预设的HV驾驶员行为（个性和意图组合）运行指定数量的回合，
   使用确定性策略评估智能体在不同场景的表现。
6. [MODIFIED] 结果统计与可视化：为每个回合生成并保存速度曲线图，并在评估结束后
   按驾驶员行为分类汇总和展示性能指标。
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
import itertools
from config import *
from stable_baselines3 import PPO
from game_engine import GameEngine
from traffic_env import TrafficEnv
from sagi_ppo import SAGIPPO



def set_seed(seed):
    """
    设置所有随机数生成器的种子，确保实验结果的可重复性
    
    参数:
        seed (int): 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
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
        --num-episodes-per-driver: [MODIFIED] 为每种类型的司机运行的评估回合数
        --seed: 随机种子，用于确保实验可重复性
    """
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO/SAGI-PPO agent against different driver types.")
    parser.add_argument("--algo", type=str, default="sagi_ppo_gru",
                        choices=["sagi_ppo_mlp", "sagi_ppo_gru", "ppo_gru", "ppo_mlp"], 
                        help="The algorithm of the trained agent to evaluate.")
    parser.add_argument("--model_path", type=str, default="D:/Code/intersection/models/sagi_ppo_gru_final_model.zip",
                        
                        help="Path to the saved model .zip file (e.g., 'models/final_model.zip').")
    parser.add_argument("--num-episodes-per-driver", type=int, default=3,
                        help="Number of episodes to run for EACH driver type for evaluation.")
    parser.add_argument("--seed", type=int, default=8491,
                        help="Random seed for reproducibility. (8491 for aggressive, 233 for conservative)")
    return parser.parse_args()


def main():
    """
    主函数，执行完整的智能体评估流程
    
    功能:
    1. 加载命令行参数并设置随机种子
    2. 初始化交通环境和可视化引擎
    3. 根据算法类型加载相应的预训练模型
    4. 创建评估日志目录和配置记录
    5. [MODIFIED] 针对不同驾驶员行为组合，执行评估回合
    6. 记录详细的评估数据，包括状态、动作、奖励等轨迹信息
    7. [MODIFIED] 为每个回合生成并保存速度曲线等可视化图表
    8. [MODIFIED] 按驾驶员类型汇总并输出评估结果，包括成功率、碰撞率等关键指标
    """
    args = parse_args()
    
    # --- 设置随机种子 ---
    set_seed(args.seed)
    print(f"--- Setting random seed to {args.seed} ---")

    GAMMA = 0.99 # 假设与训练时相同
    
    # --- 2. 创建环境和游戏引擎 ---
    # 初始场景可以设置为一个通用值，后续在reset中动态指定
    env = TrafficEnv(scenario='random_traffic')
    env.reset(seed=args.seed)
    
    game_engine = GameEngine(width=800, height=800)

    if hasattr(env, 'traffic_manager') and hasattr(env.traffic_manager, 'set_hv_speed_scaling'):
            env.traffic_manager.set_hv_speed_scaling(1.0) 
    else:
        print("警告：无法在TrafficManager中找到set_hv_speed_scaling方法。HV速度可能不正确。")

    model = None
    print(f"--- Loading {args.algo.upper()} Agent from {args.model_path} ---")
    try:
        if args.algo.startswith("ppo"):
            model = PPO.load(args.model_path, env=env)
        elif args.algo.startswith("sagi_ppo"):
            model = SAGIPPO.load(args.model_path, env=env)
        else:
            raise ValueError(f"Unknown algorithm for loading: {args.algo}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("请确保您提供的路径是一个由Stable Baselines3保存的.zip模型文件，")
        print("并且您的agent.py, config.py等文件与训练时保持一致。")
        return

    model_name = os.path.splitext(os.path.basename(args.model_path))[0]
    log_save_dir = os.path.join("evaluation_logs", f"{model_name}_{time.strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(log_save_dir, exist_ok=True)
    print(f"评估日志将保存在: {log_save_dir}")

    # --- 记录评估的配置信息 ---
    eval_config_path = os.path.join(log_save_dir, "eval_config.txt")
    with open(eval_config_path, 'w') as f:
        f.write(f"Evaluation date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Algorithm: {args.algo}\n")
        f.write(f"Model Path: {args.model_path}\n")
        f.write(f"Episodes per driver type: {args.num_episodes_per_driver}\n")
        f.write(f"Seed: {args.seed}\n")
    print(f"Evaluation configuration saved to {eval_config_path}")

    # --- [MODIFIED] 定义要评估的驾驶员行为组合 ---
    personalities = list(IDM_PARAMS.keys())  # ['aggressive', 'normal', 'conservative']
    intents = ['GO', 'YIELD']
    driver_configs = list(itertools.product(personalities, intents))
    
    overall_stats = {}

    print(f"--- 开始评估, 将为 {len(driver_configs)} 种驾驶员类型各运行 {args.num_episodes_per_driver} 个回合 ---")
    
    try:
        total_episode_counter = 0
        for personality, intent in driver_configs:
            driver_key = f"{personality}_{intent}"
            print(f"\n\n{'='*50}")
            print(f"--- 开始评估驾驶员类型: Personality='{personality}', Intent='{intent}' ---")
            print(f"{'='*50}")
            
            # 为每种驾驶员初始化统计数据
            eval_stats = {
                "rewards": [], "costs": [], "lengths": [], "discounted_costs": [],
                "success": 0, "collision": 0, "timeout": 0, "off_track": 0
            }
            
            for i in range(args.num_episodes_per_driver):
                total_episode_counter += 1
                # [MODIFIED] 在 reset 时传入驾驶员配置
                reset_options = {
                    'scenario': 'random_traffic', 
                    'algo': args.algo,
                    'hv_personality': personality,
                    'hv_intent': intent
                }
                state, info = env.reset(seed=args.seed + total_episode_counter, options=reset_options)
                
                episode_reward, episode_cost, episode_len = 0, 0, 0

                episode_discounted_cost = 0
                
                print(f"\n--- [全局回合 {total_episode_counter}, {driver_key} 第 {i+1}/{args.num_episodes_per_driver} 回合] ---")

                while game_engine.is_running():
                    game_engine.handle_events(env)
                    
                    if not game_engine.input_handler.paused:
                        action, _states = model.predict(state, deterministic=True)
                        next_state, reward, terminated, truncated, info = env.step(action)
                        state = next_state
                        
                        episode_reward += reward
                        episode_cost += info.get('cost', 0)
                        episode_len += 1
                        episode_discounted_cost += (GAMMA ** (episode_len - 1)) * info.get('cost', 0)
                        done = terminated or truncated

                        if done:
                            eval_stats["rewards"].append(episode_reward)
                            eval_stats["costs"].append(episode_cost)
                            eval_stats["lengths"].append(episode_len)
                            eval_stats["discounted_costs"].append(episode_discounted_cost)
                            
                            outcome = "success"
                            if info.get('failure') == 'collision':
                                eval_stats["collision"] += 1
                                outcome = "collision"
                                print(f"结果: 碰撞 (Collision) | 回合奖励: {episode_reward:.2f}")
                            elif info.get('failure') == 'off_track':
                                eval_stats["off_track"] += 1
                                outcome = "off_track"
                                print(f"结果: 偏离轨迹 (Off-track) | 回合奖励: {episode_reward:.2f}")
                            elif truncated:
                                eval_stats["timeout"] += 1
                                outcome = "timeout"
                                print(f"结果: 超时 (Timeout) | 回合奖励: {episode_reward:.2f}")
                            else: # terminated and not collision
                                eval_stats["success"] += 1
                                print(f"结果: 成功 (Success) | 回合奖励: {episode_reward:.2f}")

                            # --- 保存详细日志 ---
                            if "episode_log" in info and info["episode_log"]:
                                log_df = pd.DataFrame(info["episode_log"])
                                log_filename = f"ep_{total_episode_counter}_{driver_key}_{outcome}.csv"
                                log_filepath = os.path.join(log_save_dir, log_filename)
                                log_df.to_csv(log_filepath, index=False)
                                print(f"日志已保存到: {log_filepath}")
                            
                            # --- [MODIFIED] 绘制、保存并关闭绘图，防止内存泄漏 ---
                            plot_filename_base = f"ep_{total_episode_counter}_{driver_key}_{outcome}"
                            plot_base_path = os.path.join(log_save_dir, plot_filename_base)
                            
                            # 图1: RL Agent 的速度曲线
                            if hasattr(env, 'rl_agent') and env.rl_agent:
                                speed_curve = env.rl_agent.get_speed_history()
                                if speed_curve:
                                    plt.figure(figsize=(12, 6))
                                    plt.plot(speed_curve, 'r-', linewidth=2.5, label="RL Agent")
                                    plt.title(f"RL Agent Speed Curve (Episode {total_episode_counter}, {driver_key})")
                                    plt.xlabel("Time Step")
                                    plt.ylabel("Speed (m/s)")
                                    plt.grid(True)
                                    plt.xlim(0, max(1, len(speed_curve)))
                                    plt.ylim(0, 15)
                                    plt.legend()
                                    plt.tight_layout()
                                    rl_plot_path = os.path.join(log_save_dir, f"{plot_filename_base}_rl_speed.png")
                                    plt.savefig(rl_plot_path)
                                    plt.close() # <-- 关键：关闭图形以释放内存
                                    print(f"RL Agent 速度图已保存到: {rl_plot_path}")
                                    
                                env.rl_agent.plot_reward_history(save_path_base=plot_base_path)

                            # 图2: 背景车辆的速度曲线
                            all_npc_data = []
                            if env.traffic_manager.completed_vehicles_data:
                                all_npc_data.extend(env.traffic_manager.completed_vehicles_data)
                            for vehicle in env.traffic_manager.vehicles:
                                if not getattr(vehicle, 'is_rl_agent', False):
                                    all_npc_data.append({
                                        'id': vehicle.vehicle_id,
                                        'history': vehicle.get_speed_history()
                                    })
                            
                            if any(data['history'] for data in all_npc_data):
                                plt.figure(figsize=(12, 6))
                                max_len_npc = 0
                                for vehicle_data in all_npc_data:
                                    if vehicle_data['history']:
                                        plt.plot(vehicle_data['history'], label=f"NPC #{vehicle_data['id']} ({driver_key})")
                                        max_len_npc = max(max_len_npc, len(vehicle_data['history']))
                                
                                plt.title(f"NPC Vehicles Speed Curves (Episode {total_episode_counter}, {driver_key})")
                                plt.xlabel("Time Step")
                                plt.ylabel("Speed (m/s)")
                                plt.grid(True)
                                plt.xlim(0, max(1, max_len_npc))
                                plt.ylim(0, 15)
                                plt.legend(loc='best', fontsize='small')
                                plt.tight_layout()
                                npc_plot_path = os.path.join(log_save_dir, f"{plot_filename_base}_npc_speed.png")
                                plt.savefig(npc_plot_path)
                                plt.close() # <-- 关键：关闭图形以释放内存
                                print(f"NPC 车辆速度图已保存到: {npc_plot_path}")

                            break # 结束当前回合的循环
                    
                    game_engine.render(env)
                    game_engine.tick()

            # 保存该驾驶员类型的结果
            overall_stats[driver_key] = eval_stats

    except KeyboardInterrupt:
        print("\n评估被用户中断")
    finally:
        game_engine.quit()

    # --- [MODIFIED] 打印最终的分类量化评估结果 ---
    print("\n\n" + "="*30)
    print("--- 评估结果汇总 ---")
    print("="*30)

    for driver_key, stats in overall_stats.items():
        num_episodes = len(stats["rewards"])
        if num_episodes > 0:
            print(f"\n--- 结果 for 驾驶员类型: {driver_key} ---")
            print(f"总回合数: {num_episodes}")
            print(f"平均奖励: {np.mean(stats['rewards']):.2f} ± {np.std(stats['rewards']):.2f}")
            print(f"平均成本: {np.mean(stats['costs']):.2f} ± {np.std(stats['costs']):.2f}")
            print(f"平均累积折扣成本: {np.mean(stats['discounted_costs']):.2f} ± {np.std(stats['discounted_costs']):.2f}")
            print(f"平均长度: {np.mean(stats['lengths']):.2f}")
            print("-" * 20)
            print(f"成功率: {stats['success'] / num_episodes:.2%}")
            print(f"碰撞率: {stats['collision'] / num_episodes:.2%}")
            print(f"偏离轨迹率: {stats['off_track'] / num_episodes:.2%}")
            print(f"超时率: {stats['timeout'] / num_episodes:.2%}")
        else:
            print(f"\n--- 驾驶员类型 '{driver_key}' 没有完成任何回合的评估。 ---")
    
    if not overall_stats:
        print("没有完成任何回合的评估。")

if __name__ == "__main__":
    main()
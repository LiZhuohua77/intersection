"""
@file: evaluate.py
@description:
该文件用于评估和可视化已训练好的强化学习智能体在交通场景中的表现。
支持加载 PPO / SAGI-PPO / Lagrangian PPO 等模型，在可视化仿真环境中运行并评估智能体性能，
生成量化指标并记录详细日志数据。

主要函数:
- set_seed(seed): 设置所有随机数生成器的种子，确保实验可重复性
- parse_args(): 解析命令行参数，支持配置评估算法类型、模型路径、评估回合数、场景等
- main(): 主函数，执行完整评估流程，包括模型加载、环境初始化、评估循环和结果统计
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

from config import *
from stable_baselines3 import PPO
from game_engine import GameEngine
from traffic_env import TrafficEnv
from sagi_ppo import SAGIPPO
from ppo_lagrangian import PPOLagrangian


def set_seed(seed):
    """
    设置所有随机数生成器的种子，确保实验结果的可重复性
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
    """
    parser = argparse.ArgumentParser(
        description="Evaluate a trained PPO/SAGI-PPO/PPOLagrangian agent.")
    parser.add_argument("--algo", type=str, default="ppo_lagrangian_gru", choices=["sagi_ppo_mlp", "sagi_ppo_gru", "ppo_gru", "ppo_mlp", "ppo_lagrangian_gru", "ppo_lagrangian_mlp"],
        help="The algorithm of the trained agent to evaluate.")
    parser.add_argument("--model_path",type=str,default="D:/Code/intersection/models/expt1/ppo_lagrangian_gru_final_model.zip",
        help="Path to the saved model .zip file (e.g., 'models/final_model.zip')."    )
    parser.add_argument("--num-episodes",type=int,default=20,
                        help="Total number of episodes to run for evaluation.")
    parser.add_argument("--seed",type=int,default=8491,
                        help="Random seed for reproducibility."    )
    parser.add_argument("--scenario", type=str,default="agent_only_simple",choices=["agent_only_simple", "random_traffic", "crossing_conflict"],
                        help="Evaluation scenario name in TrafficEnv/TrafficManager.")

    return parser.parse_args()


def handle_episode_end(
    env,
    info,
    eval_stats,
    episode_reward,
    episode_cost,
    episode_len,
    episode_discounted_cost,
    episode_idx,
    log_save_dir,
    truncated: bool,
):
    """
    统一处理一个回合结束时的：
    - outcome 判定（success / collision / off_track / timeout）
    - 轨迹日志保存（CSV）
    - 指标计算（avg_jerk / avg_sCTE）
    - eval_stats 更新
    - 绘图（RL Agent 速度、NPC 速度），并关闭图防止内存泄漏
    """

    # --- 1. 判定 outcome 并更新分类统计 ---
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
    else:
        eval_stats["success"] += 1
        print(f"结果: 成功 (Success) | 回合奖励: {episode_reward:.2f}")

    # --- 2. 保存轨迹日志 CSV ---
    log_df = None
    if "episode_log" in info and info["episode_log"]:
        try:
            log_df = pd.DataFrame(info["episode_log"])
            log_filename = f"ep_{episode_idx}_{outcome}.csv"
            log_filepath = os.path.join(log_save_dir, log_filename)
            log_df.to_csv(log_filepath, index=False)
            print(f"轨迹日志已保存到: {log_filepath}")
        except Exception as e:
            print(f"保存轨迹日志失败: {e}")

    # --- 3. 计算 jerk / sCTE ---
    avg_jerk = 0.0
    avg_scte = 0.0
    if log_df is not None and not log_df.empty:
        # jerk
        if "action_accel" in log_df.columns:
            accels = log_df['action_accel'] * MAX_ACCELERATION
            if len(accels) > 1:
                jerks = np.abs(np.diff(accels) / env.dt)
                if len(jerks) > 0:
                    avg_jerk = float(np.mean(jerks))
        # sCTE
    if "signed_cross_track_error" in log_df.columns:
        scte_series = log_df["signed_cross_track_error"]
        avg_scte = float(np.mean(np.abs(scte_series)))   # 平均绝对偏差
        # 也可以同时算一个 signed_mean 方便调试：
        signed_mean_scte = float(scte_series.mean())
        # 如果你想记录下来，可以额外存进 eval_stats 或日志


    # --- 4. 累积统计 ---
    eval_stats["rewards"].append(episode_reward)
    eval_stats["costs"].append(episode_cost)
    eval_stats["lengths"].append(episode_len)
    eval_stats["discounted_costs"].append(episode_discounted_cost)
    eval_stats["avg_jerk"].append(avg_jerk)
    eval_stats["avg_scte"].append(avg_scte)

    # --- 5. 绘图保存 ---
    plot_filename_base = f"ep_{episode_idx}_{outcome}"
    plot_base_path = os.path.join(log_save_dir, plot_filename_base)

    # 5.1 RL Agent 速度曲线 + 奖励历史
    if hasattr(env, 'rl_agent') and getattr(env, 'rl_agent', None) is not None:
        try:
            speed_curve = env.rl_agent.get_speed_history()
            if speed_curve:
                plt.figure(figsize=(12, 6))
                plt.plot(speed_curve, 'r-', linewidth=2.5, label="RL Agent")
                plt.title(f"RL Agent Speed Curve (Episode {episode_idx})")
                plt.xlabel("Time Step")
                plt.ylabel("Speed (m/s)")
                plt.grid(True)
                plt.xlim(0, max(1, len(speed_curve)))
                plt.ylim(0, 15)
                plt.legend()
                plt.tight_layout()
                rl_plot_path = os.path.join(
                    log_save_dir, f"{plot_filename_base}_rl_speed.png"
                )
                #plt.savefig(rl_plot_path)
                plt.close()
                print(f"RL Agent 速度图已保存到: {rl_plot_path}")

            # RL Agent 奖励历史图（该方法内部应自行保存 / 关闭）
            env.rl_agent.plot_reward_history(save_path_base=plot_base_path)

        except Exception as e:
            print(f"保存 RL Agent 速度/奖励图失败: {e}")

    # 5.2 NPC 车辆速度曲线
    try:
        all_npc_data = []
        if hasattr(env, "traffic_manager"):
            tm = env.traffic_manager
            if getattr(tm, "completed_vehicles_data", None):
                all_npc_data.extend(tm.completed_vehicles_data)
            for vehicle in getattr(tm, "vehicles", []):
                if not getattr(vehicle, 'is_rl_agent', False):
                    all_npc_data.append(
                        {
                            'id': vehicle.vehicle_id,
                            'history': vehicle.get_speed_history()
                        }
                    )

        if any(data['history'] for data in all_npc_data):
            plt.figure(figsize=(12, 6))
            max_len_npc = 0
            for vehicle_data in all_npc_data:
                if vehicle_data['history']:
                    plt.plot(
                        vehicle_data['history'],
                        label=f"NPC #{vehicle_data['id']}"
                    )
                    max_len_npc = max(max_len_npc, len(vehicle_data['history']))

            plt.title(f"NPC Vehicles Speed Curves (Episode {episode_idx})")
            plt.xlabel("Time Step")
            plt.ylabel("Speed (m/s)")
            plt.grid(True)
            plt.xlim(0, max(1, max_len_npc))
            plt.ylim(0, 15)
            plt.legend(loc='best', fontsize='small')
            plt.tight_layout()
            npc_plot_path = os.path.join(
                log_save_dir, f"{plot_filename_base}_npc_speed.png"
            )
            #plt.savefig(npc_plot_path)
            plt.close()
            print(f"NPC 车辆速度图已保存到: {npc_plot_path}")
    except Exception as e:
        print(f"保存 NPC 车辆速度图失败: {e}")

    return outcome


def main():
    """
    主函数，执行完整的智能体评估流程
    """
    args = parse_args()
    
    # --- 设置随机种子 ---
    set_seed(args.seed)
    print(f"--- Setting random seed to {args.seed} ---")

    GAMMA = 0.99  # 假设与训练时相同
    
    # --- 2. 创建环境和游戏引擎 ---
    env = TrafficEnv(scenario=args.scenario)
    env.reset(seed=args.seed)
    
    game_engine = GameEngine(width=800, height=800)

    if hasattr(env, 'traffic_manager') and hasattr(env.traffic_manager, 'set_hv_speed_scaling'):
        env.traffic_manager.set_hv_speed_scaling(1.0)
    else:
        print("警告：无法在TrafficManager中找到set_hv_speed_scaling方法。HV速度可能不正确。")

    # --- 3. 加载模型 ---
    model = None
    print(f"--- Loading {args.algo.upper()} Agent from {args.model_path} ---")
    try:
        if args.algo.startswith("ppo_lagrangian"):
            # 自定义的 Lagrangian PPO
            model = PPOLagrangian.load(args.model_path, env=env)
        elif args.algo.startswith("ppo"):
            # 普通 SB3-PPO
            model = PPO.load(args.model_path, env=env)
        elif args.algo.startswith("sagi_ppo"):
            # SAGI-PPO 自定义类
            model = SAGIPPO.load(args.model_path, env=env)
        else:
            raise ValueError(f"Unknown algorithm for loading: {args.algo}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("请确保您提供的路径是一个由Stable Baselines3保存的.zip模型文件，")
        print("并且您的agent.py, config.py等文件与训练时保持一致。")
        return

    # --- 4. 日志目录 ---
    model_name = os.path.splitext(os.path.basename(args.model_path))[0]
    log_save_dir = os.path.join(
        "evaluation_logs",
        f"{model_name}_{time.strftime('%Y%m%d-%H%M%S')}"
    )
    os.makedirs(log_save_dir, exist_ok=True)
    print(f"评估日志将保存在: {log_save_dir}")

    # 记录评估配置
    eval_config_path = os.path.join(log_save_dir, "eval_config.txt")
    with open(eval_config_path, 'w') as f:
        f.write(f"Evaluation date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Algorithm: {args.algo}\n")
        f.write(f"Model Path: {args.model_path}\n")
        f.write(f"Episodes: {args.num_episodes}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Scenario: {args.scenario}\n")
    print(f"Evaluation configuration saved to {eval_config_path}")

    # --- 5. 评估统计容器（不再区分 driver type，全局一个即可） ---
    eval_stats = {
        "rewards": [],
        "costs": [],
        "lengths": [],
        "discounted_costs": [],
        "avg_jerk": [],
        "avg_scte": [],
        "success": 0,
        "collision": 0,
        "timeout": 0,
        "off_track": 0
    }

    print(f"--- 开始评估: 共 {args.num_episodes} 个回合, 场景 = {args.scenario} ---")
    
    try:
        for ep in range(1, args.num_episodes + 1):
            reset_options = {
                'scenario': args.scenario,
                'algo': args.algo,
                # 不再指定 hv_personality / hv_intent，HV 完全随机
            }
            state, info = env.reset(
                seed=args.seed + ep,
                options=reset_options
            )
            
            episode_reward = 0.0
            episode_cost = 0.0
            episode_len = 0
            episode_discounted_cost = 0.0
            
            print(f"\n--- [Episode {ep}/{args.num_episodes}] ---")

            while game_engine.is_running():
                game_engine.handle_events(env)
                
                if not game_engine.input_handler.paused:
                    action, _states = model.predict(state, deterministic=True)
                    next_state, reward, terminated, truncated, info = env.step(action)
                    state = next_state
                    
                    step_cost = info.get('cost', 0.0)
                    episode_reward += reward
                    episode_cost += step_cost
                    episode_len += 1
                    episode_discounted_cost += (GAMMA ** (episode_len - 1)) * step_cost

                    done = terminated or truncated
                    if done:
                        handle_episode_end(
                            env=env,
                            info=info,
                            eval_stats=eval_stats,
                            episode_reward=episode_reward,
                            episode_cost=episode_cost,
                            episode_len=episode_len,
                            episode_discounted_cost=episode_discounted_cost,
                            episode_idx=ep,
                            log_save_dir=log_save_dir,
                            truncated=truncated,
                        )
                        break
                
                game_engine.render(env)
                game_engine.tick()

            # 如果窗口被关掉了，直接结束评估循环
            if not game_engine.is_running():
                print("检测到窗口关闭，中止后续评估。")
                break

    except KeyboardInterrupt:
        print("\n评估被用户中断")
    finally:
        game_engine.quit()

    # --- 6. 打印最终量化评估结果 ---
    print("\n\n" + "="*30)
    print("--- 评估结果汇总 ---")
    print("="*30)

    num_episodes_finished = len(eval_stats["rewards"])
    if num_episodes_finished > 0:
        avg_travel_time = np.mean(eval_stats['lengths']) * env.dt
        print(f"完成回合数: {num_episodes_finished}")
        print(f"平均奖励: {np.mean(eval_stats['rewards']):.2f} ± {np.std(eval_stats['rewards']):.2f}")
        print(f"平均回合成本: {np.mean(eval_stats['costs']):.2f} ± {np.std(eval_stats['costs']):.2f}")
        print(f"平均通行时间 (秒): {avg_travel_time:.2f}")
        print(f"平均加加速度 (m/s^3): {np.mean(eval_stats.get('avg_jerk', [0])):.2f}")
        print(f"平均 |sCTE| (米): {np.mean(eval_stats.get('avg_scte', [0])):.2f}")
        print("-" * 20)
        print(f"成功率: {eval_stats['success'] / num_episodes_finished:.2%}")
        print(f"碰撞率: {eval_stats['collision'] / num_episodes_finished:.2%}")
        print(f"偏离轨迹率: {eval_stats['off_track'] / num_episodes_finished:.2%}")
        print(f"超时率: {eval_stats['timeout'] / num_episodes_finished:.2%}")
    else:
        print("没有完成任何回合的评估。")


if __name__ == "__main__":
    main()

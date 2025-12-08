"""
@file: evaluate_headless.py
@description:
无 GUI 版本的评估脚本，仅使用 TrafficEnv 与已训练好的 RL 智能体交互，
统计整体性能指标，并保存每个 episode 的轨迹日志和速度曲线。

支持算法:
- PPO (ppo_gru, ppo_mlp)
- SAGI-PPO (sagi_ppo_mlp, sagi_ppo_gru)
- Lagrangian PPO (ppo_lagrangian_gru, ppo_lagrangian_mlp)
"""

import os
import time
import argparse
import random

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from config import *
from stable_baselines3 import PPO
from traffic_env import TrafficEnv
from sagi_ppo import SAGIPPO
from ppo_lagrangian import PPOLagrangian


def set_seed(seed: int):
    """设置所有随机数生成器的种子，确保实验可重复性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Headless evaluation of a trained PPO/SAGI-PPO/PPOLagrangian agent."
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="ppo_lagrangian_gru",
        choices=[
            "sagi_ppo_mlp",
            "sagi_ppo_gru",
            "ppo_gru",
            "ppo_mlp",
            "ppo_lagrangian_gru",
            "ppo_lagrangian_mlp",
        ],
        help="The algorithm of the trained agent to evaluate.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="D:/Code/intersection/models/expt1/ppo_lagrangian_gru_final_model.zip",
        help="Path to the saved model .zip file (e.g., 'models/final_model.zip').",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=1000,
        help="Total number of episodes to run for evaluation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=8491,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="agent_only_simple",
        choices=["agent_only_simple", "random_traffic", "crossing_conflict"],
        help="Evaluation scenario name in TrafficEnv/TrafficManager.",
    )
    return parser.parse_args()


def handle_episode_end(
    env,
    info,
    eval_stats: dict,
    episode_reward: float,
    episode_cost: float,
    episode_len: int,
    episode_discounted_cost: float,
    episode_idx: int,
    log_save_dir: str,
    truncated: bool,
):
    """
    统一处理一个 episode 结束时的：
    - outcome 判定（success / collision / off_track / timeout）
    - 轨迹日志保存（CSV）
    - jerk / sCTE / 能耗 计算
    - 统计量更新
    - RL & NPC 速度曲线绘制
    """
    # --- 1. outcome 判定 ---
    outcome = "success"
    if info.get("failure") == "collision":
        eval_stats["collision"] += 1
        outcome = "collision"
        print(f"结果: 碰撞 (Collision) | 回合奖励: {episode_reward:.2f}")
    elif info.get("failure") == "off_track":
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

    # --- 2. 保存 episode 轨迹日志 ---
    log_df = None
    if "episode_log" in info and info["episode_log"]:
        try:
            log_df = pd.DataFrame(info["episode_log"])
            log_filename = f"ep_{episode_idx}_{outcome}.csv"
            log_filepath = os.path.join(log_save_dir, log_filename)
            # 为了加速，目前不真正写 CSV，如需保存可以取消注释
            # log_df.to_csv(log_filepath, index=False)
            print(f"轨迹日志已（预备）保存到: {log_filepath}")
        except Exception as e:
            print(f"保存轨迹日志失败: {e}")

    # --- 3. 计算 jerk、sCTE 和能耗 ---
    avg_jerk = 0.0
    avg_scte = 0.0
    episode_energy_net_kwh = 0.0
    episode_energy_consume_kwh = 0.0
    episode_energy_regen_kwh = 0.0

    if log_df is not None and not log_df.empty:
        # jerk
        if "action_accel" in log_df.columns:
            accels = log_df["action_accel"] * MAX_ACCELERATION
            if len(accels) > 1:
                jerks = np.abs(np.diff(accels) / env.dt)
                if len(jerks) > 0:
                    avg_jerk = float(np.mean(jerks))

        # sCTE（这里是有符号平均值；如果想要 |sCTE| 可以改成 np.mean(np.abs(...))）
        if "signed_cross_track_error" in log_df.columns:
            avg_scte = float(log_df["signed_cross_track_error"].mean())

        # 能耗：raw_power 为瞬时电功率(kW) -> 积分得到电能(kWh)
        if "raw_power" in log_df.columns:
            power_kw = log_df["raw_power"].to_numpy(dtype=np.float32)
            dt_hour = env.dt / 3600.0  # 秒 -> 小时

            power_pos_kw = np.clip(power_kw, 0, None)     # 只取消耗
            power_neg_kw = np.clip(power_kw, None, 0)     # 只取回收(负值)

            episode_energy_consume_kwh = float(np.sum(power_pos_kw) * dt_hour)
            # 回收能量用正值表示：-负功率的积分
            episode_energy_regen_kwh = float(-np.sum(power_neg_kw) * dt_hour)
            episode_energy_net_kwh = episode_energy_consume_kwh - episode_energy_regen_kwh

    print(
        f"[Episode {episode_idx}] avg_jerk = {avg_jerk:.4f}, "
        f"avg_sCTE = {avg_scte:.4f}, "
        f"net_energy = {episode_energy_net_kwh:.4f} kWh "
        f"(consume {episode_energy_consume_kwh:.4f}, regen {episode_energy_regen_kwh:.4f})"
    )

    # --- 4. 更新统计量 ---
    eval_stats["rewards"].append(episode_reward)
    eval_stats["costs"].append(episode_cost)
    eval_stats["lengths"].append(episode_len)
    eval_stats["discounted_costs"].append(episode_discounted_cost)
    eval_stats["avg_jerk"].append(avg_jerk)
    eval_stats["avg_scte"].append(avg_scte)
    eval_stats["episode_energy_net_kwh"].append(episode_energy_net_kwh)
    eval_stats["episode_energy_consume_kwh"].append(episode_energy_consume_kwh)
    eval_stats["episode_energy_regen_kwh"].append(episode_energy_regen_kwh)

    # --- 5. 绘图：RL Agent 速度 + NPC 速度 ---
    plot_filename_base = f"ep_{episode_idx}_{outcome}"
    plot_base_path = os.path.join(log_save_dir, plot_filename_base)

    # 5.1 RL Agent 速度 + 奖励历史
    if hasattr(env, "rl_agent") and getattr(env, "rl_agent", None) is not None:
        try:
            speed_curve = env.rl_agent.get_speed_history()
            if speed_curve:
                plt.figure(figsize=(12, 6))
                plt.plot(speed_curve, linewidth=2.0, label="RL Agent")
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
                # 为了加速，这里也先不真正保存，如需保存取消注释
                # plt.savefig(rl_plot_path)
                plt.close()
                print(f"RL Agent 速度图已（预备）保存到: {rl_plot_path}")

            # 奖励历史图（假设内部会保存图像）
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
                if not getattr(vehicle, "is_rl_agent", False):
                    all_npc_data.append(
                        {"id": vehicle.vehicle_id, "history": vehicle.get_speed_history()}
                    )

        if any(data["history"] for data in all_npc_data):
            plt.figure(figsize=(12, 6))
            max_len_npc = 0
            for vehicle_data in all_npc_data:
                if vehicle_data["history"]:
                    plt.plot(
                        vehicle_data["history"],
                        label=f"NPC #{vehicle_data['id']}",
                    )
                    max_len_npc = max(max_len_npc, len(vehicle_data["history"]))

            plt.title(f"NPC Vehicles Speed Curves (Episode {episode_idx})")
            plt.xlabel("Time Step")
            plt.ylabel("Speed (m/s)")
            plt.grid(True)
            plt.xlim(0, max(1, max_len_npc))
            plt.ylim(0, 15)
            plt.legend(loc="best", fontsize="small")
            plt.tight_layout()
            npc_plot_path = os.path.join(
                log_save_dir, f"{plot_filename_base}_npc_speed.png"
            )
            # plt.savefig(npc_plot_path)
            plt.close()
            print(f"NPC 车辆速度图已（预备）保存到: {npc_plot_path}")
    except Exception as e:
        print(f"保存 NPC 车辆速度图失败: {e}")

    return outcome


def main():
    args = parse_args()

    # 1. 设置随机种子
    set_seed(args.seed)
    print(f"--- Setting random seed to {args.seed} ---")

    GAMMA = 0.99  # 与训练时保持一致

    # 2. 创建环境
    env = TrafficEnv(scenario=args.scenario)
    env.reset(seed=args.seed)

    if hasattr(env, "traffic_manager") and hasattr(
        env.traffic_manager, "set_hv_speed_scaling"
    ):
        env.traffic_manager.set_hv_speed_scaling(1.0)
    else:
        print(
            "警告：无法在TrafficManager中找到set_hv_speed_scaling方法。HV速度可能不正确。"
        )

    # 3. 加载模型
    print(f"--- Loading {args.algo.upper()} Agent from {args.model_path} ---")
    try:
        if args.algo.startswith("ppo_lagrangian"):
            model = PPOLagrangian.load(args.model_path, env=env)
        elif args.algo.startswith("ppo"):
            model = PPO.load(args.model_path, env=env)
        elif args.algo.startswith("sagi_ppo"):
            model = SAGIPPO.load(args.model_path, env=env)
        else:
            raise ValueError(f"Unknown algorithm for loading: {args.algo}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("请确保模型是由 Stable Baselines3 保存的 .zip 文件，并且代码版本一致。")
        return

    # 4. 日志目录
    model_name = os.path.splitext(os.path.basename(args.model_path))[0]
    log_save_dir = os.path.join(
        "evaluation_logs", f"{model_name}_headless_{time.strftime('%Y%m%d-%H%M%S')}"
    )
    os.makedirs(log_save_dir, exist_ok=True)
    print(f"评估日志将保存在: {log_save_dir}")

    eval_config_path = os.path.join(log_save_dir, "eval_config.txt")
    with open(eval_config_path, "w") as f:
        f.write(f"Evaluation date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Algorithm: {args.algo}\n")
        f.write(f"Model Path: {args.model_path}\n")
        f.write(f"Episodes: {args.num_episodes}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Scenario: {args.scenario}\n")
    print(f"Evaluation configuration saved to {eval_config_path}")

    # 5. 全局统计
    eval_stats = {
        "rewards": [],
        "costs": [],
        "lengths": [],
        "discounted_costs": [],
        "avg_jerk": [],
        "avg_scte": [],
        "episode_energy_net_kwh": [],
        "episode_energy_consume_kwh": [],
        "episode_energy_regen_kwh": [],
        "success": 0,
        "collision": 0,
        "timeout": 0,
        "off_track": 0,
    }

    print(
        f"--- 开始无 GUI 评估: 共 {args.num_episodes} 个回合, 场景 = {args.scenario} ---"
    )

    # 6. 主评估循环
    for ep in range(1, args.num_episodes + 1):
        reset_options = {
            "scenario": args.scenario,
            "algo": args.algo,
            # 不再指定 HV 性格/意图，让 TrafficManager 完全随机
        }
        state, info = env.reset(seed=args.seed + ep, options=reset_options)

        episode_reward = 0.0
        episode_cost = 0.0
        episode_len = 0
        episode_discounted_cost = 0.0

        print(f"\n--- [Episode {ep}/{args.num_episodes}] ---")

        while True:
            # 纯环境交互，无 GUI
            action, _states = model.predict(state, deterministic=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            state = next_state

            step_cost = info.get("cost", 0.0)
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

    # 7. 打印并保存评估结果
    summary_lines = []
    header = "--- 评估结果汇总（整体统计） ---"
    summary_lines.append("=" * 30)
    summary_lines.append(header)
    summary_lines.append("=" * 30)

    num_episodes_finished = len(eval_stats["rewards"])
    if num_episodes_finished > 0:
        avg_travel_time = np.mean(eval_stats["lengths"]) * env.dt
        summary_lines.append(f"完成回合数: {num_episodes_finished}")
        summary_lines.append(
            f"平均奖励: {np.mean(eval_stats['rewards']):.2f} ± {np.std(eval_stats['rewards']):.2f}"
        )
        summary_lines.append(
            f"平均回合成本: {np.mean(eval_stats['costs']):.2f} ± {np.std(eval_stats['costs']):.2f}"
        )
        summary_lines.append(f"平均通行时间 (秒): {avg_travel_time:.2f}")
        summary_lines.append(
            f"平均加加速度 (m/s^3): {np.mean(eval_stats.get('avg_jerk', [0])):.2f}"
        )
        summary_lines.append(
            f"平均 sCTE (米): {np.mean(eval_stats.get('avg_scte', [0])):.2f}"
        )

        # 能耗汇总
        if eval_stats.get("episode_energy_net_kwh"):
            mean_net = np.mean(eval_stats["episode_energy_net_kwh"])
            mean_consume = np.mean(eval_stats["episode_energy_consume_kwh"])
            mean_regen = np.mean(eval_stats["episode_energy_regen_kwh"])
            summary_lines.append(
                f"平均每回合净电能 (kWh): {mean_net:.4f} "
                f"(消耗 {mean_consume:.4f} kWh, 回收 {mean_regen:.4f} kWh)"
            )

        summary_lines.append("-" * 20)
        summary_lines.append(f"成功率: {eval_stats['success'] / num_episodes_finished:.2%}")
        summary_lines.append(f"碰撞率: {eval_stats['collision'] / num_episodes_finished:.2%}")
        summary_lines.append(f"偏离轨迹率: {eval_stats['off_track'] / num_episodes_finished:.2%}")
        summary_lines.append(f"超时率: {eval_stats['timeout'] / num_episodes_finished:.2%}")
    else:
        summary_lines.append("没有完成任何回合的评估。")

    # 在控制台打印结果
    print("\n\n")
    for line in summary_lines:
        print(line)

    # 保存结果到文件
    summary_filepath = os.path.join(log_save_dir, "evaluation_summary.txt")
    try:
        with open(summary_filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(summary_lines))
        print(f"\n评估结果汇总已保存到: {summary_filepath}")
    except Exception as e:
        print(f"\n保存评估结果汇总失败: {e}")


if __name__ == "__main__":
    main()

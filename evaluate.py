"""
@file: evaluate.py
@description:
该文件是用于**评估和可视化**一个已经训练好的强化学习智能体的主脚本。
它会加载一个指定的模型权重文件，然后在带有人机交互界面的仿真环境中运行该智能体，
从而可以直观地、定性地评估其学习到的驾驶策略的性能和鲁棒性。

核心流程:

1.  **加载模型 (Load Model):**
    - 这是评估脚本与训练脚本最关键的区别。它首先会初始化一个与训练时结构相同的
      `DDPGAgent`，然后调用 `agent.load_model()` 方法，从一个检查点文件（`.pth`）
      中加载已经训练好的网络权重。
    - **注意：** 运行前需要手动修改脚本中的 `MODEL_PATH` 变量，使其指向正确的
      模型文件。

2.  **环境与引擎初始化 (Environment and Engine Initialization):**
    - 同时初始化 `TrafficEnv` (仿真后端) 和 `GameEngine` (可视化前端)，
      以便能够实时看到智能体的每一个动作和环境的反馈。

3.  **评估主循环 (Evaluation Loop):**
    - **确定性策略:** 在循环中，调用 `agent.select_action(state, add_noise=False)`
      来获取动作。`add_noise=False` 参数至关重要，它确保了我们评估的是智能体
      学习到的**确定性策略**，而不是其在训练时的探索行为。
    - **场景循环测试:** 当一个回合结束后，脚本会自动从一个预设的场景列表
      (`scenarios`) 中选择下一个场景进行测试。这允许我们系统性地、连续地评估
      智能体在不同挑战下的表现（例如，无保护左转、应对冲突等）。

4.  **可视化与交互:**
    - `GameEngine` 负责渲染仿真画面，让我们可以直观地观察智能体的驾驶行为，
      例如它是否平稳、是否能有效避免碰撞、是否能做出合理的让行决策等。
    - 用户仍然可以使用所有的交互快捷键（如暂停、缩放、移动视角）来仔细观察
      智能体的行为细节。
"""

import pygame
import torch
import time
import os
import argparse
import numpy as np
import pandas as pd
import random
from game_engine import GameEngine
from traffic_env import TrafficEnv
from sagi_ppo import SAGIPPOAgent
from ppo import PPOAgent

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
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO/SAGI-PPO agent.")
    parser.add_argument("--algo", type=str, default="ppo", choices=["sagi_ppo", "ppo"],
                        help="The algorithm of the trained agent to evaluate.")
    parser.add_argument("--model-dir", type=str,  default="models/ppo_20250825-203503",
                        help="Path to the directory containing the saved model files (e.g., 'models/sagi_ppo_YYYYMMDD-HHMMSS').")
    parser.add_argument("--num-episodes", type=int, default=1, 
                        help="Number of episodes to run for evaluation.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    return parser.parse_args()

def main():
    """主函数，用于可视化和量化评估一个训练好的 agent。"""
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
    
    scenarios = ['agent_only']

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
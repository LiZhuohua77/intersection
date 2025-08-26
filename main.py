"""
@file: main.py
@description:
该文件是整个强化学习交通仿真项目的**主程序入口**。
它负责初始化并编排仿真后端（模型 `TrafficEnv`）与可视化前端（视图/控制器 `GameEngine`）
之间的交互，并驱动主仿真循环。

核心职责:

1.  **初始化 (Initialization):**
    - 创建并实例化两个最顶层的核心对象：
        - `TrafficEnv`: 遵循Gymnasium接口的仿真环境，是整个仿真的“后端”或“模型”。
        - `GameEngine`: 基于Pygame的可视化与交互引擎，是仿真的“前端”或“视图-控制器”。

2.  **主循环 (Main Loop):**
    - 运行一个经典的“事件处理 -> 状态更新 -> 渲染”的游戏循环。
    - **事件处理:** 调用 `game_engine.handle_events()` 来响应用户输入，如暂停、退出、
      切换视角等。
    - **状态更新:** 在非暂停状态下，调用 `env.step()` 来驱动仿真世界向前演化一步。
    - **渲染:** 调用 `game_engine.render()` 将 `env` 的当前状态绘制到屏幕上。

3.  **强化学习智能体集成 (占位符):**
    - **注意:** 在当前版本的 `main` 函数中，智能体的动作是通过 `env.action_space.sample()`
      生成的，即**随机动作**。
    - 这里是为未来真正的强化学习智能体**预留的集成点**。接手开发时，需要将此行
      替换为从训练好的策略网络获取动作的逻辑（例如 `action = agent.select_action(observation)`）。

4.  **回合管理 (Episode Management):**
    - 在主循环中，它会检查 `env.step()` 返回的 `terminated` 或 `truncated` 标志，
      以判断当前回合是否结束。
    - 当一个回合结束后，它会打印最终奖励，调用 `matplotlib` 绘制智能体的速度曲线
      以供分析，并自动重置环境 (`env.reset()`) 以开始新的回合。
"""
from traffic_env import TrafficEnv
import matplotlib.pyplot as plt
import numpy as np
import argparse
from game_engine import GameEngine

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="运行交通仿真环境")
    parser.add_argument("--scenario", type=str, default="north_south_traffic", 
                      choices=["agent_only", "protected_left_turn", "unprotected_left_turn", 
                               "head_on_conflict", "random", "east_west_traffic", "north_south_traffic"],
                      help="要加载的场景")
    parser.add_argument("--scenario-aware", action="store_true", 
                      help="启用场景感知模式，生成场景树")
    return parser.parse_args()

def main():
    """主函数，负责编排RL环境和游戏引擎。"""
    # 解析命令行参数
    args = parse_args()
    
    # 1. 创建模型 (环境)
    env = TrafficEnv()
    
    # 2. 创建视图/控制器 (游戏引擎)
    game_engine = GameEngine(width=800, height=800)
    
    # 重置环境，获取初始状态
    options = {
        'scenario': args.scenario,
        'scenario_aware': args.scenario_aware
    }
    observation, info = env.reset(options=options)
    
    # 判断是否为只有背景车辆的场景
    is_background_only = args.scenario in ["east_west_traffic", "north_south_traffic"]
    
    # 打印场景信息
    print(f"加载场景: {args.scenario}")
    if is_background_only:
        print("【纯背景车辆模式】- 该场景没有RL智能体，仅用于观察和数据收集")
    if args.scenario_aware:
        print("场景感知模式已开启 - 将生成场景树用于预测分析")
    else:
        print("场景感知模式已关闭 - 不会生成场景树")
    
    try:
        while game_engine.is_running():
            # (1) 视图/控制器: 处理用户输入
            game_engine.handle_events(env)
            
            # (2) 模型: 演化 (仅在非暂停时)
            if not game_engine.input_handler.paused:
                # 对于只有背景车辆的场景，传入dummy动作(全0)
                action = np.zeros(env.action_space.shape) if is_background_only else env.action_space.sample()
                
                # b. 环境执行一步
                observation, reward, terminated, truncated, info = env.step(action)
                
                # c. 检查回合是否结束
                if terminated or truncated:
                    if is_background_only:
                        print("所有背景车辆已完成路径，场景结束!")
                    
                    # 判断是否为纯背景车辆场景
                    if is_background_only:
                        # 对于纯背景车辆场景，只绘制背景车辆的速度曲线
                        completed_vehicles = env.traffic_manager.completed_vehicles_data
                        if completed_vehicles:
                            plt.figure(figsize=(10, 6))
                            for vehicle_data in completed_vehicles:
                                vehicle_history = vehicle_data['history']
                                if vehicle_history:
                                    plt.plot(vehicle_history, label=f"Vehicle #{vehicle_data['id']}")
                            plt.title(f"Background Vehicles Speed Curves - {args.scenario}")
                            plt.xlabel("Time Steps")
                            plt.ylabel("Speed (m/s)")
                            plt.grid(True)
                            plt.legend(loc='best')
                            plt.tight_layout()
                            plt.show()
                            
                            print(f"Collected and plotted speed history data for {len(completed_vehicles)} background vehicles")
                            
                            # 这里可以添加代码来保存数据到文件
                            # import pandas as pd
                            # df = pd.DataFrame({f"vehicle_{data['id']}": data['history'] for data in completed_vehicles})
                            # df.to_csv(f"background_speeds_{args.scenario}.csv", index=False)
                    else:
                        # 对于有RL Agent的常规场景
                        speed_curve = env.rl_agent.get_speed_history()
                        if speed_curve:
                            plt.figure(figsize=(12, 6))
                            plt.subplot(1, 2, 1)
                            plt.plot(speed_curve, 'b-', linewidth=2)
                            plt.title("RL Agent Speed Curve")
                            plt.xlabel("Time Step")
                            plt.ylabel("Speed (m/s)")
                            plt.grid(True)
                        
                        # 获取并绘制所有背景车辆的速度曲线
                        completed_vehicles = env.traffic_manager.completed_vehicles_data
                        if completed_vehicles:
                            plt.subplot(1, 2, 2)
                            for vehicle_data in completed_vehicles:
                                vehicle_history = vehicle_data['history']
                                if vehicle_history:
                                    plt.plot(vehicle_history, label=f"Vehicle #{vehicle_data['id']}")
                            plt.title("Background Vehicles Speed Curves")
                            plt.xlabel("Time Step")
                            plt.ylabel("Speed (m/s)")
                            plt.grid(True)
                            plt.legend(loc='best', fontsize='small')
                        
                        plt.tight_layout()
                        plt.show()
                        
                        # 可选：保存速度历史数据用于场景树分析
                        if hasattr(env.traffic_manager, 'completed_vehicles_data') and env.traffic_manager.completed_vehicles_data:
                            print(f"收集到 {len(env.traffic_manager.completed_vehicles_data)} 辆背景车辆的速度历史数据")
                    
                    print("重置环境中...")
                    observation, info = env.reset(options=options)

            # (3) 视图: 渲染当前模型状态
            game_engine.render(env)
            
            # (4) 控制器: 保持帧率稳定
            game_engine.tick()

    except KeyboardInterrupt:
        print("\n程序被用户中断")
    finally:
        game_engine.quit()
        print("仿真结束")

if __name__ == "__main__":
    main()
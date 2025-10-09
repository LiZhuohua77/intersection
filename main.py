"""
@file: main.py
@description:
该文件是整个强化学习交通仿真项目的主程序入口。
它负责初始化仿真环境和可视化引擎，并驱动整个仿真过程。

主要函数:
1. parse_args(): 解析命令行参数，确定要加载的交通场景
2. main(): 主函数，创建环境和游戏引擎，并执行主仿真循环

核心职责:

1.  **初始化 (Initialization):**
    - 创建并实例化两个最顶层的核心对象：
        - `TrafficEnv`: 遵循Gymnasium接口的仿真环境，是整个仿真的"后端"或"模型"。
        - `GameEngine`: 基于Pygame的可视化与交互引擎，是仿真的"前端"或"视图-控制器"。

2.  **主循环 (Main Loop):**
    - 运行一个经典的"事件处理 -> 状态更新 -> 渲染"的游戏循环。
    - 事件处理: 响应用户输入，如暂停、退出、切换视角等。
    - 状态更新: 在非暂停状态下，驱动仿真世界向前演化一步。
    - 渲染: 将当前状态绘制到屏幕上。

3.  **场景管理 (Scenario Management):**
    - 支持多种预定义交通场景的加载和切换
    - 分为含有RL智能体的场景和纯背景车辆的观察场景两大类

4.  **数据收集与可视化 (Data Collection & Visualization):**
    - 收集RL智能体和背景车辆的速度历史数据
    - 在回合结束时使用matplotlib绘制速度曲线进行分析
"""
from traffic_env import TrafficEnv
import matplotlib.pyplot as plt
import numpy as np
import argparse
from game_engine import GameEngine

def parse_args():
    """
    解析命令行参数。
    
    支持通过--scenario参数选择要加载的交通场景，可选场景包括：
    - agent_only: 只有RL智能体的简单场景
    - protected_left_turn: 受保护左转场景
    - unprotected_left_turn: 非保护左转场景
    - head_on_conflict: 正面交会冲突场景
    - random: 随机场景
    - east_west_traffic: 东西方向的纯背景车辆流
    - north_south_traffic: 南北方向的纯背景车辆流
    
    返回:
        argparse.Namespace: 包含解析后参数的命名空间对象
    """
    parser = argparse.ArgumentParser(description="运行交通仿真环境")
    parser.add_argument("--scenario", type=str, default="agent_only_simple", 
                      help="要加载的场景")
    return parser.parse_args()

def main():
    """
    主函数，负责编排RL环境和游戏引擎。
    
    主要步骤:
    1. 创建交通环境(TrafficEnv)和游戏引擎(GameEngine)
    2. 根据命令行参数加载指定场景
    3. 执行主仿真循环:
       - 处理用户输入事件
       - 更新环境状态
       - 渲染环境
    4. 回合结束时收集和可视化车辆速度数据
    5. 自动重置环境开始新回合
    
    特殊处理:
    - 区分含RL智能体的场景和纯背景车辆场景
    - 针对不同类型场景提供不同的数据收集和可视化方式
    """
    # 解析命令行参数
    args = parse_args()
    
    # 1. 创建模型 (环境)
    env = TrafficEnv()
    
    # 2. 创建视图/控制器 (游戏引擎)
    game_engine = GameEngine(width=800, height=800)
    
    # 重置环境，获取初始状态
    options = {
        'scenario': args.scenario
    }
    observation, info = env.reset(options=options)
    
    # 判断是否为只有背景车辆的场景
    is_background_only = args.scenario in ["east_west_traffic", "north_south_traffic"]
    
    # 打印场景信息
    print(f"加载场景: {args.scenario}")
    if is_background_only:
        print("【纯背景车辆模式】- 该场景没有RL智能体，仅用于观察和数据收集")
    
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
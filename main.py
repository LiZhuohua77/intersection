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
from game_engine import GameEngine

def main():
    """主函数，负责编排RL环境和游戏引擎。"""
    
    # 1. 创建模型 (环境)
    env = TrafficEnv()
    
    # 2. 创建视图/控制器 (游戏引擎)
    game_engine = GameEngine(width=800, height=800)
    
    # 重置环境，获取初始状态
    observation, info = env.reset(options={'scenario': 'unprotected_left_turn'})
    
    try:
        while game_engine.is_running():
            # (1) 视图/控制器: 处理用户输入
            game_engine.handle_events(env)
            
            # (2) 模型: 演化 (仅在非暂停时)
            if not game_engine.input_handler.paused:
                # a. 获取动作 (未来由您的RL模型提供)
                action = env.action_space.sample() # 当前使用随机动作
                
                # b. 环境执行一步
                observation, reward, terminated, truncated, info = env.step(action)
                
                # c. 检查回合是否结束
                if terminated or truncated:
                    print(f"回合结束! 最终奖励: {reward:.2f}")
                    
                    # 获取并绘制速度曲线
                    speed_curve = env.rl_agent.get_speed_history()
                    if speed_curve:
                        plt.figure()
                        plt.plot(speed_curve)
                        plt.title("Agent Speed Curve")
                        plt.show()

                    print("重置环境中...")
                    observation, info = env.reset(options={'scenario': 'unprotected_left_turn'})

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
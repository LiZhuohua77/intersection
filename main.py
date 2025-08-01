
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
    observation, info = env.reset(options={'scenario': 'protected_left_turn'})
    
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
                    observation, info = env.reset(options={'scenario': 'protected_left_turn'})

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
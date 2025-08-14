# evaluate.py (最终版 - 正确的评估脚本)

import pygame
from game_engine import GameEngine
from traffic_env import TrafficEnv
from ddpg import DDPGAgent 

def main():
    """主函数，用于可视化评估一个训练好的DDPG agent。"""
    
    # 1. 创建环境 (模型)，注意这里没有render_mode
    env = TrafficEnv()
    
    # 2. 创建游戏引擎 (视图/控制器)
    game_engine = GameEngine(width=800, height=800)

    # 3. 创建DDPG agent并加载训练好的模型权重
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_high = env.action_space.high
    
    agent = DDPGAgent(state_dim=state_dim, action_dim=action_dim, action_high=action_high)
    
    # --- 重要: 将这里的文件名替换为您实际保存的模型文件 ---
    MODEL_PATH = "ddpg_episode_400.pth" # 示例文件名
    agent.load_model(MODEL_PATH)
    
    # 重置环境，为第一个测试场景做准备
    state, info = env.reset(options={'scenario': 'agent_only'})
    
    print("--- 开始评估 ---")
    try:
        while game_engine.is_running():
            game_engine.handle_events(env)
            
            if not game_engine.input_handler.paused:
                action = agent.select_action(state, add_noise=False)
                next_state, reward, terminated, truncated, info = env.step(action)
                state = next_state
                
                if terminated or truncated:
                    print(f"回合结束。正在重置...")
                    # 自动循环测试场景
                    scenarios = ['protected_left_turn', 'unprotected_left_turn', 'head_on_conflict', 'agent_only']
                    episode_count = info.get('episode', 0)
                    next_scenario = scenarios[episode_count % len(scenarios)]
                    state, info = env.reset(options={'scenario': 'agent_only'})

            game_engine.render(env)
            game_engine.tick()

    except KeyboardInterrupt:
        print("\n评估被用户中断")
    finally:
        game_engine.quit()

if __name__ == "__main__":
    main()
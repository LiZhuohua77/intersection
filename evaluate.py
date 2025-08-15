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
import numpy as np
from tqdm import tqdm  # 用于显示漂亮的进度条

# 导入您的环境类
from traffic_env import TrafficEnv

# --- 可配置参数 ---
# 您希望测试多少个回合来取平均值？10-20个通常就足够了。
NUM_EPISODES_TO_TEST = 20
# 每个回合的最大步数，以防环境因为某些原因永远不结束
MAX_STEPS_PER_EPISODE = 2000 

def run_random_policy():
    """
    运行一个纯随机策略，并收集成本统计数据。
    """
    print("--- 开始运行纯随机策略以评估基准成本 ---")
    
    # 1. 初始化您的环境
    # 您可能需要根据您的环境初始化方式传入参数，例如 scenario
    env = TrafficEnv(scenario='agent_only_simple')

    all_episode_costs = []

    # 使用tqdm来显示进度
    for episode in tqdm(range(NUM_EPISODES_TO_TEST), desc="模拟随机策略"):
        # 重置环境，开始新回合
        observation, info = env.reset()
        episode_cost = 0.0
        
        for step in range(MAX_STEPS_PER_EPISODE):
            # 2. 从动作空间中随机采样一个动作
            action = env.action_space.sample()
            
            # 3. 执行随机动作
            observation, reward, terminated, truncated, info = env.step(action)
            
            # 4. 累积成本
            #    我们使用 .get('cost', 0) 来安全地获取成本值
            episode_cost += info.get('cost', 0)
            
            # 5. 检查回合是否结束
            if terminated or truncated:
                break
        
        # 记录这个完整回合的总成本
        all_episode_costs.append(episode_cost)

    env.close()

    # --- 6. 计算并打印最终的统计结果 ---
    print("\n" + "="*50)
    print("--- 随机策略成本统计结果 ---")
    if not all_episode_costs:
        print("警告：没有收集到任何回合的成本数据。")
    else:
        mean_cost = np.mean(all_episode_costs)
        std_dev_cost = np.std(all_episode_costs)
        min_cost = np.min(all_episode_costs)
        max_cost = np.max(all_episode_costs)
        
        print(f"测试的总回合数: {len(all_episode_costs)}")
        print(f"平均回合成本 (Mean Episode Cost): {mean_cost:.2f}")
        print(f"成本标准差 (Std Dev): {std_dev_cost:.2f}")
        print(f"最小回合成本 (Min): {min_cost:.2f}")
        print(f"最大回合成本 (Max): {max_cost:.2f}")
        
        print("\n--- 操作建议 ---")
        print(f"您的'--initial-cost-limit'参数应该被设置得比平均成本 {mean_cost:.2f} 更高。")
        print(f"一个合理的初始值可以是: {round(mean_cost * 1.5, -1)} 或 {round(mean_cost * 2.0, -1)}")
    print("="*50)

if __name__ == "__main__":
    run_random_policy()

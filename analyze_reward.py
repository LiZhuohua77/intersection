import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']   # Windows: 黑体
# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # Windows: 微软雅黑
# plt.rcParams['font.sans-serif'] = ['PingFang SC']  # macOS
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macOS
# plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']  # Linux
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题
# --- 把上面的绘图函数 plot_reward_composition 和 plot_actions_and_states 复制到这里 ---

def plot_reward_composition(df):
    """绘制奖励构成堆叠面积图和总奖励曲线"""
    if df.empty:
        print("数据为空，无法绘图。")
        return

    steps = df['step']
    reward_components = {
        '进度奖励 (Progress)': df['reward_progress'],
        '能量奖励 (Energy)': df['reward_energy'],
        '成本惩罚 (Cost Penalty)': df['reward_cost_penalty']
    }
    total_reward = df['total_reward']

    fig, ax = plt.subplots(figsize=(16, 8))

    # 绘制堆叠面积图
    ax.stackplot(steps, reward_components.values(),
                 labels=reward_components.keys(),
                 alpha=0.8)
    
    # 在上方绘制总奖励的曲线，用于对比
    ax.plot(steps, total_reward, color='black', linestyle='--', linewidth=2, label='总奖励 (Total Reward)')

    ax.legend(loc='upper left')
    ax.set_title('每一步的奖励构成分析', fontsize=16)
    ax.set_xlabel('步数 (Step)', fontsize=12)
    ax.set_ylabel('奖励值 (Reward)', fontsize=12)
    ax.grid(True)
    ax.axhline(0, color='gray', linewidth=0.8) # 添加y=0的参考线
    plt.tight_layout()
    plt.show()

    
def plot_actions_and_states(df):
    """绘制三合一图表：动作、运动状态、误差与成本"""
    if df.empty:
        print("数据为空，无法绘图。")
        return
        
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 15), sharex=True)

    steps = df['step']

    # 子图1: 动作 (Actions)
    ax1.plot(steps, df['action_accel'], label='加速度动作 (Accel Action)', color='blue')
    ax1.plot(steps, df['action_steer'], label='转向动作 (Steer Action)', color='green')
    ax1.set_title('智能体动作', fontsize=14)
    ax1.set_ylabel('动作值 (-1 to 1)')
    ax1.legend()
    ax1.grid(True)

    # 子图2: 运动状态 (Motion State)
    ax2.plot(steps, df['ego_vx'], label='纵向速度 (vx)', color='red')
    ax2.set_title('车辆运动状态', fontsize=14)
    ax2.set_ylabel('速度 (m/s)')
    ax2.legend()
    ax2.grid(True)
    
    # 子图3: 误差与成本 (Errors & Cost)
    ax3.plot(steps, df['cross_track_error'], label='横向误差 (CTE)', color='purple', linestyle='--')
    ax3.plot(steps, df['heading_error'], label='航向误差 (Heading Err)', color='orange', linestyle='--')
    
    # 使用第二个Y轴来绘制成本，因为它的量级可能不同
    ax3_twin = ax3.twinx()
    ax3_twin.plot(steps, df['raw_cost_potential'], label='原始成本 (Raw Cost)', color='black', alpha=0.7)
    
    ax3.set_title('误差与路径成本', fontsize=14)
    ax3.set_xlabel('步数 (Step)', fontsize=12)
    ax3.set_ylabel('误差值 (Error)')
    ax3_twin.set_ylabel('成本值 (Cost)')
    
    # 合并两个Y轴的图例
    lines, labels = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3_twin.legend(lines + lines2, labels + labels2, loc='upper left')
    
    ax3.grid(True)

    plt.tight_layout()
    plt.show()



# --- 主程序入口 ---
if __name__ == "__main__":
    try:
        # 加载最新的日志文件
        log_df = pd.read_csv("D:\Code\intersection\logs\sagi_ppo_20250818-232605\episode_3_log.csv")

        print("成功加载日志文件，开始绘图...")
        
        # 生成并显示图表
        plot_reward_composition(log_df)
        plot_actions_and_states(log_df)
        
        print("绘图完成。")
        
    except FileNotFoundError:
        print("错误：未找到 'episode_log.csv'。请先运行您的仿真并生成日志。")
    except Exception as e:
        print(f"发生错误: {e}")
"""
@file: analyze_reward.py
@description: 
本文件用于分析和可视化强化学习训练过程中的奖励、动作和状态数据。
主要功能包括：
1. 解析训练日志中的数据
2. 可视化奖励构成及其组件贡献
3. 可视化智能体动作、车辆状态和误差指标
此工具可帮助研究人员理解和调试强化学习智能体在交通场景中的行为表现。
"""

import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']   # Windows: 黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

def plot_reward_composition(df):
    """
    绘制奖励构成堆叠面积图和总奖励曲线。
    
    参数:
        df (DataFrame): 包含奖励组件数据的数据帧，必须包含以下列:
            - step: 训练步数
            - reward_velocity_tracking: 速度跟踪奖励
            - reward_time_penalty: 时间惩罚
            - reward_action_smoothness: 动作平滑度惩罚
            - reward_cost_penalty: 成本惩罚
            - total_reward: 总奖励值
    
    功能:
        1. 创建堆叠面积图显示各奖励组件的贡献
        2. 叠加总奖励曲线以便对比
        3. 添加图例和坐标轴标签
    
    返回:
        无，直接显示图表
    """
    if df.empty:
        print("数据为空，无法绘图。")
        return

    steps = df['step']
    reward_components = {
        '速度跟踪奖励 (Velocity Tracking)': df['reward_velocity_tracking'],
        '时间惩罚 (Time Penalty)': df['reward_time_penalty'],
        '动作平滑度惩罚 (Action Smoothness)': df['reward_action_smoothness'],
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
    """
    绘制三合一图表：动作、运动状态、误差与成本。
    
    参数:
        df (DataFrame): 包含动作和状态数据的数据帧，必须包含以下列:
            - step: 训练步数
            - action_accel: 加速度控制动作
            - action_steer: 转向控制动作
            - ego_vx: 车辆纵向速度
            - cross_track_error: 横向误差
            - heading_error: 航向误差
            - raw_cost_potential: 路径成本值
    
    功能:
        1. 第一子图：显示智能体的加速度和转向控制动作
        2. 第二子图：显示车辆运动状态（主要是纵向速度）
        3. 第三子图：同时显示横向误差、航向误差和路径成本
    
    返回:
        无，直接显示图表
    """
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
    """
    主函数：加载训练日志并生成可视化图表。
    
    流程:
        1. 尝试从指定路径加载最新的日志文件
        2. 调用绘图函数生成奖励构成分析图表
        3. 调用绘图函数生成动作与状态分析图表
        4. 处理可能的文件读取错误和其他异常
    """
    try:
        # 加载最新的日志文件
        log_df = pd.read_csv("D:\Code\intersection\logs\sagi_ppo_20250819-112407\episode_1_log.csv")

        print("成功加载日志文件，开始绘图...")
        
        # 生成并显示图表
        plot_reward_composition(log_df)
        plot_actions_and_states(log_df)
        
        print("绘图完成。")
        
    except FileNotFoundError:
        print("错误：未找到 'episode_log.csv'。请先运行您的仿真并生成日志。")
    except Exception as e:
        print(f"发生错误: {e}")
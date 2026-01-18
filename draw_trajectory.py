import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, BboxConnector, BboxPatch, TransformedBbox
import pandas as pd
import numpy as np
import glob
import os
import sys

# 尝试导入 Road 类
try:
    from road import Road
except ImportError:
    print("错误: 找不到 road.py。请确保该文件在当前目录下。")
    sys.exit(1)

# ==========================================
# 1. 配置区域
# ==========================================
SHOW_TRAJECTORIES = True 

# 仿真环境参数
ENV_WIDTH = 400
ENV_HEIGHT = 400
# [关键修改] 使用真实的比例
LANE_WIDTH = 4   

# 学术风格配色
STYLE_CONFIG = {
    'bg_color': 'white',          
    'lane_color': 'black',        
    'ref_line_color': '#666666',  
    'traj_agent_color': '#D62728',
    'traj_other_color': '#1F77B4',
    'grid_color': '#E5E5E5'       
}

# ==========================================
# 2. 绘制逻辑 (封装好，方便调用两次)
# ==========================================
def plot_road_background(ax, road_env):
    """绘制道路背景"""
    # 基础设置
    ax.set_facecolor(STYLE_CONFIG['bg_color'])
    ax.grid(True, linestyle='--', color=STYLE_CONFIG['grid_color'], alpha=0.8)

    cx, cy = road_env.center_x, road_env.center_y
    lw = road_env.lane_width
    
    # --- A. 绘制物理边界 ---
    lines = [
        [(0, cy - lw), (cx - 2*lw, cy - lw)],
        [(cx + 2*lw, cy - lw), (road_env.width, cy - lw)],
        [(0, cy + lw), (cx - 2*lw, cy + lw)],
        [(cx + 2*lw, cy + lw), (road_env.width, cy + lw)],
        [(cx - lw, 0), (cx - lw, cy - 2*lw)],
        [(cx - lw, cy + 2*lw), (cx - lw, road_env.height)],
        [(cx + lw, 0), (cx + lw, cy - 2*lw)],
        [(cx + lw, cy + 2*lw), (cx + lw, road_env.height)]
    ]
    for p1, p2 in lines:
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=STYLE_CONFIG['lane_color'], linewidth=1.5)

    # --- B. 绘制转角圆弧 ---
    # 修正后的位置逻辑
    corners = [
        (cx - 2*lw, cy + 2*lw, 270, 360), # 左上
        (cx + 2*lw, cy + 2*lw, 180, 270), # 右上
        (cx - 2*lw, cy - 2*lw, 0, 90),    # 左下
        (cx + 2*lw, cy - 2*lw, 90, 180)   # 右下
    ]
    for x, y, t1, t2 in corners:
        arc = patches.Arc((x, y), 2*lw, 2*lw, angle=0, theta1=t1, theta2=t2, 
                          color=STYLE_CONFIG['lane_color'], linewidth=1.5)
        ax.add_patch(arc)

    # --- C. 绘制参考线 ---
    ref_routes = [('vertical_up', '--'), ('turn_south_to_west', ':'), ('turn_south_to_east', ':')]
    plotted_labels = set()
    for route_key, style in ref_routes:
        if route_key in road_env.routes:
            points = road_env.routes[route_key]['smoothed']
            if len(points) > 0:
                xs, ys = zip(*[(p[0], p[1]) for p in points])
                lbl = 'Reference Path' if 'Reference Path' not in plotted_labels else None
                ax.plot(xs, ys, color=STYLE_CONFIG['ref_line_color'], linestyle=style, 
                        linewidth=1, alpha=0.6, label=lbl)
                if lbl: plotted_labels.add(lbl)

def plot_trajectories(ax):
    """绘制轨迹（修正文件名版）"""
    # 1. 修改这里的文件名模式，匹配你实际的 CSV 文件名
    # 原代码: glob.glob("trajectory_vehicle_RL_AGENT_*.csv")
    # 修正后: glob.glob("trajectory_agent_RL_AGENT_*.csv")
    target_files = sorted(glob.glob("trajectory_agent_RL_AGENT_*.csv"))
    
    # 同时也找一下其他车辆的（如果有的话）
    target_files.extend(sorted(glob.glob("trajectory_unfinished_*.csv")))

    if not target_files:
        print("警告: 依然没有找到文件，请再次检查文件名！")
        return

    for path in target_files:
        try:
            df = pd.read_csv(path)
            if 't' in df.columns: df = df.sort_values("t")
            
            x, y = df["x"], df["y"]
            is_agent = "RL_AGENT" in path
            
            # 样式设置
            color = STYLE_CONFIG['traj_agent_color'] if is_agent else STYLE_CONFIG['traj_other_color']
            # 将 agent 的层级(zorder)设高，防止被斑马线遮挡
            zorder = 20 if is_agent else 10 
            
            label_text = "RL Agent" if is_agent else "Other Vehicles"
            handles, labels = ax.get_legend_handles_labels()
            if label_text in labels: label_text = None
            
            ax.plot(x, y, color=color, linewidth=2.0 if is_agent else 0.8, 
                    alpha=1.0 if is_agent else 0.4, label=label_text, zorder=zorder)
            
            # 绘制起终点
            if len(x) > 0 and is_agent:
                ax.plot(x.iloc[0], y.iloc[0], 'o', color='green', markersize=4, zorder=zorder+1)
                ax.plot(x.iloc[-1], y.iloc[-1], 'x', color='black', markersize=6, zorder=zorder+1)
                
        except Exception as e:
            print(f"读取 {path} 出错: {e}")

# ==========================================
# 3. 主程序 (包含局部放大逻辑)
# ==========================================
# ==========================================
# 3. 主程序 (完整修正版)
# ==========================================
if __name__ == "__main__":
    # 初始化环境
    road = Road(width=ENV_WIDTH, height=ENV_HEIGHT, lane_width=LANE_WIDTH)
    
    # 1. 创建主画布
    fig, ax = plt.figure(figsize=(10, 10), dpi=150), plt.gca()
    
    # 2. 在主图上绘制内容
    ax.set_xlim(0, road.width)
    ax.set_ylim(road.height, 0) # 翻转Y轴
    ax.set_aspect('equal')
    
    plot_road_background(ax, road)
    if SHOW_TRAJECTORIES:
        plot_trajectories(ax)
    
    # ==========================================
    # [核心] 添加局部放大图 (Inset Zoom)
    # ==========================================
    
    # [A. 创建子图] (你漏掉了这一步)
    # zoom=6: 放大6倍, loc=2: 放在左上角 ('upper left')
    axins = zoomed_inset_axes(ax, zoom=1.8, loc=1) 
    
    # [B. 在子图上画内容] (必须再画一遍，否则子图是空的)
    plot_road_background(axins, road)
    if SHOW_TRAJECTORIES:
        plot_trajectories(axins)
        
    # [C. 设置子图视野 - 聚焦路口中心] (你漏掉了这一步)
    zoom_radius = 50 
    x1, x2 = road.center_x - zoom_radius, road.center_x + zoom_radius
    y1, y2 = road.center_y + zoom_radius, road.center_y - zoom_radius # Y轴翻转注意
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    
    # 隐藏子图刻度，保持整洁
    axins.set_xticks([])
    axins.set_yticks([])

    # [D. 手动绘制连线 - 强制对应角相连]
    # 1. 确保放大图背景为白，防止线条穿帮
# [D. 手动绘制连线 - 修正坐标反转导致的错位]
    # ---------------------------------------------------
    # 1. 确保放大图背景为白，防止线条穿帮
    axins.set_facecolor('white')
    axins.set_zorder(20)

    # 2. 获取路口中心区域在主图中的几何位置 (Bbox)
    rect = TransformedBbox(axins.viewLim, ax.transData)

    # 3. 手动画出中心的小框 (Box)
    # zorder 设高一点，保证框框压在轨迹上面
    pp = BboxPatch(rect, fill=False, edgecolor="black", linewidth=1, zorder=15)
    ax.add_patch(pp)

    # 4. 手动画出两条连线 (Connector)
    # [关键修正]: 由于 ax.set_ylim(400, 0) 翻转了 Y 轴，
    # 主图数据框(rect)的 "Top" 和 "Bottom" 定义在视觉上是反的。
    # 视觉 Top-Left = 数据坐标的 min_x, min_y = Matplotlib Corner 3 (Bottom-Left)
    # 视觉 Bottom-Right = 数据坐标的 max_x, max_y = Matplotlib Corner 1 (Top-Right)
    
    # 第一条线：视觉左上角 连 视觉左上角
    # loc1=2 (Zoom图的左上), loc2=3 (中心框的数据Bottom-Left -> 视觉Top-Left)
    p1 = BboxConnector(axins.bbox, rect, loc1=2, loc2=3, 
                       fc="none", ec="0.5", linestyle="--", linewidth=0.8, zorder=1)
    ax.add_patch(p1)

    # 第二条线：视觉右下角 连 视觉右下角
    # loc1=4 (Zoom图的右下), loc2=1 (中心框的数据Top-Right -> 视觉Bottom-Right)
    p2 = BboxConnector(axins.bbox, rect, loc1=4, loc2=1, 
                       fc="none", ec="0.5", linestyle="--", linewidth=0.8, zorder=1)
    ax.add_patch(p2)
    # ---------------------------------------------------
    # ==========================================

    # 4. 设置标题和标签
    #ax.set_title("Trajectory Analysis with Intersection Detail", fontsize=12, pad=15)
    ax.set_xlabel("X Position (m)", fontsize=10)
    ax.set_ylabel("Y Position (m)", fontsize=10)
    
    # 图例移到右下角
    ax.legend(loc='lower right', frameon=True, framealpha=0.9) 
    
    plt.tight_layout()
    plt.show()
    # fig.savefig("trajectory_zoom_corrected.pdf", bbox_inches='tight')
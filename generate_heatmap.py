"""
@file: generate_heatmap.py
@description:
该文件用于生成高精度的交通路径势场热力图，并将其导出为PNG图像文件。
势场热力图可视化了不同路线的潜在场分布，用于分析车辆导航和路径规划策略。

主要功能:
1. 为指定的车辆路线生成高分辨率势场热力图
2. 将复杂的势场数据通过颜色映射直观呈现
3. 支持添加道路标线和路径轮廓作为参考
4. 导出为高质量PNG图像，用于后续分析和报告

主要函数:
- generate_heatmap_for_route(route_str): 为指定路径生成并导出热力图
- world_to_pixel_scaler(): 坐标转换辅助函数(在generate_heatmap_for_route内部定义)

使用方法:
通过修改TARGET_ROUTE_FOR_HEATMAP参数(如'N_S'、'E_W'等)来生成不同路径的热力图。
执行脚本后，会在当前目录生成对应的PNG图像文件。
"""

import pygame
import numpy as np
import matplotlib.pyplot as plt
from road import Road
from config import *

# ==============================================================================
# --- 配置参数 ---
# ==============================================================================

# 1. 在这里指定您想为其生成热力图的RL车辆的路径
#    格式为 '起始方向首字母_目标方向首字母', e.g., 'S_W', 'N_E', 'W_S'
TARGET_ROUTE_FOR_HEATMAP = 'N_S'  # <--- 修改这里以生成不同路径的热力图

# 2. 输出图片的文件名和尺寸
OUTPUT_FILENAME = f"potential_heatmap_for_route_{TARGET_ROUTE_FOR_HEATMAP}.png"
IMAGE_WIDTH_PX = 800
IMAGE_HEIGHT_PX = 800

# ==============================================================================
# --- 主程序 ---
# ==============================================================================

def generate_heatmap_for_route(route_str: str):
    """
    为单条指定路径，生成并导出一张静态的、高精度的势函数热力图PNG图片。
    
    该函数执行以下步骤:
    1. 初始化Road对象并预生成所有路径
    2. 逐像素计算势场值并填充网格
    3. 使用matplotlib的colormap将势场值映射为颜色
    4. 绘制热力图并添加道路轮廓作为参考
    5. 将最终图像保存为PNG文件
    
    参数:
        route_str: 路径标识符，格式为"起点_终点"，例如"N_S"表示从北向南
    
    返回:
        无，但会在当前目录生成一个PNG图像文件
    """
    print("Initializing Road object and pre-generating all paths...")
    road = Road(width=SCREEN_WIDTH, height=SCREEN_HEIGHT, lane_width=4)
    
    if route_str not in road.routes:
        print(f"错误：路径 '{route_str}' 不存在！")
        return

    print(f"Creating an image of size {IMAGE_WIDTH_PX}x{IMAGE_HEIGHT_PX} for route '{route_str}'...")
    potential_grid = np.zeros((IMAGE_HEIGHT_PX, IMAGE_WIDTH_PX))

    print("Calculating potential value for each pixel...")
    for px_y in range(IMAGE_HEIGHT_PX):
        for px_x in range(IMAGE_WIDTH_PX):
            world_x = (px_x / IMAGE_WIDTH_PX) * road.width
            world_y = (px_y / IMAGE_HEIGHT_PX) * road.height
            potential_grid[px_y, px_x] = road.get_potential_at_point(world_x, world_y, target_route_str=route_str)
        if (px_y + 1) % 100 == 0:
            print(f"Progress: {px_y + 1} / {IMAGE_HEIGHT_PX} rows complete.")

    print("Calculation complete. Generating image...")

    cmap = plt.get_cmap('jet')
    min_val, max_val = 0.0, 2.0
    normalized_grid = np.clip((potential_grid - min_val) / (max_val - min_val), 0, 1)
    colored_grid = (cmap(normalized_grid)[:, :, :3] * 255).astype(np.uint8)

    heatmap_surface = pygame.surfarray.make_surface(colored_grid.transpose(1, 0, 2))

    # --- ▼▼▼ 核心修正部分 ▼▼▼ ---

    # 1. 创建一个从"世界坐标"到"图像像素坐标"的缩放函数
    def world_to_pixel_scaler(world_x, world_y):
        """
        将世界坐标转换为热力图图像上的像素坐标
        
        参数:
            world_x: 世界坐标系中的X坐标
            world_y: 世界坐标系中的Y坐标
            
        返回:
            tuple: 对应的像素坐标(px_x, px_y)
        """
        px_x = (world_x / road.width) * IMAGE_WIDTH_PX
        px_y = (world_y / road.height) * IMAGE_HEIGHT_PX
        return (int(px_x), int(px_y))

    # 2. 在所有绘图函数中，传入这个缩放函数
    print("Drawing road context on top of the heatmap...")
    # 现在道路标线会被正确缩放
    #road.draw_road_lines(heatmap_surface, transform_func=world_to_pixel_scaler)
    
    route_data = road.routes[route_str]
    
    # 3. 对要绘制的路径点也应用缩放
    # (为了验证对齐，建议您取消下面几行的注释)
    
    # 绘制边界
    left_b = [world_to_pixel_scaler(p[0], p[1]) for p in route_data['boundaries']['left']]
    right_b = [world_to_pixel_scaler(p[0], p[1]) for p in route_data['boundaries']['right']]
    #pygame.draw.lines(heatmap_surface, (255, 255, 255, 100), False, left_b, 2)
    #pygame.draw.lines(heatmap_surface, (255, 255, 255, 100), False, right_b, 2)
    
    # 绘制中心线
    centerline = [world_to_pixel_scaler(p[0], p[1]) for p in route_data['smoothed']]
    #pygame.draw.lines(heatmap_surface, (255, 0, 255), False, centerline, 3) # 洋红色中心线
    
    # --- ▲▲▲ 核心修正部分结束 ▲▲▲ ---

    print(f"Saving image to {OUTPUT_FILENAME}...")
    pygame.image.save(heatmap_surface, OUTPUT_FILENAME)
    print("Done!")
    
if __name__ == '__main__':
    pygame.init()
    generate_heatmap_for_route(TARGET_ROUTE_FOR_HEATMAP)
    pygame.quit()
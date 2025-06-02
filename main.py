import pygame
import math
from road import Road
from vehicle import Vehicle

# 初始化 pygame
pygame.init()

# 设置窗口
width, height = 800, 800
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("道路十字路口")

# 创建道路对象
road = Road(width=width, height=height, lane_width=40)

# 创建车辆
vehicles = [
    Vehicle(road, 'west', 'north', 1),  # 南到北直行
]

# 游戏循环
clock = pygame.time.Clock()
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    # 清屏（深灰色背景，模拟沥青路面）
    screen.fill((50, 50, 50))
    
    # 绘制道路
    road.draw_road_lines(screen)      # 绘制白色道路边界线
    road.draw_center_lines(screen)    # 绘制黄色车道中心线
    road.visualize_route_points(screen, 'west', 'north', point_radius=4)  # 绘制路径点

    # 更新并绘制车辆
    for vehicle in vehicles:
        vehicle.update(0.1)  # 传入较小的时间步长使移动更平滑
        vehicle.draw(screen)

    # 更新显示
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
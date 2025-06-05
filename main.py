import pygame
import math
from road import Road
from vehicle import Vehicle

# 初始化 pygame
pygame.init()

# 设置窗口
width, height = 800, 800
screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
pygame.display.set_caption("道路十字路口")

# 创建道路对象
road = Road()

# 创建车辆
vehicles = [
    Vehicle(road, 'west', 'north', 1),  # 南到北直行
    #Vehicle(road, 'west', 'north', 2),  # 北到东直行
]

# 游戏循环
clock = pygame.time.Clock()
running = True
paused = False  # 添加暂停状态

# 视图变换参数
zoom = 1.0
camera_x, camera_y = width/2, height/2  # 摄像机位置（屏幕中心）
dragging = False
drag_start_x, drag_start_y = 0, 0

# 视图变换函数
def world_to_screen(x, y):
    screen_x = (x - camera_x) * zoom + width/2
    screen_y = (y - camera_y) * zoom + height/2
    return screen_x, screen_y

def screen_to_world(screen_x, screen_y):
    x = (screen_x - width/2) / zoom + camera_x
    y = (screen_y - height/2) / zoom + camera_y
    return x, y

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # 左键按下开始拖动
                dragging = True
                drag_start_x, drag_start_y = event.pos
            elif event.button == 4:  # 滚轮向上滚动，放大
                zoom *= 1.1
                zoom = min(zoom, 10.0)  # 限制最大缩放
            elif event.button == 5:  # 滚轮向下滚动，缩小
                zoom *= 0.9
                zoom = max(zoom, 0.1)  # 限制最小缩放
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:  # 左键释放结束拖动
                dragging = False
        elif event.type == pygame.MOUSEMOTION:
            if dragging:
                # 计算拖动距离并更新摄像机位置
                dx, dy = event.pos[0] - drag_start_x, event.pos[1] - drag_start_y
                camera_x -= dx / zoom
                camera_y -= dy / zoom
                drag_start_x, drag_start_y = event.pos
        elif event.type == pygame.VIDEORESIZE:
            # 处理窗口大小变化
            width, height = event.size
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:  # 按空格键暂停/恢复
                paused = not paused
                print(f"仿真 {'暂停' if paused else '继续'}")
            elif event.key == pygame.K_p:  # 按P键也可以暂停/恢复
                paused = not paused
                print(f"仿真 {'暂停' if paused else '继续'}")
            elif event.key == pygame.K_b:  # 按B键切换自行车模型可视化
                for vehicle in vehicles:
                    vehicle.toggle_bicycle_visualization()
            elif event.key == pygame.K_d:  # 按D键切换调试信息
                for vehicle in vehicles:
                    vehicle.toggle_debug_info()
            elif event.key == pygame.K_r:  # 按R键重置仿真
                # 重新创建车辆
                vehicles.clear()
                vehicles.extend([

                    Vehicle(road, 'west', 'north', 1),
                    #Vehicle(road, 'west', 'north', 2),
                ])
                print("仿真已重置")
    
    # 清屏（深灰色背景，模拟沥青路面）
    screen.fill((50, 50, 50))
    
    # 创建一个临时的Surface用于视图变换
    temp_surface = pygame.Surface((width, height), pygame.SRCALPHA)
    temp_surface.fill((50, 50, 50))
    
    # 绘制道路到临时Surface
    road.draw_road_lines(temp_surface, transform_func=world_to_screen)
    road.draw_center_lines(temp_surface, transform_func=world_to_screen)
    road.visualize_route_points(temp_surface, 'west', 'north', point_radius=4, transform_func=world_to_screen)
    
    # 只有在非暂停状态下才更新车辆位置
    if not paused:
        for vehicle in vehicles:
            vehicle.update(0.1)  # 传入较小的时间步长使移动更平滑
    
    # 绘制车辆（无论是否暂停都要绘制）
    for vehicle in vehicles:
        vehicle.draw(temp_surface, transform_func=world_to_screen, scale=zoom)
    
    # 将临时Surface绘制到屏幕
    screen.blit(temp_surface, (0, 0))
    
    # 显示暂停状态和控制提示
    font = pygame.font.SysFont(None, 24)
    
    # 暂停状态指示
    if paused:
        pause_text = font.render("PAUSED", True, (255, 255, 0))
        pause_rect = pause_text.get_rect(center=(width//2, 30))
        # Draw semi-transparent background
        pause_bg = pygame.Surface((pause_rect.width + 20, pause_rect.height + 10), pygame.SRCALPHA)
        pause_bg.fill((0, 0, 0, 128))
        screen.blit(pause_bg, (pause_rect.x - 10, pause_rect.y - 5))
        screen.blit(pause_text, pause_rect)
    
    # Control hints
    controls = [
        "Space/P: Pause/Resume",
        "B: Toggle Bicycle Model",
        "D: Toggle Debug Info", 
        "R: Reset Simulation",
        "Mouse Drag: Move View",
        "Mouse Wheel: Zoom"
    ]
    
    small_font = pygame.font.SysFont(None, 18)
    for i, control in enumerate(controls):
        text = small_font.render(control, True, (200, 200, 200))
        screen.blit(text, (10, height - 120 + i * 20))
    
    # Display current zoom ratio and position info (optional)
    info_text = small_font.render(f"Zoom: {zoom:.2f}x  Position: ({camera_x:.0f}, {camera_y:.0f})", 
                                  True, (150, 150, 150))
    screen.blit(info_text, (10, 10))

    # 更新显示
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
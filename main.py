import pygame
import math
from road import Road
from traffic import TrafficManager

# 初始化 pygame
pygame.init()

# 设置窗口
width, height = 800, 800
screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
pygame.display.set_caption("道路十字路口 - 交通流量仿真")

# 创建道路对象和交通管理器
road = Road()

# 创建自动驾驶车辆
""" vehicles = [
    Vehicle(road, 'west', 'north', 1),  # 自动驾驶车辆
    Vehicle(road, 'east', 'south', 2),  # 添加更多车辆用于测试
] """

traffic_manager = TrafficManager(road, max_vehicles=15)

# 游戏循环
clock = pygame.time.Clock()
running = True
paused = False

# 视图变换参数
zoom = 1.0
camera_x, camera_y = width/2, height/2
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
                zoom = min(zoom, 10.0)
            elif event.button == 5:  # 滚轮向下滚动，缩小
                zoom *= 0.9
                zoom = max(zoom, 0.1)
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                dragging = False
        elif event.type == pygame.MOUSEMOTION:
            if dragging:
                dx, dy = event.pos[0] - drag_start_x, event.pos[1] - drag_start_y
                camera_x -= dx / zoom
                camera_y -= dy / zoom
                drag_start_x, drag_start_y = event.pos
        elif event.type == pygame.VIDEORESIZE:
            width, height = event.size
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                paused = not paused
                print(f"仿真 {'暂停' if paused else '继续'}")
            elif event.key == pygame.K_p:
                paused = not paused
                print(f"仿真 {'暂停' if paused else '继续'}")
            elif event.key == pygame.K_b:
                for vehicle in traffic_manager.vehicles:
                    vehicle.toggle_bicycle_visualization()
            elif event.key == pygame.K_d:
                for vehicle in traffic_manager.vehicles:
                    vehicle.toggle_debug_info()
            elif event.key == pygame.K_r:
                traffic_manager.clear_all_vehicles()
                print("仿真已重置")
            elif event.key == pygame.K_h:
                print("控制帮助:")
                print("Space/P - 暂停/恢复")
                print("B - 切换自行车模型可视化")
                print("D - 切换调试信息")
                print("R - 重置仿真")
                print("1-4 - 切换交通模式")
                print("H - 显示此帮助")
            # 交通模式切换
            elif event.key == pygame.K_1:
                traffic_manager.set_traffic_pattern('light')
            elif event.key == pygame.K_2:
                traffic_manager.set_traffic_pattern('normal')
            elif event.key == pygame.K_3:
                traffic_manager.set_traffic_pattern('rush_hour')
            elif event.key == pygame.K_4:
                traffic_manager.set_traffic_pattern('night')
    
    # 清屏
    screen.fill((50, 50, 50))
    
    # 创建临时Surface用于视图变换
    temp_surface = pygame.Surface((width, height), pygame.SRCALPHA)
    temp_surface.fill((50, 50, 50))
    
    # 绘制道路
    road.draw_road_lines(temp_surface, transform_func=world_to_screen)
    road.draw_center_lines(temp_surface, transform_func=world_to_screen)
    
    # 更新交通管理器
    if not paused:
        traffic_manager.update(0.1)
    
    # 绘制所有车辆
    for vehicle in traffic_manager.vehicles:
        vehicle.draw(temp_surface, transform_func=world_to_screen, scale=zoom)
    
    # 将临时Surface绘制到屏幕
    screen.blit(temp_surface, (0, 0))
    
    # 显示状态信息
    font = pygame.font.SysFont(None, 24)
    small_font = pygame.font.SysFont(None, 18)
    
    # 暂停状态
    if paused:
        pause_text = font.render("PAUSED", True, (255, 255, 0))
        pause_rect = pause_text.get_rect(center=(width//2, 30))
        pause_bg = pygame.Surface((pause_rect.width + 20, pause_rect.height + 10), pygame.SRCALPHA)
        pause_bg.fill((0, 0, 0, 128))
        screen.blit(pause_bg, (pause_rect.x - 10, pause_rect.y - 5))
        screen.blit(pause_text, pause_rect)
    
    # 交通统计信息
    traffic_manager.draw_debug_info(screen, small_font)
    
    # 控制提示
    controls = [
        "1-4: Traffic Patterns",
        "Space/P: Pause/Resume", 
        "B: Bicycle Model",
        "D: Debug Info",
        "R: Reset"
    ]
    
    for i, control in enumerate(controls):
        text = small_font.render(control, True, (150, 150, 150))
        screen.blit(text, (width - 180, height - 90 + i * 15))
    
    # 缩放和位置信息
    info_text = small_font.render(f"Zoom: {zoom:.2f}x  Position: ({camera_x:.0f}, {camera_y:.0f})", 
                                  True, (150, 150, 150))
    screen.blit(info_text, (width - 250, 10))

    # 更新显示
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
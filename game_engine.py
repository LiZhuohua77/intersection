# game_engine.py

import pygame
import math

class Camera:
    """相机系统，负责视图变换"""
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.zoom = 1.0
        self.x = width / 2
        self.y = height / 2
        self.dragging = False
        self.drag_start_x = 0
        self.drag_start_y = 0
    
    def world_to_screen(self, x, y):
        """世界坐标转屏幕坐标"""
        screen_x = (x - self.x) * self.zoom + self.width / 2
        screen_y = (y - self.y) * self.zoom + self.height / 2
        return screen_x, screen_y
    
    def screen_to_world(self, screen_x, screen_y):
        """屏幕坐标转世界坐标"""
        x = (screen_x - self.width / 2) / self.zoom + self.x
        y = (screen_y - self.height / 2) / self.zoom + self.y
        return x, y
    
    def zoom_in(self, factor=1.1):
        """放大"""
        self.zoom *= factor
        self.zoom = min(self.zoom, 10.0)
    
    def zoom_out(self, factor=0.9):
        """缩小"""
        self.zoom *= factor
        self.zoom = max(self.zoom, 0.1)
    
    def start_drag(self, pos):
        """开始拖拽"""
        self.dragging = True
        self.drag_start_x, self.drag_start_y = pos
    
    def update_drag(self, pos):
        """更新拖拽"""
        if self.dragging:
            dx, dy = pos[0] - self.drag_start_x, pos[1] - self.drag_start_y
            self.x -= dx / self.zoom
            self.y -= dy / self.zoom
            self.drag_start_x, self.drag_start_y = pos
    
    def stop_drag(self):
        """停止拖拽"""
        self.dragging = False
    
    def resize(self, width, height):
        """窗口大小改变"""
        self.width = width
        self.height = height


class InputHandler:
    """输入处理器，负责处理键盘和鼠标事件"""
    def __init__(self):
        self.paused = False
        self.show_help = False
    
    def handle_event(self, event, camera, traffic_manager):
        """处理单个事件"""
        if event.type == pygame.MOUSEBUTTONDOWN:
            return self._handle_mouse_down(event, camera)
        elif event.type == pygame.MOUSEBUTTONUP:
            return self._handle_mouse_up(event, camera)
        elif event.type == pygame.MOUSEMOTION:
            return self._handle_mouse_motion(event, camera)
        elif event.type == pygame.VIDEORESIZE:
            return self._handle_resize(event, camera)
        elif event.type == pygame.KEYDOWN:
            return self._handle_keydown(event, traffic_manager)
        return False
    
    def _handle_mouse_down(self, event, camera):
        if event.button == 1:  # 左键按下开始拖动
            camera.start_drag(event.pos)
        elif event.button == 4:  # 滚轮向上滚动，放大
            camera.zoom_in()
        elif event.button == 5:  # 滚轮向下滚动，缩小
            camera.zoom_out()
        return False
    
    def _handle_mouse_up(self, event, camera):
        if event.button == 1:
            camera.stop_drag()
        return False
    
    def _handle_mouse_motion(self, event, camera):
        camera.update_drag(event.pos)
        return False
    
    def _handle_resize(self, event, camera):
        camera.resize(*event.size)
        return False
    
    def _handle_keydown(self, event, traffic_manager):
        if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
            return True  # Exit game
        elif event.key == pygame.K_SPACE or event.key == pygame.K_p:
            self.paused = not self.paused
            print(f"Simulation {'paused' if self.paused else 'resumed'}")
        elif event.key == pygame.K_b:
            for vehicle in traffic_manager.vehicles:
                vehicle.toggle_bicycle_visualization()
        elif event.key == pygame.K_d:
            for vehicle in traffic_manager.vehicles:
                vehicle.toggle_debug_info()
        elif event.key == pygame.K_t: 
             for vehicle in traffic_manager.vehicles:
                vehicle.toggle_path_visualization()
        elif event.key == pygame.K_r:
            traffic_manager.clear_all_vehicles()
            print("Simulation reset")
        elif event.key == pygame.K_h:
            self.show_help = not self.show_help
            if self.show_help:
                self._print_help()
        elif event.key == pygame.K_w:
            south_north_car, east_west_car = traffic_manager.create_test_scenario()
            print("测试场景已创建！")
        elif event.key == pygame.K_1:
            traffic_manager.set_traffic_pattern('light')
        elif event.key == pygame.K_2:
            traffic_manager.set_traffic_pattern('normal')
        elif event.key == pygame.K_3:
            traffic_manager.set_traffic_pattern('rush_hour')
        elif event.key == pygame.K_4:
            traffic_manager.set_traffic_pattern('night')
        return False

    
    def _print_help(self):
        """打印帮助信息"""
        help_text = """
        控制帮助:
        Space/P - 暂停/恢复
        B - 切换自行车模型可视化
        D - 切换调试信息
        R - 重置仿真
        1-4 - 切换交通模式 (轻度/正常/高峰/夜间)
        H - 显示/隐藏此帮助
        ESC/Q - 退出
        鼠标滚轮 - 缩放
        鼠标拖拽 - 移动视角
        """
        #print(help_text)


class Renderer:
    """渲染器，负责绘制所有内容"""
    def __init__(self, screen):
        self.screen = screen
        self.font = pygame.font.SysFont(None, 24)
        self.small_font = pygame.font.SysFont(None, 18)
        self.background_color = (50, 50, 50)
    
    def render_frame(self, road, traffic_manager, camera, input_handler):
        """渲染一帧"""
        width, height = self.screen.get_size()
        
        # 清屏
        self.screen.fill(self.background_color)
        
        # 创建临时Surface用于视图变换
        temp_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        temp_surface.fill(self.background_color)
        
        # 绘制道路
        road.draw_road_lines(temp_surface, transform_func=camera.world_to_screen)
        road.draw_center_lines(temp_surface, transform_func=camera.world_to_screen)
        road.draw_conflict_zones(temp_surface, transform_func=camera.world_to_screen)
        
        # 绘制所有车辆
        for vehicle in traffic_manager.vehicles:
            vehicle.draw(temp_surface, transform_func=camera.world_to_screen, small_font=self.small_font, scale=camera.zoom)
        
        
        # 将临时Surface绘制到屏幕
        self.screen.blit(temp_surface, (0, 0))
        
        # 绘制UI
        self._render_ui(traffic_manager, camera, input_handler, width, height)
    
    
    def _render_ui(self, traffic_manager, camera, input_handler, width, height):
        """渲染用户界面"""
        # 暂停状态
        if input_handler.paused:
            self._render_pause_indicator(width, height)
        
        # 交通统计信息
        traffic_manager.draw_debug_info(self.screen, self.small_font)
        
        # 控制提示
        self._render_controls(width, height)
        
        # 缩放和位置信息
        self._render_camera_info(camera, width)
        
        # 帮助信息
        if input_handler.show_help:
            self._render_help(width, height)
    
    def _render_pause_indicator(self, width, height):
        """渲染暂停指示器"""
        pause_text = self.font.render("PAUSED", True, (255, 255, 0))
        pause_rect = pause_text.get_rect(center=(width//2, 30))
        pause_bg = pygame.Surface((pause_rect.width + 20, pause_rect.height + 10), pygame.SRCALPHA)
        pause_bg.fill((0, 0, 0, 128))
        self.screen.blit(pause_bg, (pause_rect.x - 10, pause_rect.y - 5))
        self.screen.blit(pause_text, pause_rect)
    
    def _render_controls(self, width, height):
        """渲染控制提示"""
        controls = [
            "Space/P: Pause/Resume", 
            "B: Bicycle Model",
            "D: Debug Info",
            "T: Toggle Path Visualization",
            "R: Reset",
            "H: Help",
            "ESC/Q: Quit"
        ]
        
        for i, control in enumerate(controls):
            text = self.small_font.render(control, True, (150, 150, 150))
            self.screen.blit(text, (width - 180, height - 105 + i * 15))
    
    def _render_camera_info(self, camera, width):
        """渲染相机信息"""
        info_text = self.small_font.render(
            f"Zoom: {camera.zoom:.2f}x  Position: ({camera.x:.0f}, {camera.y:.0f})", 
            True, (150, 150, 150)
        )
        self.screen.blit(info_text, (width - 250, 10))
    
    def _render_help(self, width, height):
        """渲染帮助信息"""
        help_bg = pygame.Surface((400, 300), pygame.SRCALPHA)
        help_bg.fill((0, 0, 0, 200))
        help_rect = help_bg.get_rect(center=(width//2, height//2))
        self.screen.blit(help_bg, help_rect)
        
        help_lines = [
            "Keyboard Controls:",
            "Space/P - Pause/Resume",
            "B - Toggle Bicycle Model", 
            "D - Toggle Debug Info",
            "R - Reset Simulation",
            "T - Toggle Path Visualization",
            "H - Show/Hide Help",
            "ESC/Q - Quit",
            "",
            "Mouse Controls:",
            "Scroll Wheel - Zoom",
            "Drag - Move View"
        ]
        
        y_offset = help_rect.y + 20
        for line in help_lines:
            if line:
                text = self.small_font.render(line, True, (255, 255, 255))
            else:
                text = self.small_font.render("", True, (255, 255, 255))
            self.screen.blit(text, (help_rect.x + 20, y_offset))
            y_offset += 20


class GameEngine:
    """游戏引擎，现在主要负责渲染和用户交互。"""
    def __init__(self, width=800, height=800, title="RL Intersection Simulation"):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
        pygame.display.set_caption(title)
        
        self.clock = pygame.time.Clock()
        self.fps = 60
        self.running = True
        
        # 引擎现在拥有视图和控制器组件
        self.camera = Camera(width, height)
        self.input_handler = InputHandler()
        self.renderer = Renderer(self.screen)
    
    def handle_events(self, env):
        """处理所有事件。现在它需要env来访问traffic_manager。"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            # 将env.traffic_manager传递给输入处理器
            elif self.input_handler.handle_event(event, self.camera, env.traffic_manager):
                self.running = False
    
    def render(self, env):
        """渲染一帧。它从env获取所有需要绘制的对象。"""
        # 从env获取road和traffic_manager
        road = env.road
        traffic_manager = env.traffic_manager
        self.renderer.render_frame(road, traffic_manager, self.camera, self.input_handler)
        pygame.display.flip()
    
    def is_running(self):
        return self.running
    
    def tick(self):
        """控制帧率。"""
        self.clock.tick(self.fps)
    
    def quit(self):
        pygame.quit()
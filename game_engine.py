"""
@file: game_engine.py
@description:
该文件提供了交通仿真项目的可视化引擎，基于Pygame构建，负责处理所有用户交互和图形渲染功能。
它采用模块化设计，将相机系统、输入处理和渲染逻辑分离成独立的类，提高代码的可维护性和扩展性。

主要类和功能:
1. Camera - 相机系统，处理视图变换、缩放和移动
   - world_to_screen/screen_to_world: 坐标系转换函数
   - zoom_in/zoom_out: 视图缩放功能
   - start_drag/update_drag/stop_drag: 视图拖动功能
   
2. InputHandler - 输入处理器，负责处理所有用户输入
   - handle_event: 分发和处理各类事件
   - 提供键盘控制功能: 暂停/继续、切换可视化选项、场景重置、帮助显示等
   - 提供鼠标控制功能: 拖拽视图、缩放视图
   
3. Renderer - 渲染器，负责所有图形绘制
   - render_frame: 渲染完整场景，包括世界对象和UI元素
   - 支持势场热力图的生成和显示
   - 提供多种UI元素: 暂停指示器、控制提示、相机信息、帮助面板等
   
4. GameEngine - 游戏引擎主类，协调其他组件工作
   - 初始化Pygame环境和主要组件
   - 提供主循环控制: 事件处理、渲染、帧率控制
   - 为外部代码提供简洁的接口

交互功能:
- 相机控制: 鼠标拖拽移动视图，鼠标滚轮缩放
- 模拟控制: 暂停/继续、重置仿真、场景生成
- 可视化选项: 切换车辆模型、调试信息、路径显示、势场热力图等
"""

import pygame
import math

class Camera:
    """
    相机系统，负责管理视图变换、缩放和平移
    
    属性:
        width, height: 视口尺寸
        zoom: 当前缩放倍率
        x, y: 相机在世界坐标系中的位置
        dragging: 是否正在拖拽视图的标志
        drag_start_x, drag_start_y: 拖拽起始位置
    """
    def __init__(self, width, height):
        """
        初始化相机系统
        
        参数:
            width: 视口宽度(像素)
            height: 视口高度(像素)
        """
        self.width = width
        self.height = height
        self.zoom = 2.0
        self.x = 200
        self.y = 200
        self.dragging = False
        self.drag_start_x = 0
        self.drag_start_y = 0
    
    def world_to_screen(self, x, y):
        """
        将世界坐标转换为屏幕坐标
        
        参数:
            x, y: 世界坐标系中的点坐标
        
        返回:
            screen_x, screen_y: 对应的屏幕坐标
        """
        screen_x = (x - self.x) * self.zoom + self.width / 2
        screen_y = (y - self.y) * self.zoom + self.height / 2
        return screen_x, screen_y
    
    def screen_to_world(self, screen_x, screen_y):
        """
        将屏幕坐标转换为世界坐标
        
        参数:
            screen_x, screen_y: 屏幕坐标
        
        返回:
            x, y: 对应的世界坐标
        """
        x = (screen_x - self.width / 2) / self.zoom + self.x
        y = (screen_y - self.height / 2) / self.zoom + self.y
        return x, y
    
    def zoom_in(self, factor=1.1):
        """
        放大视图
        
        参数:
            factor: 缩放因子，默认为1.1(放大10%)
        """
        self.zoom *= factor
        self.zoom = min(self.zoom, 10.0)  # 限制最大缩放
    
    def zoom_out(self, factor=0.9):
        """
        缩小视图
        
        参数:
            factor: 缩放因子，默认为0.9(缩小10%)
        """
        self.zoom *= factor
        self.zoom = max(self.zoom, 0.1)  # 限制最小缩放
    
    def start_drag(self, pos):
        """
        开始拖拽视图
        
        参数:
            pos: 鼠标位置(x, y)，为拖拽起始点
        """
        self.dragging = True
        self.drag_start_x, self.drag_start_y = pos
    
    def update_drag(self, pos):
        """
        更新拖拽位置
        
        参数:
            pos: 当前鼠标位置(x, y)
        """
        if self.dragging:
            dx, dy = pos[0] - self.drag_start_x, pos[1] - self.drag_start_y
            self.x -= dx / self.zoom
            self.y -= dy / self.zoom
            self.drag_start_x, self.drag_start_y = pos
    
    def stop_drag(self):
        """停止拖拽操作"""
        self.dragging = False
    
    def resize(self, width, height):
        """
        调整视口大小，响应窗口大小变化
        
        参数:
            width, height: 新的视口尺寸
        """
        self.width = width
        self.height = height


class InputHandler:
    """
    输入处理器，负责处理所有键盘和鼠标事件，控制仿真状态
    
    属性:
        paused: 仿真是否暂停
        show_help: 是否显示帮助菜单
        show_potential_map: 是否显示势场热力图
    """
    def __init__(self):
        """初始化输入处理器"""
        self.paused = False
        self.show_help = False
        self.show_potential_map = False

    def handle_event(self, event, camera, traffic_manager):
        """
        处理单个pygame事件
        
        参数:
            event: pygame事件对象
            camera: 相机对象，用于视图操作
            traffic_manager: 交通管理器，控制车辆和交通流
        
        返回:
            bool: 如果需要退出程序则返回True，否则返回False
        """
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
        """
        处理鼠标按下事件
        
        参数:
            event: pygame鼠标事件
            camera: 相机对象
            
        返回:
            bool: 是否退出程序
        """
        if event.button == 1:  # 左键按下开始拖动
            camera.start_drag(event.pos)
        elif event.button == 4:  # 滚轮向上滚动，放大
            camera.zoom_in()
        elif event.button == 5:  # 滚轮向下滚动，缩小
            camera.zoom_out()
        return False
    
    def _handle_mouse_up(self, event, camera):
        """
        处理鼠标释放事件
        
        参数:
            event: pygame鼠标事件
            camera: 相机对象
            
        返回:
            bool: 是否退出程序
        """
        if event.button == 1:
            camera.stop_drag()
        return False
    
    def _handle_mouse_motion(self, event, camera):
        """
        处理鼠标移动事件
        
        参数:
            event: pygame鼠标移动事件
            camera: 相机对象
            
        返回:
            bool: 是否退出程序
        """
        camera.update_drag(event.pos)
        return False
    
    def _handle_resize(self, event, camera):
        """
        处理窗口大小改变事件
        
        参数:
            event: pygame窗口大小改变事件
            camera: 相机对象
            
        返回:
            bool: 是否退出程序
        """
        camera.resize(*event.size)
        return False
    
    def _handle_keydown(self, event, traffic_manager):
        """
        处理键盘按键事件
        
        参数:
            event: pygame键盘事件
            traffic_manager: 交通管理器对象
            
        返回:
            bool: 如果按ESC或Q则返回True表示退出，否则返回False
        """
        if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
            return True  # 退出游戏
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
        elif event.key == pygame.K_m: # M for Map
            self.show_potential_map = not self.show_potential_map
            print(f"Potential Field Map {'shown' if self.show_potential_map else 'hidden'}")
        return False

    
    def _print_help(self):
        """
        在控制台打印帮助信息
        """
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
    """
    渲染器，负责所有图形绘制，包括世界对象和UI元素
    
    属性:
        screen: pygame屏幕对象
        font/small_font: 用于文字渲染的字体
        background_color: 背景颜色
        heatmap_surface: 用于存储热力图的表面
        heatmap_generated_for_view: 当前视图是否已生成热力图
        last_camera_state: 上一帧的相机状态，用于检测视图变化
    """
    def __init__(self, screen):
        """
        初始化渲染器
        
        参数:
            screen: pygame屏幕对象
        """
        self.screen = screen
        self.font = pygame.font.SysFont(None, 24)
        self.small_font = pygame.font.SysFont(None, 18)
        self.background_color = (50, 50, 50)
        self.heatmap_surface = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        self.heatmap_generated_for_view = False
        self.last_camera_state = None

    def render_frame(self, road, traffic_manager, camera, input_handler):
        """
        渲染完整的一帧画面
        
        参数:
            road: 道路对象
            traffic_manager: 交通管理器
            camera: 相机对象
            input_handler: 输入处理器
        """
        width, height = self.screen.get_size()
        
        # 检查相机视角是否变化
        if self._has_camera_view_changed(camera):
            self.heatmap_generated_for_view = False

        # 清理主屏幕
        self.screen.fill(self.background_color)
        
        # 创建透明画板用于世界对象渲染
        temp_surface = pygame.Surface((width, height), pygame.SRCALPHA)

        # 绘制热力图(如果需要)
        if input_handler.show_potential_map:
            if not self.heatmap_generated_for_view:
                self._render_potential_heatmap(road, camera, self.heatmap_surface)
                self.heatmap_generated_for_view = True
            temp_surface.blit(self.heatmap_surface, (0, 0))

        # 绘制道路和车辆
        road.draw_road_lines(temp_surface, transform_func=camera.world_to_screen)
        road.draw_conflict_zones(temp_surface, transform_func=camera.world_to_screen)
        
        for vehicle in traffic_manager.vehicles:
            vehicle.draw(temp_surface, transform_func=camera.world_to_screen, small_font=self.small_font, scale=camera.zoom)
        
        # 将世界对象画布贴到主屏幕
        self.screen.blit(temp_surface, (0, 0))
        
        # 绘制UI元素
        self._render_ui(traffic_manager, camera, input_handler, width, height)
    
    
    def _render_ui(self, traffic_manager, camera, input_handler, width, height):
        """
        渲染所有UI元素
        
        参数:
            traffic_manager: 交通管理器
            camera: 相机对象
            input_handler: 输入处理器
            width: 屏幕宽度
            height: 屏幕高度
        """
        # 暂停状态指示
        if input_handler.paused:
            self._render_pause_indicator(width, height)
        
        # 交通统计信息
        traffic_manager.draw_debug_info(self.screen, self.small_font)
        
        # 控制提示
        self._render_controls(width, height)
        
        # 相机信息
        self._render_camera_info(camera, width)
        
        # 帮助菜单
        if input_handler.show_help:
            self._render_help(width, height)
    
    def _render_pause_indicator(self, width, height):
        """
        渲染暂停指示器
        
        参数:
            width: 屏幕宽度
            height: 屏幕高度
        """
        pause_text = self.font.render("PAUSED", True, (255, 255, 0))
        pause_rect = pause_text.get_rect(center=(width//2, 30))
        pause_bg = pygame.Surface((pause_rect.width + 20, pause_rect.height + 10), pygame.SRCALPHA)
        pause_bg.fill((0, 0, 0, 128))
        self.screen.blit(pause_bg, (pause_rect.x - 10, pause_rect.y - 5))
        self.screen.blit(pause_text, pause_rect)
    
    def _render_controls(self, width, height):
        """
        渲染控制按键提示
        
        参数:
            width: 屏幕宽度
            height: 屏幕高度
        """
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
        """
        渲染相机状态信息(缩放倍率和位置)
        
        参数:
            camera: 相机对象
            width: 屏幕宽度
        """
        info_text = self.small_font.render(
            f"Zoom: {camera.zoom:.2f}x  Position: ({camera.x:.0f}, {camera.y:.0f})", 
            True, (150, 150, 150)
        )
        self.screen.blit(info_text, (width - 250, 10))
    
    def _render_help(self, width, height):
        """
        渲染帮助菜单
        
        参数:
            width: 屏幕宽度
            height: 屏幕高度
        """
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

    def _has_camera_view_changed(self, camera):
        """
        检查相机视角是否发生变化
        
        参数:
            camera: 相机对象
        
        返回:
            bool: 视角是否变化
        """
        current_state = (camera.x, camera.y, camera.zoom)
        if current_state != self.last_camera_state:
            self.last_camera_state = current_state
            return True
        return False

    def _render_potential_heatmap(self, road, camera, target_surface):
        """
        生成并渲染势场热力图(计算密集型操作)
        
        参数:
            road: 道路对象，提供势场数据
            camera: 相机对象，用于坐标转换
            target_surface: 目标绘制表面
        """
        print("Generating potential field heatmap...")
        
        width, height = target_surface.get_size()
        grid_resolution = 20 # 分辨率，值越小越精细但越慢
        
        # 先清空旧的热力图
        target_surface.fill((0, 0, 0, 0))

        for y_screen in range(0, height, grid_resolution):
            for x_screen in range(0, width, grid_resolution):
                world_x, world_y = camera.screen_to_world(x_screen, y_screen)
                potential = road.get_potential_at_point(world_x, world_y)
                norm_potential = min(potential / 2.0, 1.0) # 归一化到 [0, 1]
                
                red = int(255 * norm_potential)
                green = int(255 * (1 - norm_potential))
                blue = 0
                alpha = 90
                
                rect = pygame.Rect(x_screen, y_screen, grid_resolution, grid_resolution)
                target_surface.fill((red, green, blue, alpha), rect)
        
        print("Heatmap generation complete.")

class GameEngine:
    """
    游戏引擎主类，协调相机、输入处理和渲染系统，管理主循环
    
    属性:
        screen: pygame屏幕对象
        clock: 用于控制帧率
        fps: 目标帧率
        running: 引擎运行状态标志
        camera: 相机系统
        input_handler: 输入处理器
        renderer: 渲染器
    """
    def __init__(self, width=800, height=800, title="RL Intersection Simulation"):
        """
        初始化游戏引擎
        
        参数:
            width: 窗口宽度，默认800像素
            height: 窗口高度，默认800像素
            title: 窗口标题
        """
        pygame.init()
        self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
        pygame.display.set_caption(title)
        
        self.clock = pygame.time.Clock()
        self.fps = 60
        self.running = True
        
        self.camera = Camera(width, height)
        self.input_handler = InputHandler()
        self.renderer = Renderer(self.screen)
    
    def handle_events(self, env):
        """
        处理所有事件
        
        参数:
            env: 环境对象，用于访问traffic_manager等组件
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            # 将env.traffic_manager传递给输入处理器
            elif self.input_handler.handle_event(event, self.camera, env.traffic_manager):
                self.running = False
    
    def render(self, env):
        """
        渲染一帧画面
        
        参数:
            env: 环境对象，提供road和traffic_manager等绘制所需的组件
        """
        # 从env获取road和traffic_manager
        road = env.road
        traffic_manager = env.traffic_manager
        self.renderer.render_frame(road, traffic_manager, self.camera, self.input_handler)
        pygame.display.flip()
    
    def is_running(self):
        """
        检查引擎是否仍在运行
        
        返回:
            bool: 引擎运行状态
        """
        return self.running
    
    def tick(self):
        """控制帧率，限制循环速度"""
        self.clock.tick(self.fps)
    
    def quit(self):
        """清理资源并退出Pygame"""
        pygame.quit()
import pygame
import math
import numpy as np

class Road:
    def __init__(self, width=200, height=200, lane_width=4):
        self.width = width
        self.height = height
        self.lane_width = lane_width
        self.center_x = width // 2
        self.center_y = height // 2
        
        # 转弯半径（左转和右转不同）
        self.left_turn_radius = lane_width * 2.5  # 左转半径较大
        self.right_turn_radius = lane_width * 1.5  # 右转半径较小




    def draw_road_lines(self, surface, transform_func=None):
        """绘制道路标线
        
        Args:
            surface: 要绘制的表面
            transform_func: 坐标转换函数，接收 (x, y) 返回转换后的 (x, y)
        """
        # 如果未提供转换函数，则使用恒等映射
        if transform_func is None:
            transform_func = lambda x, y: (x, y)
            
        # 白色实线（双向分隔线）
        # 水平道路中央分隔线
        start_pos = transform_func(0, self.center_y)
        end_pos = transform_func(self.center_x - 2 * self.lane_width, self.center_y)
        pygame.draw.line(surface, (255, 255, 255), start_pos, end_pos, 3)
        
        start_pos = transform_func(self.center_x + 2 * self.lane_width, self.center_y)
        end_pos = transform_func(self.width, self.center_y)
        pygame.draw.line(surface, (255, 255, 255), start_pos, end_pos, 3)
        
        # 垂直道路中央分隔线
        start_pos = transform_func(self.center_x, 0)
        end_pos = transform_func(self.center_x, self.center_y - 2 * self.lane_width)
        pygame.draw.line(surface, (255, 255, 255), start_pos, end_pos, 3)
        
        start_pos = transform_func(self.center_x, self.center_y + 2 * self.lane_width)
        end_pos = transform_func(self.center_x, self.height)
        pygame.draw.line(surface, (255, 255, 255), start_pos, end_pos, 3)
        
        # 车道边界线（带圆弧缓冲）
        corner_radius = self.lane_width   # 圆弧半径
        
        # 水平道路边界（上边界）
        start_pos = transform_func(0, self.center_y - self.lane_width)
        end_pos = transform_func(self.center_x - self.lane_width - corner_radius, self.center_y - self.lane_width)
        pygame.draw.line(surface, (255, 255, 255), start_pos, end_pos, 2)
        
        start_pos = transform_func(self.center_x + self.lane_width + corner_radius, self.center_y - self.lane_width)
        end_pos = transform_func(self.width, self.center_y - self.lane_width)
        pygame.draw.line(surface, (255, 255, 255), start_pos, end_pos, 2)
        
        # 水平道路边界（下边界）
        start_pos = transform_func(0, self.center_y + self.lane_width)
        end_pos = transform_func(self.center_x - self.lane_width - corner_radius, self.center_y + self.lane_width)
        pygame.draw.line(surface, (255, 255, 255), start_pos, end_pos, 2)
        
        start_pos = transform_func(self.center_x + self.lane_width + corner_radius, self.center_y + self.lane_width)
        end_pos = transform_func(self.width, self.center_y + self.lane_width)
        pygame.draw.line(surface, (255, 255, 255), start_pos, end_pos, 2)
        
        # 垂直道路边界（左边界）
        start_pos = transform_func(self.center_x - self.lane_width, 0)
        end_pos = transform_func(self.center_x - self.lane_width, self.center_y - self.lane_width - corner_radius)
        pygame.draw.line(surface, (255, 255, 255), start_pos, end_pos, 2)
        
        start_pos = transform_func(self.center_x - self.lane_width, self.center_y + self.lane_width + corner_radius)
        end_pos = transform_func(self.center_x - self.lane_width, self.height)
        pygame.draw.line(surface, (255, 255, 255), start_pos, end_pos, 2)
        
        # 垂直道路边界（右边界）
        start_pos = transform_func(self.center_x + self.lane_width, 0)
        end_pos = transform_func(self.center_x + self.lane_width, self.center_y - self.lane_width - corner_radius)
        pygame.draw.line(surface, (255, 255, 255), start_pos, end_pos, 2)
        
        start_pos = transform_func(self.center_x + self.lane_width, self.center_y + self.lane_width + corner_radius)
        end_pos = transform_func(self.center_x + self.lane_width, self.height)
        pygame.draw.line(surface, (255, 255, 255), start_pos, end_pos, 2)
        
        # 绘制圆弧缓冲
        self.draw_corner_arcs(surface, corner_radius, transform_func)
    
    def draw_corner_arcs(self, surface, radius, transform_func=None):
        """绘制道路边界的圆弧缓冲
        
        Args:
            surface: 要绘制的表面
            radius: 圆弧半径
            transform_func: 坐标转换函数
        """
        if transform_func is None:
            transform_func = lambda x, y: (x, y)
            
        # 左上角
        start_angle = 0
        end_angle = math.pi * 0.5
        center_x = self.center_x - self.lane_width - radius
        center_y = self.center_y - self.lane_width - radius
        self.draw_arc(surface, center_x, center_y, radius, start_angle, end_angle, transform_func=transform_func)
        
        # 右上角
        start_angle = math.pi * 0.5
        end_angle = math.pi
        center_x = self.center_x + self.lane_width + radius
        center_y = self.center_y - self.lane_width - radius
        self.draw_arc(surface, center_x, center_y, radius, start_angle, end_angle, transform_func=transform_func)
        
        # 右下角
        start_angle = math.pi
        end_angle = math.pi * 1.5
        center_x = self.center_x + self.lane_width + radius
        center_y = self.center_y + self.lane_width + radius
        self.draw_arc(surface, center_x, center_y, radius, start_angle, end_angle, transform_func=transform_func)
        
        # 左下角
        start_angle = math.pi * 1.5
        end_angle = math.pi * 2
        center_x = self.center_x - self.lane_width - radius
        center_y = self.center_y + self.lane_width + radius
        self.draw_arc(surface, center_x, center_y, radius, start_angle, end_angle, transform_func=transform_func)
    
    def draw_arc(self, surface, center_x, center_y, radius, start_angle, end_angle, color=(255, 255, 255), transform_func=None):
        """绘制边界圆弧
        
        Args:
            surface: 要绘制的表面
            center_x, center_y: 圆心坐标
            radius: 半径
            start_angle, end_angle: 起始和结束角度
            color: 颜色
            transform_func: 坐标转换函数
        """
        if transform_func is None:
            transform_func = lambda x, y: (x, y)
            
        # 创建临时surface (如果需要透明)
        temp_surface = None
        if len(color) > 3:  # 颜色包含alpha通道
            temp_surface = pygame.Surface((surface.get_width(), surface.get_height()), pygame.SRCALPHA)
            drawing_surface = temp_surface
        else:
            drawing_surface = surface
        
        points = []
        num_points = 30
        
        # 确定角度方向
        if start_angle > end_angle and abs(start_angle - end_angle) < math.pi:
            # 顺时针
            step = (end_angle - start_angle) / num_points
        elif start_angle < end_angle and abs(start_angle - end_angle) > math.pi:
            # 顺时针（跨越0/2pi边界）
            start_angle += 2 * math.pi
            step = (end_angle - start_angle) / num_points
        else:
            # 逆时针
            step = (end_angle - start_angle) / num_points
        
        # 生成圆弧点
        for i in range(num_points + 1):
            angle = start_angle + step * i
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            # 应用坐标转换
            x, y = transform_func(x, y)
            points.append((int(x), int(y)))
        
        if len(points) > 1:
            pygame.draw.lines(drawing_surface, color, False, points, 2)
        
        # 如果使用了临时surface，则将其绘制到主surface
        if temp_surface:
            surface.blit(temp_surface, (0, 0))
    
    def draw_center_lines(self, surface, alpha=128, transform_func=None):  # 添加transform_func参数
        """绘制车道中线（包含转弯圆弧）
        
        Args:
            surface: 要绘制的表面
            alpha: 透明度
            transform_func: 坐标转换函数
        """
        if transform_func is None:
            transform_func = lambda x, y: (x, y)
            
        # 创建透明临时surface
        temp_surface = pygame.Surface((surface.get_width(), surface.get_height()), pygame.SRCALPHA)
        
        # 黄色半透明
        yellow = (255, 255, 0, alpha)
        red = (255, 0, 0, alpha)
        
        # 直行车道中线
        # 水平道路中线（向右行驶）
        # 入口道
        start_pos = transform_func(0, self.center_y + self.lane_width//2)
        end_pos = transform_func(self.center_x - 2 * self.lane_width, self.center_y + self.lane_width//2)
        pygame.draw.line(temp_surface, yellow, start_pos, end_pos, 2)
        
        # 出口道
        start_pos = transform_func(self.center_x + 2 * self.lane_width, self.center_y + self.lane_width//2)
        end_pos = transform_func(self.width, self.center_y + self.lane_width//2)
        pygame.draw.line(temp_surface, yellow, start_pos, end_pos, 2)
        
        # 直行
        start_pos = transform_func(self.center_x - 2 * self.lane_width, self.center_y + self.lane_width//2)
        end_pos = transform_func(self.center_x + 2 * self.lane_width, self.center_y + self.lane_width//2)
        pygame.draw.line(temp_surface, yellow, start_pos, end_pos, 2)
        
        # 水平道路中线（向左行驶）
        # 入口道
        start_pos = transform_func(self.center_x + 2 * self.lane_width, self.center_y - self.lane_width//2)
        end_pos = transform_func(self.width, self.center_y - self.lane_width//2)
        pygame.draw.line(temp_surface, yellow, start_pos, end_pos, 2)

        # 出口道
        start_pos = transform_func(0, self.center_y - self.lane_width//2)
        end_pos = transform_func(self.center_x - 2 * self.lane_width, self.center_y - self.lane_width//2)
        pygame.draw.line(temp_surface, yellow, start_pos, end_pos, 2)
        
        # 直行
        start_pos = transform_func(self.center_x - 2 * self.lane_width, self.center_y - self.lane_width//2)
        end_pos = transform_func(self.center_x + 2 * self.lane_width, self.center_y - self.lane_width//2)
        pygame.draw.line(temp_surface, yellow, start_pos, end_pos, 2)
        
        # 垂直道路中线（向下行驶）
        # 入口道
        start_pos = transform_func(self.center_x - self.lane_width//2, 0)
        end_pos = transform_func(self.center_x - self.lane_width//2, self.center_y - 2 * self.lane_width)
        pygame.draw.line(temp_surface, yellow, start_pos, end_pos, 2)
        
        # 出口道
        start_pos = transform_func(self.center_x - self.lane_width//2, self.center_y + 2 * self.lane_width)
        end_pos = transform_func(self.center_x - self.lane_width//2, self.height)
        pygame.draw.line(temp_surface, yellow, start_pos, end_pos, 2)
        
        # 直行
        start_pos = transform_func(self.center_x - self.lane_width//2, self.center_y - 2 * self.lane_width)
        end_pos = transform_func(self.center_x - self.lane_width//2, self.center_y + 2 * self.lane_width)
        pygame.draw.line(temp_surface, yellow, start_pos, end_pos, 2)
        
        # 垂直道路中线（向上行驶）
        # 入口道
        start_pos = transform_func(self.center_x + self.lane_width//2, self.center_y + 2 * self.lane_width)
        end_pos = transform_func(self.center_x + self.lane_width//2, self.height)
        pygame.draw.line(temp_surface, yellow, start_pos, end_pos, 2)

        # 出口道
        start_pos = transform_func(self.center_x + self.lane_width//2, 0)
        end_pos = transform_func(self.center_x + self.lane_width//2, self.center_y - 2 * self.lane_width)
        pygame.draw.line(temp_surface, yellow, start_pos, end_pos, 2)
        
        # 直行
        start_pos = transform_func(self.center_x + self.lane_width//2, self.center_y - 2 * self.lane_width)
        end_pos = transform_func(self.center_x + self.lane_width//2, self.center_y + 2 * self.lane_width)
        pygame.draw.line(temp_surface, yellow, start_pos, end_pos, 2)
        
        # 将临时surface绘制到主surface
        surface.blit(temp_surface, (0, 0))
        
        # 绘制转弯圆弧
        self.draw_turn_arcs(surface, alpha, transform_func)

    def draw_turn_arcs(self, surface, alpha=128, transform_func=None):
        """绘制转弯圆弧连接线（使用圆心和角度定义方式）
        
        Args:
            surface: 要绘制的表面
            alpha: 透明度
            transform_func: 坐标转换函数
        """
        if transform_func is None:
            transform_func = lambda x, y: (x, y)
            
        yellow = (255, 255, 0, alpha)
        red = (255, 0, 0, alpha)
        
        # 从南向东（右转）
        center_x = self.center_x + self.lane_width + self.lane_width
        center_y = self.center_y + self.lane_width + self.lane_width
        self.draw_arc(surface, center_x, center_y, self.right_turn_radius, 
                      math.pi, math.pi * 1.5, yellow, transform_func)
        
        # 从南向西（左转）
        center_x = self.center_x + self.lane_width//2 - self.left_turn_radius 
        center_y = self.center_y - self.lane_width//2 + self.left_turn_radius 
        self.draw_arc(surface, center_x, center_y, self.left_turn_radius, 
                      0, -math.pi * 0.5, yellow, transform_func)
        
        # 从北向西（右转）
        center_x = self.center_x - self.lane_width - self.lane_width
        center_y = self.center_y - self.lane_width - self.lane_width
        self.draw_arc(surface, center_x, center_y, self.right_turn_radius, 
                      0, math.pi * 0.5, yellow, transform_func)
        
        # 从北向东（左转）
        center_x = self.center_x - self.lane_width//2 + self.left_turn_radius
        center_y = self.center_y + self.lane_width//2 - self.left_turn_radius 
        self.draw_arc(surface, center_x, center_y, self.left_turn_radius, 
                      math.pi * 0.5, math.pi, yellow, transform_func)
        
        # 从东向北（右转）
        center_x = self.center_x - self.lane_width//2 + self.left_turn_radius
        center_y = self.center_y + self.lane_width//2 - self.left_turn_radius
        self.draw_arc(surface, center_x, center_y, self.right_turn_radius, 
                      math.pi * 0.5, math.pi, yellow, transform_func)
        
        # 从东向南（左转）
        center_x = self.center_x + self.lane_width + self.lane_width
        center_y = self.center_y + self.lane_width + self.lane_width
        self.draw_arc(surface, center_x, center_y, self.left_turn_radius, 
                      math.pi, math.pi * 1.5, yellow, transform_func)
        
        # 从西向南（右转）
        center_x = self.center_x - self.lane_width - self.lane_width
        center_y = self.center_y + self.lane_width + self.lane_width
        self.draw_arc(surface, center_x, center_y, self.right_turn_radius, 
                      math.pi * 1.5, math.pi * 2, yellow, transform_func)
        
        # 从西向北（左转）
        center_x = self.center_x - self.lane_width - self.lane_width
        center_y = self.center_y - self.lane_width - self.lane_width
        self.draw_arc(surface, center_x, center_y, self.left_turn_radius, 
                      0, math.pi * 0.5, yellow, transform_func)

    def generate_centerline_points(self, segment_length=5, straight_segment_length=None):
        """生成中心线的点序列，用于车辆追踪
        
        Args:
            segment_length: 每段的长度（像素）
            
        Returns:
            dict: 包含各个方向和转弯的点序列
        """

        # 直线路段使用更大的间距
        if straight_segment_length is None:
            straight_segment_length = segment_length * 7  # 直线间距是转弯间距的4倍

        points = {
            'horizontal_right': [],  # 水平向右行驶
            'horizontal_left': [],   # 水平向左行驶
            'vertical_down': [],     # 垂直向下行驶
            'vertical_up': [],       # 垂直向上行驶
            'turn_south_to_east': [],  # 南向东右转
            'turn_south_to_west': [],  # 南向西左转
            'turn_north_to_west': [],  # 北向西右转
            'turn_north_to_east': [],  # 北向东左转
            'turn_east_to_north': [],  # 东向北右转
            'turn_east_to_south': [],  # 东向南左转
            'turn_west_to_south': [],  # 西向南右转
            'turn_west_to_north': []   # 西向北左转
        }
        
        # 生成直行道路中心线点
        # 水平向右行驶中心线
        y = self.center_y + self.lane_width // 2
        for x in range(0, self.width, straight_segment_length):
            points['horizontal_right'].append((x, y))
        
        # 水平向左行驶中心线
        y = self.center_y - self.lane_width // 2
        for x in range(self.width, 0, -straight_segment_length):
            points['horizontal_left'].append((x, y))
        
        # 垂直向下行驶中心线
        x = self.center_x - self.lane_width // 2
        for y in range(0, self.height, straight_segment_length):
            points['vertical_down'].append((x, y))
        
        # 垂直向上行驶中心线
        x = self.center_x + self.lane_width // 2
        for y in range(self.height, 0, -straight_segment_length):
            points['vertical_up'].append((x, y))
        
        # 生成转弯圆弧中心线点
        num_arc_points = 19
        
        # 南向东右转 (包含进口道、转弯圆弧、出口道)
        route_points = []
        # 进口道 - 从交叉口边界到转弯起点
        start_y = self.height
        end_y = self.center_y + 2 * self.lane_width
        x = self.center_x + self.lane_width // 2
        for y in range(start_y, end_y, -straight_segment_length):
            route_points.append((x, y))
        
        # 转弯圆弧
        center_x = self.center_x + self.lane_width + self.lane_width
        center_y = self.center_y + self.lane_width + self.lane_width
        for i in range(num_arc_points + 1):
            angle = math.pi + i * (math.pi * 0.5) / num_arc_points
            x = center_x + self.right_turn_radius * math.cos(angle)
            y = center_y + self.right_turn_radius * math.sin(angle)
            route_points.append((int(x), int(y)))
        
        # 出口道 - 从转弯终点到交叉口边界
        start_x = self.center_x + 2 * self.lane_width
        end_x = self.width
        y = self.center_y + self.lane_width // 2
        for x in range(start_x, end_x, straight_segment_length):
            route_points.append((x, y))
        points['turn_south_to_east'] = route_points
        
        # 南向西左转
        route_points = []
        # 进口道
        start_y = self.height
        end_y = self.center_y + 2 * self.lane_width
        x = self.center_x + self.lane_width // 2
        for y in range(start_y, end_y, -straight_segment_length):
            route_points.append((x, y))
        
        # 转弯圆弧
        center_x = self.center_x + self.lane_width//2 - self.left_turn_radius
        center_y = self.center_y - self.lane_width//2 + self.left_turn_radius
        for i in range(num_arc_points + 1):
            angle = 0 - i * (math.pi * 0.5) / num_arc_points
            x = center_x + self.left_turn_radius * math.cos(angle)
            y = center_y + self.left_turn_radius * math.sin(angle)
            route_points.append((int(x), int(y)))
        
        # 出口道
        start_x = self.center_x - 2 * self.lane_width
        end_x = 0
        y = self.center_y - self.lane_width // 2
        for x in range(start_x, end_x, -straight_segment_length):
            route_points.append((x, y))
        points['turn_south_to_west'] = route_points
        
        # 北向西右转
        route_points = []
        # 进口道
        start_y = 0
        end_y = self.center_y - 2 * self.lane_width
        x = self.center_x - self.lane_width // 2
        for y in range(start_y, end_y, straight_segment_length):
            route_points.append((x, y))
        
        # 转弯圆弧
        center_x = self.center_x - self.lane_width - self.lane_width
        center_y = self.center_y - self.lane_width - self.lane_width
        for i in range(num_arc_points + 1):
            angle = 0 + i * (math.pi * 0.5) / num_arc_points
            x = center_x + self.right_turn_radius * math.cos(angle)
            y = center_y + self.right_turn_radius * math.sin(angle)
            route_points.append((int(x), int(y)))
        
        # 出口道
        start_x = self.center_x - 2 * self.lane_width
        end_x = 0
        y = self.center_y - self.lane_width // 2
        for x in range(start_x, end_x, -straight_segment_length):
            route_points.append((x, y))
        points['turn_north_to_west'] = route_points
        
        # 北向东左转
        route_points = []
        # 进口道
        start_y = 0
        end_y = self.center_y - 2 * self.lane_width
        x = self.center_x - self.lane_width // 2
        for y in range(start_y, end_y, straight_segment_length):
            route_points.append((x, y))
        
        # 转弯圆弧
        center_x = self.center_x - self.lane_width//2 + self.left_turn_radius
        center_y = self.center_y + self.lane_width//2 - self.left_turn_radius
        for i in range(num_arc_points + 1):
            angle = math.pi - i * (math.pi * 0.5) / num_arc_points
            x = center_x + self.left_turn_radius * math.cos(angle)
            y = center_y + self.left_turn_radius * math.sin(angle)
            route_points.append((int(x), int(y)))
        
        # 出口道
        start_x = self.center_x + 2 * self.lane_width
        end_x = self.width
        y = self.center_y + self.lane_width // 2
        for x in range(start_x, end_x, straight_segment_length):
            route_points.append((x, y))
        points['turn_north_to_east'] = route_points
        
        # 东向北右转
        route_points = []
        # 进口道
        start_x = self.width
        end_x = self.center_x + 2 * self.lane_width
        y = self.center_y - self.lane_width // 2
        for x in range(start_x, end_x, -straight_segment_length):
            route_points.append((x, y))
        
        # 转弯圆弧
        center_x = self.center_x - self.lane_width//2 + self.left_turn_radius
        center_y = self.center_y + self.lane_width//2 - self.left_turn_radius
        for i in range(num_arc_points + 1):
            angle = math.pi * 0.5 + i * (math.pi * 0.5) / num_arc_points
            x = center_x + self.right_turn_radius * math.cos(angle)
            y = center_y + self.right_turn_radius * math.sin(angle)
            route_points.append((int(x), int(y)))
        
        # 出口道
        start_y = self.center_y - 2 * self.lane_width
        end_y = 0
        x = self.center_x + self.lane_width // 2
        for y in range(start_y, end_y, -straight_segment_length):
            route_points.append((x, y))
        points['turn_east_to_north'] = route_points
        
        # 东向南左转
        route_points = []
        # 进口道
        start_x = self.width
        end_x = self.center_x + 2 * self.lane_width
        y = self.center_y - self.lane_width // 2
        for x in range(start_x, end_x, -straight_segment_length):
            route_points.append((x, y))
        
        # 转弯圆弧
        center_x = self.center_x + self.lane_width + self.lane_width
        center_y = self.center_y + self.lane_width + self.lane_width
        for i in range(num_arc_points + 1):
            angle = math.pi * 1.5 - i * (math.pi * 0.5) / num_arc_points
            x = center_x + self.left_turn_radius * math.cos(angle)
            y = center_y + self.left_turn_radius * math.sin(angle)
            route_points.append((int(x), int(y)))
        
        # 出口道
        start_y = self.center_y + 2 * self.lane_width
        end_y = self.height
        x = self.center_x - self.lane_width // 2
        for y in range(start_y, end_y, straight_segment_length):
            route_points.append((x, y))
        points['turn_east_to_south'] = route_points
        
        # 西向南右转
        route_points = []
        # 进口道
        start_x = 0
        end_x = self.center_x - 2 * self.lane_width
        y = self.center_y + self.lane_width // 2
        for x in range(start_x, end_x, straight_segment_length):
            route_points.append((x, y))
        
        # 转弯圆弧
        center_x = self.center_x - self.lane_width - self.lane_width
        center_y = self.center_y + self.lane_width + self.lane_width
        for i in range(num_arc_points + 1):
            angle = math.pi * 1.5 + i * (math.pi * 0.5) / num_arc_points
            x = center_x + self.right_turn_radius * math.cos(angle)
            y = center_y + self.right_turn_radius * math.sin(angle)
            route_points.append((int(x), int(y)))
        
        # 出口道
        start_y = self.center_y + 2 * self.lane_width
        end_y = self.height
        x = self.center_x - self.lane_width // 2
        for y in range(start_y, end_y, straight_segment_length):
            route_points.append((x, y))
        points['turn_west_to_south'] = route_points
        
        # 西向北左转
        route_points = []
        # 进口道
        start_x = 0
        end_x = self.center_x - 2 * self.lane_width
        y = self.center_y + self.lane_width // 2
        for x in range(start_x, end_x, straight_segment_length):
            route_points.append((x, y))
        
        # 转弯圆弧
        center_x = self.center_x - self.lane_width - self.lane_width
        center_y = self.center_y - self.lane_width - self.lane_width
        for i in range(num_arc_points + 1):
            angle = math.pi * 0.5 - i * (math.pi * 0.5) / num_arc_points
            x = center_x + self.left_turn_radius * math.cos(angle)
            y = center_y + self.left_turn_radius * math.sin(angle)
            route_points.append((int(x), int(y)))
        
        # 出口道
        start_y = self.center_y - 2 * self.lane_width
        end_y = 0
        x = self.center_x + self.lane_width // 2
        for y in range(start_y, end_y, -straight_segment_length):
            route_points.append((x, y))
        points['turn_west_to_north'] = route_points
        
        return points

    def get_route_points(self, start_direction, end_direction, segment_length=5, straight_segment_length=None):
        """获取从起始方向到结束方向的完整路径点序列
        
        Args:
            start_direction: 起始方向 ('north', 'south', 'east', 'west')
            end_direction: 结束方向 ('north', 'south', 'east', 'west')
            segment_length: 转弯时每段的长度（像素）
            straight_segment_length: 直线时每段的长度（像素），如果为None则使用segment_length的4倍
            
        Returns:
            list: 完整路径的点序列
        """
        # 使用变密度生成路径点
        centerlines = self.generate_centerline_points(segment_length, straight_segment_length)
        route_points = []
        
        # 定义路径映射
        if start_direction == 'south':
            if end_direction == 'north':  # 直行
                route_points = centerlines['vertical_up']
            elif end_direction == 'east':  # 右转
                route_points = centerlines['turn_south_to_east']
            elif end_direction == 'west':  # 左转
                route_points = centerlines['turn_south_to_west']
        
        elif start_direction == 'north':
            if end_direction == 'south':  # 直行
                route_points = centerlines['vertical_down']
            elif end_direction == 'west':  # 右转
                route_points = centerlines['turn_north_to_west']
            elif end_direction == 'east':  # 左转
                route_points = centerlines['turn_north_to_east']
        
        elif start_direction == 'east':
            if end_direction == 'west':  # 直行
                route_points = centerlines['horizontal_left']
            elif end_direction == 'north':  # 右转
                route_points = centerlines['turn_east_to_north']
            elif end_direction == 'south':  # 左转
                route_points = centerlines['turn_east_to_south']
        
        elif start_direction == 'west':
            if end_direction == 'east':  # 直行
                route_points = centerlines['horizontal_right']
            elif end_direction == 'south':  # 右转
                route_points = centerlines['turn_west_to_south']
            elif end_direction == 'north':  # 左转
                route_points = centerlines['turn_west_to_north']

        
        # Remove duplicate consecutive points
        filtered_points = self._remove_duplicate_points(route_points)
        
        return filtered_points

    def _remove_duplicate_points(self, points, tolerance=1):
        """Remove consecutive duplicate points from a list of points
        
        Args:
            points: List of (x, y) tuples
            tolerance: Minimum distance between points to be considered different
            
        Returns:
            list: Points with consecutive duplicates removed
        """
        if not points:
            return points
        
        filtered_points = [points[0]]  # Always keep the first point
        
        for point in points[1:]:
            last_point = filtered_points[-1]
            # Calculate distance between current point and last kept point
            distance = math.sqrt((point[0] - last_point[0])**2 + (point[1] - last_point[1])**2)
            
            # Only add point if it's far enough from the last point
            if distance > tolerance:
                filtered_points.append(point)
        
        return filtered_points

    def visualize_centerline_points(self, surface, point_radius=2, show_all=True, route_key=None, transform_func=None):
        """可视化中心线点序列
        
        Args:
            surface: pygame surface对象
            point_radius: 点的半径
            show_all: 是否显示所有路径的点
            route_key: 如果指定，只显示特定路径的点
            transform_func: 坐标转换函数
        """
        if transform_func is None:
            transform_func = lambda x, y: (x, y)
            
        points_dict = self.generate_centerline_points()
        
        # 定义颜色
        colors = {
            'horizontal_right': (255, 0, 0),    # 红色
            'horizontal_left': (0, 255, 0),     # 绿色
            'vertical_down': (0, 0, 255),       # 蓝色
            'vertical_up': (255, 255, 0),       # 黄色
            'turn_south_to_east': (255, 0, 255), # 紫色
            'turn_south_to_west': (0, 255, 255), # 青色
            'turn_north_to_west': (255, 128, 0), # 橙色
            'turn_north_to_east': (128, 255, 0), # 黄绿色
            'turn_east_to_north': (128, 0, 255), # 紫蓝色
            'turn_east_to_south': (255, 128, 128), # 粉红色
            'turn_west_to_south': (128, 255, 128), # 浅绿色
            'turn_west_to_north': (128, 128, 255)  # 浅蓝色
        }
        
        if show_all:
            # 显示所有路径的点
            for key, points in points_dict.items():
                color = colors.get(key, (255, 255, 255))
                for i, point in enumerate(points):
                    if i % 3 == 0:  # 稀疏显示点，避免太密集
                        # 应用坐标转换
                        screen_x, screen_y = transform_func(point[0], point[1])
                        pygame.draw.circle(surface, color, (screen_x, screen_y), point_radius)
        elif route_key and route_key in points_dict:
            # 只显示指定路径的点
            color = colors.get(route_key, (255, 255, 255))
            for point in points_dict[route_key]:
                # 应用坐标转换
                screen_x, screen_y = transform_func(point[0], point[1])
                pygame.draw.circle(surface, color, (screen_x, screen_y), point_radius)

    def visualize_route_points(self, surface, start_direction, end_direction, point_radius=1, color=(255, 0, 0), transform_func=None):
        """可视化特定路径的点序列
        
        Args:
            surface: pygame surface对象
            start_direction: 起始方向
            end_direction: 结束方向
            point_radius: 点的半径
            color: 点的颜色
            transform_func: 坐标转换函数
        """
        if transform_func is None:
            transform_func = lambda x, y: (x, y)
            
        route_points = self.get_route_points(start_direction, end_direction)
        
        for i, point in enumerate(route_points):
            # 使用渐变颜色表示方向
            alpha = min(255, 100 + i * 2)
            current_color = (*color, alpha) if len(color) == 3 else color
            
            # 应用坐标转换
            screen_x, screen_y = transform_func(point[0], point[1])
            
            # 创建临时surface来支持alpha混合
            temp_surface = pygame.Surface((point_radius * 2, point_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surface, current_color, (point_radius, point_radius), point_radius)
            surface.blit(temp_surface, (screen_x - point_radius, screen_y - point_radius))


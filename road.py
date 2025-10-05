"""
@file: road.py
@description: 
此文件实现了一个四向交叉路口环境的静态几何结构、路径生成与冲突检测系统。

主要功能:
1. 交叉口几何定义 - 程序化生成标准四向交叉路口布局
2. 路径生成与处理 - 生成、平滑和缓存12种可能的行驶路线
3. 冲突检测 - 通过预定义的冲突矩阵判断路径冲突关系
4. 可视化渲染 - 提供一系列方法绘制道路元素到Pygame界面
5. 势场计算 - 为任意点计算势能值，用于车辆路径规划

主要类:
- Road: 作为仿真的"地图"，负责管理交叉路口的几何表示和渲染

主要函数:
- draw_*系列函数: 处理道路元素的可视化渲染
- generate_centerline_points: 生成原始路径点序列
- get_route_points: 获取平滑处理后的路径数据
- do_paths_conflict: 判断两条路径是否存在冲突
- get_potential_at_point: 计算点的势场值，用于导航
"""

import pygame
import math
import numpy as np
from path_smoother import smooth_path, resample_path, recalculate_angles
from config import *

CONFLICT_MATRIX = {
    'E_N': {'E_N': False, 'E_S': False, 'E_W': False, 'N_E': False, 'N_S': False, 'N_W': False, 'S_E': False, 'S_N': True, 'S_W': False, 'W_E': False, 'W_N': True, 'W_S': False},
    'E_S': {'E_N': False, 'E_S': False, 'E_W': False, 'N_E': True, 'N_S': True, 'N_W': False, 'S_E': False, 'S_N': True, 'S_W': True, 'W_E': True, 'W_N': False, 'W_S': False},
    'E_W': {'E_N': False, 'E_S': False, 'E_W': False, 'N_E': True, 'N_S': True, 'N_W': True, 'S_E': False, 'S_N': True, 'S_W': True, 'W_E': False, 'W_N': True, 'W_S': False},
    'N_E': {'E_N': False, 'E_S': True, 'E_W': True, 'N_E': False, 'N_S': False, 'N_W': False, 'S_E': True, 'S_N': True, 'S_W': False, 'W_E': True, 'W_N': True, 'W_S': False},
    'N_S': {'E_N': False, 'E_S': True, 'E_W': True, 'N_E': False, 'N_S': False, 'N_W': False, 'S_E': False, 'S_N': False, 'S_W': True, 'W_E': True, 'W_N': True, 'W_S': True},
    'N_W': {'E_N': False, 'E_S': False, 'E_W': True, 'N_E': False, 'N_S': False, 'N_W': True, 'S_E': False, 'S_N': False, 'S_W': False, 'W_E': False, 'W_N': False, 'W_S': False},
    'S_E': {'E_N': False, 'E_S': False, 'E_W': False, 'N_E': True, 'N_S': False, 'N_W': False, 'S_E': False, 'S_N': False, 'S_W': False, 'W_E': True, 'W_N': False, 'W_S': False},
    'S_N': {'E_N': True, 'E_S': True, 'E_W': True, 'N_E': False, 'N_S': False, 'N_W': True, 'S_E': False, 'S_N': False, 'S_W': False, 'W_E': True, 'W_N': True, 'W_S': False},
    'S_W': {'E_N': False, 'E_S': True, 'E_W': True, 'N_E': False, 'N_S': True, 'N_W': True, 'S_E': False, 'S_N': False, 'S_W': False, 'W_E': True, 'W_N': True, 'W_S': False},
    'W_E': {'E_N': False, 'E_S': True, 'E_W': False, 'N_E': True, 'N_S': True, 'N_W': False, 'S_E': True, 'S_N': True, 'S_W': True, 'W_E': False, 'W_N': False, 'W_S': False},
    'W_N': {'E_N': True, 'E_S': False, 'E_W': True, 'N_E': True, 'N_S': True, 'N_W': False, 'S_E': False, 'S_N': True, 'S_W': True, 'W_E': False, 'W_N': False, 'W_S': False},
    'W_S': {'E_N': False, 'E_S': True, 'E_W': False, 'N_E': False, 'N_S': True, 'N_W': False, 'S_E': False, 'S_N': False, 'S_W': False, 'W_E': False, 'W_N': False, 'W_S': False}
}
class Road:
    def __init__(self, width=400, height=400, lane_width=4):
        """
        初始化交叉路口环境对象
        
        Args:
            width (int): 场景总宽度(像素)
            height (int): 场景总高度(像素)
            lane_width (int): 车道宽度(像素)
            
        Notes:
            - 初始化过程会建立道路几何结构、冲突区域和势场计算参数
            - 自动预生成所有12种路径(N/S/E/W之间的组合)并缓存
        """
        self.width = width
        self.height = height
        self.lane_width = lane_width
        self.center_x = width // 2
        self.center_y = height // 2
        self.center = np.array([self.center_x, self.center_y])

        self.routes = {}
        
        # 转弯半径（左转和右转不同）
        self.left_turn_radius = lane_width * 2.5  # 左转半径较大
        self.right_turn_radius = lane_width * 1.5  # 右转半径较小

        self.conflict_zone_radius = 4 * self.lane_width
        self.conflict_zone = {
            'center': (self.center_x, self.center_y),
            'radius': self.conflict_zone_radius
        }

        safe_stopping_distance = 60
        self.extended_conflict_zone_radius = self.conflict_zone_radius + safe_stopping_distance
        self.extended_conflict_zone = {
            'center': (self.center_x, self.center_y),
            'radius': self.extended_conflict_zone_radius
        }

        self.conflict_matrix = CONFLICT_MATRIX

        # --- 势能函数参数（可调） ---
        self.eps = 0.05    # 远处平缓斜率
        self.A = 5.0       # 近处快速增长强度
        self.sigma = 0.2   # 快增区宽度（峰在 0.5）

        # --- 查表范围与密度设置 ---
        self.u_max = 3.0   # 只查表到 3，之后线性外推
        n0_1 = 2000        # [0,1] 高密度
        n1_3 = 1000        # (1,3] 次高密度

        # 非均匀 t 网格：先 [0,1]，再 (1,3]
        t0 = np.linspace(0.0, 1.0, n0_1, endpoint=True)
        t1 = np.linspace(1.0, self.u_max, n1_3, endpoint=True)[1:]  # 去掉重复的 1.0
        self.t_grid = np.concatenate([t0, t1])  # 递增且唯一

        # 预计算积分查表：I(u) = ∫_0^{u} (eps + A * exp(-((t-0.5)/sigma)^2)) dt
        integrand = self.eps + self.A * np.exp(-((self.t_grid - 0.5) / self.sigma) ** 2)
        # 用累积梯形积分；I(0)=0
        diffs = np.diff(self.t_grid)
        avg_vals = 0.5 * (integrand[:-1] + integrand[1:])
        self.I_grid = np.zeros_like(self.t_grid)
        self.I_grid[1:] = np.cumsum(diffs * avg_vals)

        self._pre_generate_all_routes()

    def _calculate_offset_points(self, points, offset_distance):
        """
        根据中心线点和偏移距离，计算平行线点(用于生成车道边界)
        
        Args:
            points (list): 中心线点列表，每个点为(x,y)坐标
            offset_distance (float): 偏移距离，正值向外偏移，负值向内偏移
            
        Returns:
            list: 平行偏移后的点列表
            
        Notes:
            使用垂直于路径方向的法向量计算偏移点
        """
        offset_points = []
        points_np = np.array(points, dtype=float)
        
        # 计算每段的方向向量
        vectors = np.diff(points_np, axis=0)
        # 计算法向量
        normals = np.array([-vectors[:, 1], vectors[:, 0]]).T
        # 归一化法向量
        norms = np.linalg.norm(normals, axis=1)
        normals /= norms[:, np.newaxis]
        
        # 为第一个点计算法线（使用第一段的法线）
        offset_points.append(points_np[0] + normals[0] * offset_distance)
        
        # 为中间点计算法线（使用前后两段法线的平均值，以平滑拐角）
        for i in range(len(normals) - 1):
            avg_normal = (normals[i] + normals[i+1]) / 2
            avg_normal /= np.linalg.norm(avg_normal)
            offset_points.append(points_np[i+1] + avg_normal * offset_distance)
            
        # 为最后一个点计算法线（使用最后一段的法线）
        offset_points.append(points_np[-1] + normals[-1] * offset_distance)
        
        return offset_points

    def _generate_and_process_boundaries(self, raw_centerline):
        """
        根据原始中心线，生成并处理左右边界
        
        Args:
            raw_centerline (list): 原始中心线点列表
            
        Returns:
            tuple: (左边界点列表, 右边界点列表)，经过平滑和重采样处理
            
        Notes:
            边界线会经过与中心线相同的平滑和重采样流程，确保平滑过渡
        """
        if len(raw_centerline) < 2:
            return [], []

        half_lane_width = self.lane_width / 2
        
        # 1. 计算原始边界点
        raw_left_boundary = self._calculate_offset_points(raw_centerline, half_lane_width)
        raw_right_boundary = self._calculate_offset_points(raw_centerline, -half_lane_width)
        
        # 2. 对边界进行与中心线相同的平滑和重采样处理
        smoothed_left = smooth_path(raw_left_boundary, alpha=0.3, beta=0.5, iterations=100)
        resampled_left = resample_path(smoothed_left, segment_length=0.5)
        
        smoothed_right = smooth_path(raw_right_boundary, alpha=0.3, beta=0.5, iterations=100)
        resampled_right = resample_path(smoothed_right, segment_length=0.5)
        
        return resampled_left, resampled_right

    def draw_boundaries(self, surface, transform_func=None):
        """
        绘制所有已生成路径的逻辑车道边界
        
        Args:
            surface (pygame.Surface): 要绘制的pygame表面
            transform_func (callable, optional): 坐标转换函数，默认为恒等映射
            
        Notes:
            主要用于调试和可视化，使用淡蓝色线条表示车道边界
        """
        if transform_func is None:
            transform_func = lambda x, y: (x, y)
        
        for route_key, route_data in self.routes.items():
            if "boundaries" in route_data:
                left_boundary = route_data["boundaries"]["left"]
                right_boundary = route_data["boundaries"]["right"]
                
                if len(left_boundary) > 1:
                    transformed_left = [transform_func(p[0], p[1]) for p in left_boundary]
                    pygame.draw.lines(surface, (150, 150, 255, 100), False, transformed_left, 5) #淡蓝色
                
                if len(right_boundary) > 1:
                    transformed_right = [transform_func(p[0], p[1]) for p in right_boundary]
                    pygame.draw.lines(surface, (150, 150, 255, 100), False, transformed_right, 5) #淡蓝色


    def draw_conflict_zones(self, surface, transform_func=None):
        """
        可视化交叉口的核心冲突区和扩展感知区
        
        Args:
            surface (pygame.Surface): 要绘制的pygame表面
            transform_func (callable, optional): 坐标转换函数，默认为恒等映射
            
        Notes:
            - 核心冲突区(红色半透明)：车辆实际可能相撞的区域
            - 扩展感知区(橙色边界)：车辆需要提前感知以决策的区域
        """
        if transform_func is None:
            transform_func = lambda x, y: (x, y)

        # 创建一个临时表面以支持半透明绘图
        temp_surface = pygame.Surface(surface.get_size(), pygame.SRCALPHA)

        # 转换中心点坐标
        screen_center = transform_func(*self.conflict_zone['center'])
        p1_screen = transform_func(0, 0)
        p2_screen = transform_func(1, 0)
        current_zoom = p2_screen[0] - p1_screen[0]
        # 1. 绘制扩展感知区 (橙色)
        extended_color_fill = (255, 165, 0, 30) # Semi-transparent orange
        extended_color_border = (255, 165, 0, 80)
        extended_radius_px = self.extended_conflict_zone['radius'] * current_zoom        
        pygame.draw.circle(temp_surface, extended_color_fill, screen_center, extended_radius_px)
        pygame.draw.circle(temp_surface, extended_color_border, screen_center, extended_radius_px, 2)

        # 2. 绘制核心冲突区 (红色)
        conflict_color_fill = (255, 0, 0, 60) # Semi-transparent red
        conflict_color_border = (255, 0, 0, 120)
        conflict_radius_px = self.conflict_zone['radius'] * current_zoom
        pygame.draw.circle(temp_surface, conflict_color_fill, screen_center, conflict_radius_px)
        pygame.draw.circle(temp_surface, conflict_color_border, screen_center, conflict_radius_px, 3)

        # 将带有透明区域的临时表面绘制到主屏幕上
        surface.blit(temp_surface, (0, 0))

    def draw_road_lines(self, surface, transform_func=None):
        """
        绘制道路标线(车道线、分隔线等)
        
        Args:
            surface (pygame.Surface): 要绘制的pygame表面
            transform_func (callable, optional): 坐标转换函数
            
        Notes:
            绘制双向分隔线(中央白实线)和车道边界线
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
        """
        绘制道路边界的四个圆弧缓冲(路口转角)
        
        Args:
            surface (pygame.Surface): 要绘制的pygame表面
            radius (float): 圆弧半径
            transform_func (callable, optional): 坐标转换函数
            
        Notes:
            在四个转角处绘制圆弧，使道路转角更自然
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
        """
        绘制边界圆弧
        
        Args:
            surface (pygame.Surface): 要绘制的pygame表面
            center_x, center_y (float): 圆心坐标
            radius (float): 圆弧半径
            start_angle, end_angle (float): 起始和结束角度(弧度)
            color (tuple): RGB或RGBA颜色元组
            transform_func (callable, optional): 坐标转换函数
            
        Notes:
            通过生成多个线段点来近似圆弧，支持透明度
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
    
    def draw_center_lines(self, surface, alpha=128, transform_func=None):
        """
        绘制车道中线(包含直行段和转弯段)
        
        Args:
            surface (pygame.Surface): 要绘制的pygame表面
            alpha (int): 透明度值(0-255)
            transform_func (callable, optional): 坐标转换函数
            
        Notes:
            使用黄色半透明线条表示车道中线，分别绘制直行段和转弯段
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
        """
        绘制转弯圆弧连接线(左转和右转的路径)
        
        Args:
            surface (pygame.Surface): 要绘制的pygame表面
            alpha (int): 透明度值(0-255)
            transform_func (callable, optional): 坐标转换函数
            
        Notes:
            使用不同半径的圆弧分别表示左转(大半径)和右转(小半径)的路径
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
        """
        生成所有可能行驶路线的原始中心线点序列
        
        Args:
            segment_length (int): 转弯段每段的采样长度(像素)
            straight_segment_length (int, optional): 直线段的采样长度，默认为转弯段的15倍
            
        Returns:
            dict: 包含各个方向和转弯路线的原始点序列
            
        Notes:
            生成12种可能路线的原始点，包括四个直行和八个转弯路线
            这些点将进一步被平滑和重采样处理
        """

        # 直线路段使用更大的间距
        if straight_segment_length is None:
            straight_segment_length = segment_length * 15  # 直线间距是转弯间距的4倍

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
        num_arc_points = 15
        
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

    def get_route_points(self, start_direction, end_direction, segment_length=3, straight_segment_length=None):
        """
        获取从起始方向到结束方向的完整路径点序列
        
        Args:
            start_direction (str): 起始方向 ('north', 'south', 'east', 'west')
            end_direction (str): 结束方向 ('north', 'south', 'east', 'west')
            segment_length (int): 转弯时每段的长度(像素)
            straight_segment_length (int, optional): 直线时每段的长度
            
        Returns:
            dict: 包含路径数据的字典，包括原始点、平滑点和边界数据
            
        Notes:
            1. 首先检查缓存是否存在该路径
            2. 如不存在，从原始点开始生成、平滑、重采样
            3. 计算路径的朝向角度和左右边界
            4. 将结果缓存以提高性能
        """
        move_str = f"{start_direction[0].upper()}_{end_direction[0].upper()}"
        
        # 如果已经计算过这条路径，直接返回缓存结果
        if move_str in self.routes:
            return self.routes[move_str]
        
        # 使用变密度生成路径点
        centerlines = self.generate_centerline_points(segment_length, straight_segment_length)
        route_points = []
        
        # 定义路径映射 
        move_key = f'turn_{start_direction}_to_{end_direction}'
        if start_direction == 'south' and end_direction == 'north': move_key = 'vertical_up'
        elif start_direction == 'north' and end_direction == 'south': move_key = 'vertical_down'
        elif start_direction == 'east' and end_direction == 'west': move_key = 'horizontal_left'
        elif start_direction == 'west' and end_direction == 'east': move_key = 'horizontal_right'

        route_points = centerlines.get(move_key, [])
        
        # 删除连续的重复点
        filtered_points = self._remove_duplicate_points(route_points)
        if len(filtered_points) < 3:
            return {} # 返回空字典表示失败

        xy_points = [[p[0], p[1]] for p in filtered_points]
        
        # --- 平滑+重采样流程 ---
        smoothed_xy = smooth_path(xy_points, alpha=0.3, beta=0.5, iterations=100)
        resampled_xy = resample_path(smoothed_xy, segment_length=0.5)
        final_points_with_angle = recalculate_angles(resampled_xy)

        left_boundary, right_boundary = self._generate_and_process_boundaries(filtered_points)


        result = {
            "raw": filtered_points,
            "smoothed": final_points_with_angle,
            "boundaries": {
                "left": left_boundary,
                "right": right_boundary
            }
        }
        
        # 缓存计算结果
        self.routes[move_str] = result
        
        return result

    def _pre_generate_all_routes(self):
        """
        在仿真开始前，预先生成并缓存所有可能的路径及其边界
        
        Notes:
            生成所有12种可能的行驶路线(四个方向的两两组合，去除相同起终点)
            预计算可以提高仿真性能，避免运行时的路径计算延迟
        """
        print("Pre-generating all 12 routes and their boundaries...")
        directions = ['north', 'south', 'east', 'west']
        for start in directions:
            for end in directions:
                if start == end:
                    continue
                self.get_route_points(start, end)
        print("All routes pre-generated and cached.")

    def _remove_duplicate_points(self, points, tolerance=1):
        """
        删除点列表中连续的重复点
        
        Args:
            points (list): (x, y)坐标点列表
            tolerance (float): 最小距离阈值，低于此值的点被视为重复点
            
        Returns:
            list: 删除重复点后的点列表
            
        Notes:
            用于优化路径点，减少不必要的计算和渲染
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

    def do_paths_conflict(self, move_str1, move_str2):
        """
        判断两条路径是否存在潜在冲突
        
        Args:
            move_str1 (str): 第一条路径标识符(如'N_S'表示从北到南)
            move_str2 (str): 第二条路径标识符
            
        Returns:
            bool: 如果两条路径存在潜在冲突返回True，否则返回False
            
        Notes:
            使用预计算的冲突矩阵进行O(1)复杂度的快速查询
        """
        # 使用.get()方法来安全地查询，如果键不存在则返回False
        return self.conflict_matrix.get(move_str1, {}).get(move_str2, False)

    def get_conflict_entry_index(self, move_str):
        """
        获取指定路径进入冲突区域的第一个点的索引
        
        Args:
            move_str (str): 路径标识符(如'N_S')
            
        Returns:
            int: 第一个进入冲突区域的点的索引，如果路径不存在或不经过冲突区，返回-1
            
        Notes:
            对于交通控制决策非常重要，可确定车辆何时需要决策避让
        """
        # 确保路径存在
        if not hasattr(self, 'routes') or move_str not in self.routes:
            return -1
                
        path_points = self.routes[move_str]["smoothed"]
        center_x, center_y = self.conflict_zone['center']
        radius_sq = self.conflict_zone['radius'] ** 2

        for i, point in enumerate(path_points):
            # 检查点是否在预定义的冲突区圆形内
            dist_sq = (point[0] - center_x)**2 + (point[1] - center_y)**2
            if dist_sq <= radius_sq:
                return i # 返回第一个冲突点的索引
        
        return -1 # 如果路径不经过冲突区，返回-1
    
    def visualize_conflict_entry_point(self, surface, start_direction, end_direction, color=(255, 0, 0), point_radius=5, transform_func=None):
        """
        可视化指定路径进入扩展冲突区域的第一个点
        
        Args:
            surface (pygame.Surface): pygame表面对象
            start_direction (str): 起始方向 ('north', 'south', 'east', 'west')
            end_direction (str): 结束方向
            color (tuple): 点的RGB颜色
            point_radius (int): 点的半径(像素)
            transform_func (callable, optional): 坐标转换函数
            
        Notes:
            用于调试和可视化，以红色圆点标记路径进入冲突区的位置
        """
        if transform_func is None:
            transform_func = lambda x, y: (x, y)
        
        # 获取路径点
        route_data = self.get_route_points(start_direction, end_direction)
        if not route_data or "smoothed" not in route_data:
            return  # 路径不存在
            
        path_points = route_data["smoothed"]
        
        # 查找进入冲突区的第一个点
        entry_index = -1
        center_x, center_y = self.extended_conflict_zone['center']
        radius_sq = self.extended_conflict_zone['radius'] ** 2
        for i, point in enumerate(path_points):
            dist_sq = (point[0] - center_x)**2 + (point[1] - center_y)**2
            if dist_sq <= radius_sq:
                entry_index = i
                break
                
        if entry_index == -1:
            return  # 没有点进入冲突区
            
        # 获取点坐标
        point = path_points[entry_index]
        print(point)
        x, y = point[0], point[1]
        
        # 应用坐标转换
        screen_x, screen_y = transform_func(x, y)
        
        # 绘制高亮点
        pygame.draw.circle(surface, color, (screen_x, screen_y), point_radius)
        
        # 添加一个环形标记使点更明显
        pygame.draw.circle(surface, (255, 255, 255), (screen_x, screen_y), point_radius + 2, 2)


    def _potential_func(self, u):
        """
        利用查表插值+线性外推计算势能值
        
        Args:
            u (float): 归一化的横向距离
            
        Returns:
            float: 计算得到的势能值
            
        Notes:
            使用预计算的积分值进行高效插值，并保持偶函数性质
            在查表范围外使用线性外推
        """
        u = abs(float(u))
        if u <= self.u_max:
            # 在非均匀网格上插值 I(u)
            return float(np.interp(u, self.t_grid, self.I_grid))
        else:
            # 线性外推：I(u) ≈ I(u_max) + eps * (u - u_max)
            slope = self.eps + self.A * np.exp(-((self.u_max - 0.5) / self.sigma) ** 2)
            return float(self.I_grid[-1] + (u - self.u_max) * slope)

    def get_potential_at_point(self, x, y, target_route_str=None):
        """
        计算世界坐标系中任意一点(x, y)的势能值
        
        Args:
            x, y (float): 要计算势能的点坐标
            target_route_str (str, optional): 目标路径标识符，如果指定则只考虑该路径
            
        Returns:
            float: 该点的势能值，值越小表示该点越适合车辆行驶
            
        Notes:
            1. 如指定路径，计算点到该路径中心线的最小距离
            2. 如未指定，找到距离点最近的路径
            3. 根据归一化的横向距离计算势能值
        """
        min_dist_to_centerline = float('inf')
        is_inside = False

        # ---- 选路径/算最近中心线距离逻辑 ----
        if target_route_str and target_route_str in self.routes:
            route_data = self.routes[target_route_str]
            centerline_np = np.array([p[:2] for p in route_data["smoothed"]])
            min_dist_to_centerline = np.min(
                np.linalg.norm(centerline_np - np.array([x, y]), axis=1)
            )
            is_inside = min_dist_to_centerline <= self.lane_width / 2
        else:
            relevant_routes = []
            for route_data in self.routes.values():
                bbox = route_data.get("bbox")
                if bbox and (bbox['min_x'] - 1 < x < bbox['max_x'] + 1 and 
                             bbox['min_y'] - 1 < y < bbox['max_y'] + 1):
                    relevant_routes.append(route_data)
            if not relevant_routes:
                relevant_routes = self.routes.values()
            for route_data in relevant_routes:
                centerline_np = np.array([p[:2] for p in route_data["smoothed"]])
                dist = np.min(np.linalg.norm(centerline_np - np.array([x, y]), axis=1))
                if dist < min_dist_to_centerline:
                    min_dist_to_centerline = dist
                    is_inside = dist <= self.lane_width / 2

        # --- 势能计算：normalized_lateral_dist 作为自变量 u ---
        normalized_lateral_dist = min_dist_to_centerline / (self.lane_width / 2)
        potential = self._potential_func(normalized_lateral_dist)
        return potential
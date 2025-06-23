# vehicle.py

import pygame
import numpy as np
import random
from config import *
from longitudinal_control import GippsModel, PIDController
from lateral_control import MPCController

class Vehicle:
    """
    一个独立的智能车辆代理。
    它拥有自己的物理属性、状态、路径以及一套完整的控制器。
    """
    def __init__(self, road, start_direction, end_direction, vehicle_id):
        self.road = road
        self.start_direction = start_direction
        self.end_direction = end_direction
        self.vehicle_id = vehicle_id

        self.move_str = f"{start_direction[0].upper()}_{end_direction[0].upper()}"
        
        path_data = self.road.get_route_points(start_direction, end_direction)
        if not path_data or not path_data["smoothed"]:
            raise ValueError(f"无法为车辆 {self.vehicle_id} 生成从 {self.move_str} 的路径")
            
        self.raw_path = path_data["raw"]
        self.reference_path = path_data["smoothed"] 
        self.path_points_np = np.array([[p[0], p[1]] for p in self.reference_path])
        self.path_distances = np.insert(np.cumsum(np.sqrt(np.sum(np.diff(self.path_points_np, axis=0)**2, axis=1))), 0, 0)

        initial_pos = self.reference_path[0]
        initial_psi = self.reference_path[1][2]

        # 车辆状态
        self.state = {
            'x': initial_pos[0],
            'y': initial_pos[1],
            'psi': initial_psi,
            'vx': 0.1,  # 初始给一个很小的速度防止计算问题
            'vy': 0.0,
            'psi_dot': 0.0
        }

        # 车辆物理属性 (从config加载)
        self.m, self.Iz, self.lf, self.lr, self.Cf, self.Cr = VEHICLE_MASS, VEHICLE_IZ, VEHICLE_LF, VEHICLE_LR, VEHICLE_CF, VEHICLE_CR
        self.width, self.length = VEHICLE_W, VEHICLE_L

        # --- 每个车辆拥有自己的一套控制器 ---
        self.gipps_model = GippsModel()
        self.pid_controller = PIDController()
        self.mpc_controller = MPCController()

        # 仿真状态
        self.completed = False
        self.color = (200, 200, 200) # 默认颜色
        self.path_index = 0

        path_data = self.road.get_route_points(start_direction, end_direction)
        if not path_data or not path_data["smoothed"]:
            raise ValueError(f"无法为车辆 {self.vehicle_id} 生成从 {start_direction} 到 {end_direction} 的路径")
        
        self.raw_path = path_data["raw"]
        self.reference_path = path_data["smoothed"] # 控制器使用平滑后的路径


        self.show_path_visualization = False
        self.show_debug = True
        self.show_bicycle_model = False
        self.last_steering_angle = 0.0

        self.decision_point_index = self._calculate_decision_point_index()
        self.is_approaching_decision_point = False
        self.is_in_intersection = False
        self.has_passed_intersection = False
        self.is_yielding = False # 是否处于让行停车状态
        self.decision_made = False # 是否已做出通过/让行的决策

    def _calculate_decision_point_index(self):
        """计算并返回决策点在路径上的索引"""
        a_max = abs(GIPPS_B)
        v_safe = GIPPS_V_DESIRED 
        d_stop = (v_safe**2) / (2 * a_max) + self.length # 加上车长作为缓冲

        # 找到路径上第一个进入交叉口冲突区的点
        entry_index = -1
        for i, point in enumerate(self.reference_path):
            if self.road.conflict_zone.collidepoint(point[0], point[1]):
                entry_index = i
                break
        if entry_index == -1: return -1 # 路径不经过交叉口

        # 从入口点向前回溯，找到距离满足d_stop的点
        for i in range(entry_index, -1, -1):
            distance_to_entry = self.path_distances[entry_index] - self.path_distances[i]
            if distance_to_entry >= d_stop:
                return i
        return 0 # 如果路径太短，就在起点决策

    def _update_intersection_status(self):
        """根据车辆位置更新交叉口相关的状态标志"""
        if self.has_passed_intersection:
            return

        # 找到当前在路径上的投影点索引
        current_pos = np.array([self.state['x'], self.state['y']])
        distances_to_ego = np.linalg.norm(self.path_points_np - current_pos, axis=1)
        current_path_index = np.argmin(distances_to_ego)

        # 更新状态机
        if not self.road.conflict_zone.collidepoint(current_pos[0], current_pos[1]):
            # 在交叉口外
            if current_path_index > self.decision_point_index and self.decision_point_index != -1:
                self.is_approaching_decision_point = True
            if self.is_in_intersection: # 刚开出交叉口
                self.has_passed_intersection = True
                self.is_in_intersection = False
        else:
            # 在交叉口内
            self.is_in_intersection = True
            self.is_approaching_decision_point = False
            self.is_yielding = False # 进入后不再让行，必须快速通过
            self.decision_made = True

    def find_leader(self, all_vehicles, lane_width=3.5):
        
        leader = None
        min_longitudinal_dist = float('inf')

        ego_pos = np.array([self.state['x'], self.state['y']])
        
        distances_to_ego = np.linalg.norm(self.path_points_np - ego_pos, axis=1)
        ego_proj_index = np.argmin(distances_to_ego)
        ego_longitudinal_pos = self.path_distances[ego_proj_index]

        for other_vehicle in all_vehicles:
            if other_vehicle.vehicle_id == self.vehicle_id:
                continue
                
            other_pos = np.array([other_vehicle.state['x'], other_vehicle.state['y']])
            distances_to_other = np.linalg.norm(self.path_points_np - other_pos, axis=1)
            other_proj_index = np.argmin(distances_to_other)
            
            lateral_dist = distances_to_other[other_proj_index]
            if lateral_dist > lane_width:
                continue

            other_longitudinal_pos = self.path_distances[other_proj_index]
            if other_longitudinal_pos <= ego_longitudinal_pos:
                continue
                
            longitudinal_dist = other_longitudinal_pos - ego_longitudinal_pos
            if longitudinal_dist < min_longitudinal_dist:
                min_longitudinal_dist = longitudinal_dist
                leader = other_vehicle
                
        return leader

    def update(self, dt, all_vehicles, traffic_manager):
        """
        车辆的“心跳”函数，执行一步完整的感知-决策-控制循环。
        """
        if self.completed:
            return

        # 1. 更新自身状态
        self._update_intersection_status()

        # 2. 交叉口决策
        # 只在到达决策点且未做决策时，进行一次决策
        if self.is_approaching_decision_point and not self.decision_made:
            self.is_yielding = traffic_manager.should_vehicle_yield(self, all_vehicles)
            
            # 增加概率性让行
            if not self.is_yielding and random.random() < 0.05:
                self.is_yielding = True
                print(f"Vehicle {self.vehicle_id} is yielding voluntarily.")

            # 无论结果如何，决策已做出
            self.decision_made = True 

            if self.is_yielding:
                print(f"Vehicle {self.vehicle_id} DECIDED TO YIELD.")
            else:
                print(f"Vehicle {self.vehicle_id} DECIDED TO PROCEED.")


        # 3. 控制计算
        # a. 纵向控制
        should_stop = self.is_yielding and not self.has_passed_intersection
        if should_stop:
            target_speed = 0.0
        else:
            leader_vehicle = self.find_leader(all_vehicles)
            leader_state = leader_vehicle.get_state() if leader_vehicle else None
            target_speed = self.gipps_model.calculate_target_speed(self.state, leader_state)
        
        throttle_brake = self.pid_controller.step(target_speed, self.state['vx'], dt)
        
        # b. 横向控制
        steering_angle = self.mpc_controller.solve(self.state, self.reference_path)
        self.last_steering_angle = steering_angle
        
        # 4. 执行
        self._update_physics(throttle_brake, steering_angle, dt)

        # 5. 检查完成
        dist_to_end = np.linalg.norm([self.state['x'] - self.reference_path[-1][0], self.state['y'] - self.reference_path[-1][1]])
        if dist_to_end < 10.0:
            self.completed = True

    def _update_physics(self, fx_total, delta, dt):
        """物理引擎，与上一版相同"""
        psi, vx, vy, psi_dot = self.state['psi'], self.state['vx'], self.state['vy'], self.state['psi_dot']
        if abs(vx) < 0.1: vx = np.sign(vx) * 0.1
        
        alpha_f = np.arctan((vy + self.lf * psi_dot) / vx) - delta
        alpha_r = np.arctan((vy - self.lr * psi_dot) / vx)

        Fyf = -self.Cf * alpha_f
        Fyr = -self.Cr * alpha_r
        
        ax = fx_total / self.m
        ay = (Fyf + Fyr) / self.m - vx * psi_dot
        psi_ddot_deriv = (self.lf * Fyf - self.lr * Fyr) / self.Iz
        
        self.state['vx'] += ax * dt
        self.state['vy'] += ay * dt
        self.state['psi_dot'] += psi_ddot_deriv * dt
        
        self.state['x'] += (vx * np.cos(psi) - vy * np.sin(psi)) * dt
        self.state['y'] += (vx * np.sin(psi) + vy * np.cos(psi)) * dt
        self.state['psi'] += psi_dot * dt
        
    def draw(self, surface, transform_func, small_font, scale=1.0):
        """
        以更精细的方式绘制车辆，包括车轮、风挡和调试信息。
        """
        x, y, psi = self.state['x'], self.state['y'], self.state['psi']
        delta = self.last_steering_angle

        # --- 绘制车身 ---
        w, l = self.width, self.length
        body_corners = [(-l/2, -w/2), (l/2, -w/2), (l/2, w/2), (-l/2, w/2)]
        rotated_body = self._rotate_and_transform(body_corners, x, y, psi, transform_func)
        pygame.draw.polygon(surface, self.color, rotated_body)
        pygame.draw.polygon(surface, (0, 0, 0), rotated_body, 1) # 黑色边框

        # --- 绘制风挡 ---
        windshield_corners = [(l/4, -w/2*0.9), (l/2*0.9, -w/2*0.8), (l/2*0.9, w/2*0.8), (l/4, w/2*0.9)]
        rotated_windshield = self._rotate_and_transform(windshield_corners, x, y, psi, transform_func)
        pygame.draw.polygon(surface, (100, 150, 200, 150), rotated_windshield) # 半透明蓝色

        # --- 绘制车轮 ---
        wheel_l, wheel_w = 0.6, 0.3
        wheel_positions = [
            (self.lf, -self.width / 2 * 0.9),  # 前右轮
            (self.lf, self.width / 2 * 0.9),   # 前左轮
            (-self.lr, -self.width / 2 * 0.9), # 后右轮
            (-self.lr, self.width / 2 * 0.9),  # 后左轮
        ]
        
        # 后轮 (无转向)
        rear_right_wheel = self._create_wheel_poly(wheel_positions[2], wheel_l, wheel_w, x, y, psi, 0, transform_func)
        rear_left_wheel = self._create_wheel_poly(wheel_positions[3], wheel_l, wheel_w, x, y, psi, 0, transform_func)
        pygame.draw.polygon(surface, (30, 30, 30), rear_right_wheel)
        pygame.draw.polygon(surface, (30, 30, 30), rear_left_wheel)

        # 前轮 (带转向)
        front_right_wheel = self._create_wheel_poly(wheel_positions[0], wheel_l, wheel_w, x, y, psi, delta, transform_func)
        front_left_wheel = self._create_wheel_poly(wheel_positions[1], wheel_l, wheel_w, x, y, psi, delta, transform_func)
        pygame.draw.polygon(surface, (30, 30, 30), front_right_wheel)
        pygame.draw.polygon(surface, (30, 30, 30), front_left_wheel)

        # 绘制自行车模型可视化
        if self.show_bicycle_model:
            self._draw_bicycle_model_visualization(surface, transform_func, scale)

        if self.show_path_visualization:
            # 绘制原始路径 (红色)
            if len(self.raw_path) > 1:
                transformed_raw_path = [transform_func(p[0], p[1]) for p in self.raw_path]
                pygame.draw.lines(surface, (255, 100, 100, 100), False, transformed_raw_path, 1)

            # 绘制平滑并重采样后的路径 (绿色)
            if len(self.reference_path) > 1:
                transformed_ref_path = [transform_func(p[0], p[1]) for p in self.reference_path]
                pygame.draw.lines(surface, (100, 255, 100), False, transformed_ref_path, 2)

        # --- 绘制调试信息 ---
        if self.show_debug:
            info_text = f"ID:{self.vehicle_id} V:{self.state['vx']:.1f}m/s"
            text_surface = small_font.render(info_text, True, (255, 255, 255))
            # 将文本放在车辆上方
            screen_pos = transform_func(x, y)
            text_rect = text_surface.get_rect(center=(screen_pos[0], screen_pos[1] - 20 * scale))
            surface.blit(text_surface, text_rect)

    def _draw_bicycle_model_visualization(self, surface, transform_func, scale):
        """在车辆上叠加绘制自行车模型的关键元素"""
        x, y, psi = self.state['x'], self.state['y'], self.state['psi']
        vx, vy, psi_dot = self.state['vx'], self.state['vy'], self.state['psi_dot']
        delta = self.last_steering_angle
        
        # 车辆中心点
        center_screen = transform_func(x, y)
        
        # 绘制底盘 (连接前后轴的线)
        rear_axle_world = (x - self.lr * np.cos(psi), y - self.lr * np.sin(psi))
        front_axle_world = (x + self.lf * np.cos(psi), y + self.lf * np.sin(psi))
        rear_axle_screen = transform_func(*rear_axle_world)
        front_axle_screen = transform_func(*front_axle_world)
        pygame.draw.line(surface, (255, 255, 0), rear_axle_screen, front_axle_screen, 2)
        
        # 绘制前后轮方向线
        wheel_len = self.lf * 0.8 # 可视化线长度
        # 后轮
        rear_wheel_end_world = (rear_axle_world[0] + wheel_len * np.cos(psi), rear_axle_world[1] + wheel_len * np.sin(psi))
        rear_wheel_end_screen = transform_func(*rear_wheel_end_world)
        pygame.draw.line(surface, (0, 255, 255), rear_axle_screen, rear_wheel_end_screen, 2)
        # 前轮 (考虑了转向角)
        front_wheel_angle = psi + delta
        front_wheel_end_world = (front_axle_world[0] + wheel_len * np.cos(front_wheel_angle), front_axle_world[1] + wheel_len * np.sin(front_wheel_angle))
        front_wheel_end_screen = transform_func(*front_wheel_end_world)
        pygame.draw.line(surface, (255, 0, 255), front_axle_screen, front_wheel_end_screen, 2)

        # 绘制速度矢量 (在车辆坐标系下)
        speed_angle_body = np.arctan2(vy, vx) # 车辆坐标系下的速度方向
        speed_angle_world = psi + speed_angle_body # 转换到世界坐标系
        speed_magnitude = np.sqrt(vx**2 + vy**2)
        speed_vector_end_world = (x + speed_magnitude * np.cos(speed_angle_world), y + speed_magnitude * np.sin(speed_angle_world))
        speed_vector_end_screen = transform_func(*speed_vector_end_world)
        pygame.draw.line(surface, (0, 255, 0), center_screen, speed_vector_end_screen, 2)
        pygame.draw.circle(surface, (0, 255, 0), speed_vector_end_screen, 4)

    def _rotate_and_transform(self, points, x, y, angle, transform_func):
        """辅助函数，旋转点集并变换到屏幕坐标"""
        rotated_points = []
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        for px, py in points:
            rx = px * cos_a - py * sin_a + x
            ry = px * sin_a + py * cos_a + y
            rotated_points.append(transform_func(rx, ry))
        return rotated_points

    def _create_wheel_poly(self, pos, l, w, x, y, psi, delta, transform_func):
        """辅助函数，创建并变换单个车轮的多边形"""
        wheel_corners = [(-l/2, -w/2), (l/2, -w/2), (l/2, w/2), (-l/2, w/2)]
        
        # 先进行车轮自身的转向
        rotated_wheel = []
        cos_d, sin_d = np.cos(delta), np.sin(delta)
        for wx, wy in wheel_corners:
            rx = wx * cos_d - wy * sin_d + pos[0]
            ry = wx * sin_d + wy * cos_d + pos[1]
            rotated_wheel.append((rx, ry))
            
        # 再随车身一起旋转和平移
        return self._rotate_and_transform(rotated_wheel, x, y, psi, transform_func)

    def toggle_debug_info(self):
        """切换调试信息的显示"""
        self.show_debug = not self.show_debug

    def get_state(self):
        """返回当前车辆状态的字典"""
        return self.state

    # 添加你的 game_engine.py 需要的空方法，防止报错
    def toggle_bicycle_visualization(self):
        """切换自行车模型的可视化显示"""
        self.show_bicycle_model = not self.show_bicycle_model

    def toggle_path_visualization(self):
        self.show_path_visualization = not self.show_path_visualization
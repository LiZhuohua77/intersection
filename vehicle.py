# vehicle.py

import pygame
import numpy as np
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
        
        # 获取路径并初始化状态
        self.reference_path = self.road.get_route_points(start_direction, end_direction)
        print(self.reference_path)
        if not self.reference_path:
            raise ValueError(f"无法为车辆 {self.vehicle_id} 生成从 {start_direction} 到 {end_direction} 的路径")
            
        initial_pos = self.reference_path[0]
        initial_psi = self.reference_path[1][2] # 使用路径第二点的角度作为初始角度

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

    def find_leader(self, all_vehicles):
        """在路径上寻找领头车辆"""
        leader = None
        min_dist = float('inf')

        for other_vehicle in all_vehicles:
            if other_vehicle.vehicle_id == self.vehicle_id:
                continue
            
            # 简单检查是否在同一条大概路径上（实际需要更复杂的路径匹配）
            if other_vehicle.start_direction == self.start_direction and other_vehicle.end_direction == self.end_direction:
                # 检查对方是否在自己前方
                dist = np.linalg.norm([self.state['x']-other_vehicle.state['x'], self.state['y']-other_vehicle.state['y']])
                
                # 判断前后关系（简化）
                dx = other_vehicle.state['x'] - self.state['x']
                is_in_front = dx * np.cos(self.state['psi']) + (other_vehicle.state['y'] - self.state['y']) * np.sin(self.state['psi']) > 0

                if is_in_front and dist < min_dist:
                    min_dist = dist
                    leader = other_vehicle
        
        return leader

    def update(self, dt, all_vehicles):
        """
        车辆的“心跳”函数，执行一步完整的感知-决策-控制循环。
        """
        if self.completed:
            return

        # 1. 感知 (Perception)
        leader_vehicle = self.find_leader(all_vehicles)
        leader_state = leader_vehicle.get_state() if leader_vehicle else None

        # 2. 决策与控制 (Decision & Control)
        # a. 纵向控制
        target_speed = self.gipps_model.calculate_target_speed(self.state, leader_state)
        throttle_brake = self.pid_controller.step(target_speed, self.state['vx'], dt)
        
        # b. 横向控制
        steering_angle = self.mpc_controller.solve(self.state, self.reference_path)
        
        # 3. 执行 (Actuation) - 更新车辆物理状态
        self._update_physics(throttle_brake, steering_angle, dt)

        # 4. 检查是否完成路径
        dist_to_end = np.linalg.norm([self.state['x'] - self.reference_path[-1][0], self.state['y'] - self.reference_path[-1][1]])
        if dist_to_end < 10.0: # 到达终点附近
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
        
    def draw(self, surface, transform_func, scale=1.0):
        """绘制车辆"""
        x, y, psi = self.state['x'], self.state['y'], self.state['psi']
        
        # 车辆矩形顶点（世界坐标）
        w, l = self.width, self.length
        corners = [
            (-l/2, -w/2), (l/2, -w/2), (l/2, w/2), (-l/2, w/2)
        ]
        
        # 旋转和平移
        rotated_corners = []
        for cx, cy in corners:
            rx = cx * np.cos(psi) - cy * np.sin(psi) + x
            ry = cx * np.sin(psi) + cy * np.cos(psi) + y
            rotated_corners.append(transform_func(rx, ry))

        # 绘制车身
        pygame.draw.polygon(surface, self.color, rotated_corners)
        
        # 绘制一个指向车头的三角形，方便观察方向
        front_point = (x + l/2 * np.cos(psi), y + l/2 * np.sin(psi))
        screen_front_point = transform_func(front_point[0], front_point[1])
        pygame.draw.line(surface, (255, 255, 255), rotated_corners[1], screen_front_point, 2)
        pygame.draw.line(surface, (255, 255, 255), rotated_corners[2], screen_front_point, 2)

    def get_state(self):
        """返回当前车辆状态的字典"""
        return self.state

    # 添加你的 game_engine.py 需要的空方法，防止报错
    def toggle_bicycle_visualization(self): pass
    def toggle_debug_info(self): pass
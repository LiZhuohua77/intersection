import numpy as np
import math
import pygame

# 车辆物理参数
max_steer = np.radians(30.0)  # [rad] max steering angle
L = 3  # [m] Wheel base of vehicle
dt = 0.1
Lr = L / 2  # [m]
Lf = L - Lr
Cf = 1600.0 * 20  # N/rad
Cr = 1700.0 * 20  # N/rad
Iz = 2250.0  # kg/m2
m = 1500.0  # kg

# IDM参数
class IDMParams:
    """IDM跟车模型参数"""
    def __init__(self):
        self.v_desired = 20.0      # 期望速度 (m/s)
        self.T = 1.5               # 安全时间间隔 (s)
        self.s0 = 5.0              # 最小跟车距离 (m)
        self.a_max = 3.0           # 最大加速度 (m/s²)
        self.b_comfortable = 3.0    # 舒适减速度 (m/s²)
        self.delta = 4.0           # 加速度指数


def normalize_angle(angle):
    """
    Normalize an angle to [-pi, pi].
    :param angle: (float)
    :return: (float) Angle in radian in [-pi, pi]
    """
    while angle > np.pi:
        angle -= 2.0 * np.pi

    while angle < -np.pi:
        angle += 2.0 * np.pi

    return angle


class NonLinearBicycleModel:
    """非线性自行车模型"""
    def __init__(self, x=0.0, y=0.0, yaw=0.0, vx=0.01, vy=0, omega=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.vx = vx
        self.vy = vy
        self.omega = omega
        # Aerodynamic and friction coefficients
        self.c_a = 1.36
        self.c_r1 = 0.01

    def update(self, throttle, delta):
        """更新车辆状态"""
        delta = np.clip(delta, -max_steer, max_steer)
        self.x = self.x + self.vx * math.cos(self.yaw) * dt - self.vy * math.sin(self.yaw) * dt
        self.y = self.y + self.vx * math.sin(self.yaw) * dt + self.vy * math.cos(self.yaw) * dt
        self.yaw = self.yaw + self.omega * dt
        self.yaw = normalize_angle(self.yaw)
        Ffy = -Cf * math.atan2(((self.vy + Lf * self.omega) / self.vx - delta), 1.0)
        Fry = -Cr * math.atan2((self.vy - Lr * self.omega) / self.vx, 1.0)
        R_x = self.c_r1 * self.vx
        F_aero = self.c_a * self.vx ** 2
        F_load = F_aero + R_x
        self.vx = self.vx + (throttle - Ffy * math.sin(delta) / m - F_load/m + self.vy * self.omega) * dt
        self.vy = self.vy + (Fry / m + Ffy * math.cos(delta) / m - self.vx * self.omega) * dt
        self.omega = self.omega + (Ffy * Lf * math.cos(delta) - Fry * Lr) / Iz * dt


class Vehicle:
    """自动驾驶车辆类，使用自行车模型和IDM纵向控制"""
    def __init__(self, road, start_direction, end_direction, vehicle_id):
        self.road = road
        self.start_direction = start_direction
        self.end_direction = end_direction
        self.vehicle_id = vehicle_id
        
        # 车辆物理属性
        self.length = 4
        self.width = 2
        self.color = (200, 50, 50)  # 红色
        
        # 自行车模型状态 - 以后轮为参考点
        self.x = 0  # 后轮位置 x
        self.y = 0  # 后轮位置 y
        self.yaw = 0  # 车身朝向角
        self.speed = 15.0  # m/s
        self.steering_angle = 0.0  # 前轮转向角
        
        # IDM纵向控制
        self.idm_params = IDMParams()
        self.acceleration = 0.0  # 当前加速度
        self.leading_vehicle = None  # 前车
        self.gap_distance = float('inf')  # 与前车的间距
        self.relative_speed = 0.0  # 相对速度
        
        # 显示设置
        self.show_bicycle_model = False
        self.show_debug_info = False
        
        # 路径规划
        self.path_points = []
        self.current_path_index = 0
        self.completed = False
        
        # 轨迹记录（后轮轨迹）
        self.trajectory = []
        self.max_trajectory_points = 20
        
        # 初始化位置和路径
        self._initialize_position_and_path()
    
    def _initialize_position_and_path(self):
        """初始化车辆位置和路径"""
        # 获取路径点
        self.path_points = self.road.get_route_points(self.start_direction, self.end_direction)
        
        if self.path_points:
            # 设置初始位置（前轮位置）
            start_point = self.path_points[0]
            front_x = start_point[0]
            front_y = start_point[1]
            
            # 计算初始朝向
            if len(self.path_points) > 1:
                next_point = self.path_points[1]
                dx = next_point[0] - front_x
                dy = next_point[1] - front_y
                self.yaw = math.atan2(dy, dx)
            
            # 根据前轮位置和朝向计算后轮位置
            self.x = front_x - L * math.cos(self.yaw)
            self.y = front_y - L * math.sin(self.yaw)
    
    def find_leading_vehicle(self, all_vehicles):
        """寻找同车道前方最近的车辆"""
        self.leading_vehicle = None
        min_distance = float('inf')
        
        # 计算前轮位置
        front_x, front_y = self.get_front_wheel_position()
        
        for other_vehicle in all_vehicles:
            if other_vehicle == self or other_vehicle.completed:
                continue
            
            # 检查是否在同一车道（简化：检查是否在相似路径上）
            if not self._is_same_lane(other_vehicle):
                continue
            
            # 计算其他车辆的后轮位置
            other_rear_x = other_vehicle.x
            other_rear_y = other_vehicle.y
            
            # 计算到其他车辆的距离
            distance = math.sqrt((front_x - other_rear_x)**2 + (front_y - other_rear_y)**2)
            
            # 检查是否在前方（简化：距离判断）
            if self._is_vehicle_ahead(other_vehicle) and distance < min_distance:
                min_distance = distance
                self.leading_vehicle = other_vehicle
        
        # 更新跟车参数
        if self.leading_vehicle:
            self.gap_distance = min_distance - self.length  # 净距离
            self.relative_speed = self.speed - self.leading_vehicle.speed
        else:
            self.gap_distance = float('inf')
            self.relative_speed = 0.0
    
    def _is_same_lane(self, other_vehicle):
        """判断是否在同一车道（简化版本）"""
        # 简化判断：如果起始和结束方向相同，认为在同一车道
        return (self.start_direction == other_vehicle.start_direction and 
                self.end_direction == other_vehicle.end_direction)
    
    def _is_vehicle_ahead(self, other_vehicle):
        """判断其他车辆是否在前方"""
        # 计算本车前轮位置
        front_x, front_y = self.get_front_wheel_position()
        
        # 计算其他车辆后轮位置
        other_rear_x = other_vehicle.x
        other_rear_y = other_vehicle.y
        
        # 计算方向向量
        direction_x = math.cos(self.yaw)
        direction_y = math.sin(self.yaw)
        
        # 计算到其他车辆的向量
        to_other_x = other_rear_x - front_x
        to_other_y = other_rear_y - front_y
        
        # 点积判断是否在前方
        dot_product = to_other_x * direction_x + to_other_y * direction_y
        return dot_product > 0
    
    def calculate_idm_acceleration(self):
        """计算IDM加速度"""
        v = self.speed
        v_desired = self.idm_params.v_desired
        T = self.idm_params.T
        s0 = self.idm_params.s0
        a_max = self.idm_params.a_max
        b = self.idm_params.b_comfortable
        delta = self.idm_params.delta
        
        # 自由流加速度项
        free_flow_term = a_max * (1 - (v / v_desired) ** delta)
        
        # 跟车减速度项
        if self.leading_vehicle and self.gap_distance < float('inf'):
            s = max(self.gap_distance, 0.1)  # 避免除零
            dv = self.relative_speed
            
            # 期望距离
            s_star = s0 + max(0, v * T + (v * dv) / (2 * math.sqrt(a_max * b)))
            
            # 跟车项
            interaction_term = a_max * (s_star / s) ** 2
            
            acceleration = free_flow_term - interaction_term
        else:
            # 没有前车，只考虑自由流项
            acceleration = free_flow_term
        
        # 限制加速度范围
        max_decel = -3.0  # 最大减速度
        max_accel = a_max
        acceleration = np.clip(acceleration, max_decel, max_accel)
        
        return acceleration
    
    def update(self, dt_sim, all_vehicles=None):
        """使用自行车模型和IDM更新车辆状态"""
        if self.completed or not self.path_points:
            return
        
        # 寻找前车
        if all_vehicles:
            self.find_leading_vehicle(all_vehicles)
        
        # 计算IDM加速度
        self.acceleration = self.calculate_idm_acceleration()
        
        # 更新速度
        self.speed += self.acceleration * dt_sim
        self.speed = max(0.0, self.speed)  # 确保速度非负
        
        # 路径跟踪控制（横向控制）
        if self.current_path_index < len(self.path_points):
            target_point = self.path_points[self.current_path_index]
            
            # 计算前轮位置
            front_x = self.x + L * math.cos(self.yaw)
            front_y = self.y + L * math.sin(self.yaw)
            
            # 计算到目标点的距离和角度
            dx = target_point[0] - front_x
            dy = target_point[1] - front_y
            distance = math.sqrt(dx**2 + dy**2)
            
            if distance < 10.0:  # 接近目标点
                self.current_path_index += 1
                if self.current_path_index >= len(self.path_points):
                    self.completed = True
                    return
            
            # 计算期望的前轮朝向
            desired_front_yaw = math.atan2(dy, dx)
            
            # 计算转向角度（前轮相对于车身的角度）
            self.steering_angle = normalize_angle(desired_front_yaw - self.yaw)
            self.steering_angle = np.clip(self.steering_angle, -max_steer, max_steer)
            
            # 自行车模型动力学更新
            # 后轮速度分量
            vx = self.speed * math.cos(self.yaw)
            vy = self.speed * math.sin(self.yaw)
            
            # 更新后轮位置（车辆轨迹）
            self.x += vx * dt_sim
            self.y += vy * dt_sim
            
            # 更新车身朝向（基于自行车模型）
            # 角速度 = (v * sin(delta)) / L，其中delta是前轮转向角
            angular_velocity = (self.speed * math.sin(self.steering_angle)) / L
            self.yaw += angular_velocity * dt_sim
            self.yaw = normalize_angle(self.yaw)
            
            # 记录轨迹（后轮轨迹）
            self.trajectory.append((self.x, self.y))
            if len(self.trajectory) > self.max_trajectory_points:
                self.trajectory.pop(0)
    
    def get_front_wheel_position(self):
        """获取前轮位置"""
        front_x = self.x + L * math.cos(self.yaw)
        front_y = self.y + L * math.sin(self.yaw)
        return front_x, front_y
    
    def get_vehicle_center_position(self):
        """获取车辆中心位置（用于绘制车身）"""
        center_x = self.x + (L/2) * math.cos(self.yaw)
        center_y = self.y + (L/2) * math.sin(self.yaw)
        return center_x, center_y
    
    def toggle_bicycle_visualization(self):
        """切换自行车模型可视化"""
        self.show_bicycle_model = not self.show_bicycle_model
    
    def toggle_debug_info(self):
        """切换调试信息显示"""
        self.show_debug_info = not self.show_debug_info
    
    def draw(self, surface, transform_func=None, scale=1.0):
        """绘制车辆"""
        if transform_func is None:
            transform_func = lambda x, y: (x, y)
        
        if self.completed:
            return
        
        # 获取车辆中心位置用于绘制车身
        center_x, center_y = self.get_vehicle_center_position()
        
        # 车辆顶点（相对于车辆中心）
        half_length = self.length / 2
        half_width = self.width / 2
        
        corners = [
            (-half_length, -half_width),
            (half_length, -half_width),
            (half_length, half_width),
            (-half_length, half_width)
        ]
        
        # 变换到世界坐标
        cos_yaw = math.cos(self.yaw)
        sin_yaw = math.sin(self.yaw)
        
        world_corners = []
        for lx, ly in corners:
            wx = lx * cos_yaw - ly * sin_yaw + center_x
            wy = lx * sin_yaw + ly * cos_yaw + center_y
            sx, sy = transform_func(wx, wy)
            world_corners.append((sx, sy))
        
        # 根据加速度状态调整车身颜色
        if self.acceleration > 0:
            # 加速时显示绿色边框
            border_color = (0, 255, 0)
        elif self.acceleration < 0:
            # 减速时显示红色边框
            border_color = (255, 0, 0)
        else:
            # 正常状态白色边框
            border_color = (255, 255, 255)
        
        # 绘制车身
        pygame.draw.polygon(surface, self.color, world_corners)
        pygame.draw.polygon(surface, border_color, world_corners, 2)
        
        # 绘制方向指示器
        front_center_x = center_x + half_length * cos_yaw
        front_center_y = center_y + half_length * sin_yaw
        arrow_length = 3
        arrow_x = front_center_x + arrow_length * cos_yaw
        arrow_y = front_center_y + arrow_length * sin_yaw
        
        front_screen = transform_func(front_center_x, front_center_y)
        arrow_screen = transform_func(arrow_x, arrow_y)
        
        pygame.draw.line(surface, (255, 255, 0), front_screen, arrow_screen, 3)
        
        # 绘制跟车关系指示线
        if self.leading_vehicle and self.show_debug_info:
            # 绘制到前车的连线
            front_x, front_y = self.get_front_wheel_position()
            leader_rear_x = self.leading_vehicle.x
            leader_rear_y = self.leading_vehicle.y
            
            front_screen = transform_func(front_x, front_y)
            leader_screen = transform_func(leader_rear_x, leader_rear_y)
            
            # 根据距离选择颜色
            if self.gap_distance < 20:
                line_color = (255, 0, 0)  # 距离太近，红色
            elif self.gap_distance < 40:
                line_color = (255, 255, 0)  # 距离适中，黄色
            else:
                line_color = (0, 255, 0)  # 距离安全，绿色
            
            pygame.draw.line(surface, line_color, front_screen, leader_screen, 1)
        
        # 绘制轨迹（后轮轨迹）
        if len(self.trajectory) > 1:
            trajectory_points = []
            for tx, ty in self.trajectory:
                sx, sy = transform_func(tx, ty)
                trajectory_points.append((sx, sy))
            
            # 绘制轨迹线
            if len(trajectory_points) > 1:
                pygame.draw.lines(surface, (100, 100, 255), False, trajectory_points, 2)
        
        # 绘制自行车模型
        if self.show_bicycle_model:
            wheel_length = 2  # 轮子长度（像素）
            wheel_width = 1   # 轮子宽度（像素）
            
            # 后轮位置（车辆参考点）
            rear_wheel_x = self.x
            rear_wheel_y = self.y
            
            # 前轮位置
            front_wheel_x, front_wheel_y = self.get_front_wheel_position()
            
            # 绘制后轮（蓝色，与车身方向一致）
            rear_wheel_corners = [
                (-wheel_length/2, -wheel_width/2),
                (wheel_length/2, -wheel_width/2),
                (wheel_length/2, wheel_width/2),
                (-wheel_length/2, wheel_width/2)
            ]
            
            rear_world_corners = []
            for lx, ly in rear_wheel_corners:
                wx = lx * cos_yaw - ly * sin_yaw + rear_wheel_x
                wy = lx * sin_yaw + ly * cos_yaw + rear_wheel_y
                sx, sy = transform_func(wx, wy)
                rear_world_corners.append((sx, sy))
            
            pygame.draw.polygon(surface, (0, 0, 255), rear_world_corners)
            pygame.draw.polygon(surface, (255, 255, 255), rear_world_corners, 2)
            
            # 绘制前轮（绿色，有独立转向角度）
            front_wheel_yaw = self.yaw + self.steering_angle
            front_cos_yaw = math.cos(front_wheel_yaw)
            front_sin_yaw = math.sin(front_wheel_yaw)
            
            front_wheel_corners = [
                (-wheel_length/2, -wheel_width/2),
                (wheel_length/2, -wheel_width/2),
                (wheel_length/2, wheel_width/2),
                (-wheel_length/2, wheel_width/2)
            ]
            
            front_world_corners = []
            for lx, ly in front_wheel_corners:
                wx = lx * front_cos_yaw - ly * front_sin_yaw + front_wheel_x
                wy = lx * front_sin_yaw + ly * front_cos_yaw + front_wheel_y
                sx, sy = transform_func(wx, wy)
                front_world_corners.append((sx, sy))
            
            pygame.draw.polygon(surface, (0, 255, 0), front_world_corners)
            pygame.draw.polygon(surface, (255, 255, 255), front_world_corners, 2)
            
            # 绘制轴距线
            front_wheel_screen = transform_func(front_wheel_x, front_wheel_y)
            rear_wheel_screen = transform_func(rear_wheel_x, rear_wheel_y)
            pygame.draw.line(surface, (255, 255, 255), 
                           (int(front_wheel_screen[0]), int(front_wheel_screen[1])),
                           (int(rear_wheel_screen[0]), int(rear_wheel_screen[1])), 2)
            
            # 标记后轮（轨迹参考点）- 红色圆点
            pygame.draw.circle(surface, (255, 0, 0), 
                             (int(rear_wheel_screen[0]), int(rear_wheel_screen[1])), 
                             int(1 * scale))
            
            # 绘制转向角度指示线
            if abs(self.steering_angle) > 0.01:
                indicator_length = 3
                indicator_x = front_wheel_x + indicator_length * front_cos_yaw
                indicator_y = front_wheel_y + indicator_length * front_sin_yaw
                indicator_screen = transform_func(indicator_x, indicator_y)
                
                pygame.draw.line(surface, (255, 0, 255), 
                               (int(front_wheel_screen[0]), int(front_wheel_screen[1])),
                               (int(indicator_screen[0]), int(indicator_screen[1])), 3)
        
        # 绘制调试信息
        if self.show_debug_info:
            center_x, center_y = self.get_vehicle_center_position()
            screen_x, screen_y = transform_func(center_x, center_y)
            
            debug_texts = [
                f"Speed: {self.speed:.1f} m/s",
                f"Accel: {self.acceleration:.1f} m/s²",
                f"Yaw: {math.degrees(self.yaw):.1f}°",
                f"Steering: {math.degrees(self.steering_angle):.1f}°",
                f"Path: {self.current_path_index}/{len(self.path_points)}"
            ]
            
            # IDM相关信息
            if self.leading_vehicle:
                debug_texts.extend([
                    f"Leader: #{self.leading_vehicle.vehicle_id}",
                    f"Gap: {self.gap_distance:.1f}m",
                    f"RelSpd: {self.relative_speed:.1f}m/s"
                ])
            else:
                debug_texts.append("No Leader")
            
            if self.show_bicycle_model:
                front_x, front_y = self.get_front_wheel_position()
                debug_texts.extend([
                    f"Rear: ({self.x:.1f}, {self.y:.1f})",
                    f"Front: ({front_x:.1f}, {front_y:.1f})"
                ])
            
            font = pygame.font.SysFont(None, 16)
            line_height = 16
            text_width = 220
            text_height = len(debug_texts) * line_height + 10
            
            info_bg = pygame.Surface((text_width, text_height), pygame.SRCALPHA)
            info_bg.fill((0, 0, 0, 180))
            surface.blit(info_bg, (screen_x + 25, screen_y - text_height // 2))
            
            for i, text in enumerate(debug_texts):
                if "Accel:" in text:
                    if self.acceleration > 0.5:
                        color = (0, 255, 0)  # 加速绿色
                    elif self.acceleration < -1.0:
                        color = (255, 0, 0)  # 减速红色
                    else:
                        color = (255, 255, 255)
                elif "Leader:" in text or "Gap:" in text or "RelSpd:" in text:
                    color = (255, 255, 0)  # IDM信息黄色
                elif "No Leader" in text:
                    color = (150, 150, 150)  # 无前车灰色
                else:
                    color = (255, 255, 255)
                    
                text_surface = font.render(text, True, color)
                surface.blit(text_surface, (screen_x + 30, screen_y - text_height // 2 + 5 + i * line_height))
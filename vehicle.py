import pygame
import math
import numpy as np

class Vehicle:
    def __init__(self, road, start_direction, end_direction, id=0):
        self.road = road
        self.id = id
        self.start_direction = start_direction
        self.end_direction = end_direction
        
        # 获取完整路径点序列
        self.route_points = road.get_route_points(start_direction, end_direction)
        self.current_point_index = 0
        
        # 初始位置和角度
        self.x, self.y = self.route_points[0] if self.route_points else (0, 0)
        self.angle = self._get_initial_angle(start_direction)
        
        # 车辆物理参数
        self.width = 20
        self.length = 40
        self.wheelbase = 30  # 轴距
        
        # 动态自行车模型状态变量
        self.vx = 0  # 纵向速度
        self.vy = 0  # 横向速度
        self.yaw_rate = 0  # 角速度
        
        # IDM参数
        self.desired_speed = 20  # 期望速度
        self.max_acceleration = 2.0  # 最大加速度
        self.max_deceleration = -4.0  # 最大减速度
        self.time_headway = 1.5  # 时间间距
        self.min_spacing = 5.0  # 最小间距
        self.delta = 4.0  # IDM指数
        
        # Pure Pursuit参数
        self.lookahead_distance = 30  # 前视距离
        self.max_steering_angle = math.radians(30)  # 最大转向角
        
        # 车辆动力学参数
        self.mass = 1500  # 质量 kg
        self.Iz = 2000  # 绕z轴转动惯量 kg*m²
        self.Cf = 60000  # 前轮侧偏刚度 N/rad
        self.Cr = 80000  # 后轮侧偏刚度 N/rad
        self.lf = self.wheelbase * 0.6  # 质心到前轴距离
        self.lr = self.wheelbase * 0.4  # 质心到后轴距离
        
        self.completed = False

    def _get_initial_angle(self, direction):
        """根据方向获取初始角度（弧度）"""
        angle_map = {
            'north': math.pi * 1.5,
            'south': math.pi * 0.5,
            'east': math.pi,
            'west': 0
        }
        return angle_map.get(direction, 0)

    def _find_lookahead_point(self):
        """使用Pure Pursuit算法找到前视点"""
        if not self.route_points or self.current_point_index >= len(self.route_points) - 1:
            return None
        
        # 从当前点开始寻找前视点
        for i in range(self.current_point_index, len(self.route_points)):
            point = self.route_points[i]
            distance = math.sqrt((point[0] - self.x)**2 + (point[1] - self.y)**2)
            
            if distance >= self.lookahead_distance:
                #print(f"车辆 {self.id}: 前视点 {point}, 距离 {distance:.2f}")
                return point
        
        # 如果没找到足够远的点，返回最后一个点
        #print(f"车辆 {self.id}: 使用最后一个点作为前视点 {self.route_points[-1]}")
        return self.route_points[-1]

    def _pure_pursuit_control(self):
        """Pure Pursuit横向控制"""
        lookahead_point = self._find_lookahead_point()
        if lookahead_point is None:
            return 0
        
        # 计算车辆坐标系下的前视点位置
        dx = lookahead_point[0] - self.x
        dy = lookahead_point[1] - self.y
        
        # 转换到车辆坐标系
        local_x = dx * math.cos(self.angle) + dy * math.sin(self.angle)
        local_y = -dx * math.sin(self.angle) + dy * math.cos(self.angle)
        
        # 计算曲率
        ld = math.sqrt(local_x**2 + local_y**2)
        if ld < 1e-6:
            return 0
        
        curvature = 2 * local_y / (ld**2)
        
        # 计算前轮转向角
        steering_angle = math.atan(self.wheelbase * curvature)
        
        # 限制转向角
        steering_angle = max(-self.max_steering_angle, 
                           min(self.max_steering_angle, steering_angle))
        
        return steering_angle

    def _idm_longitudinal_control(self, front_vehicle=None):
        """IDM纵向控制"""
        current_speed = math.sqrt(self.vx**2 + self.vy**2)
        
        # 计算自由流加速度
        speed_ratio = current_speed / self.desired_speed
        free_acceleration = self.max_acceleration * (1 - speed_ratio**self.delta)
        
        # 如果没有前车，使用自由流加速度
        if front_vehicle is None:
            acceleration = free_acceleration
        else:
            # 计算与前车的距离和相对速度
            distance = math.sqrt((front_vehicle.x - self.x)**2 + 
                               (front_vehicle.y - self.y)**2)
            front_speed = math.sqrt(front_vehicle.vx**2 + front_vehicle.vy**2)
            relative_speed = current_speed - front_speed
            
            # 计算期望间距
            desired_spacing = (self.min_spacing + 
                             self.time_headway * current_speed +
                             (current_speed * relative_speed) / 
                             (2 * math.sqrt(self.max_acceleration * abs(self.max_deceleration))))
            
            # IDM加速度
            spacing_ratio = desired_spacing / max(distance, 0.1)
            interaction_acceleration = -self.max_acceleration * (spacing_ratio**2)
            acceleration = free_acceleration + interaction_acceleration
        
        # 限制加速度
        acceleration = max(self.max_deceleration, 
                         min(self.max_acceleration, acceleration))
        
        return acceleration

    def _dynamic_bicycle_model(self, steering_angle, acceleration, dt):
        """动态自行车模型"""
        # 限制时间步长
        dt = min(dt, 0.1)
        
        # 限制转向角
        steering_angle = max(-self.max_steering_angle, min(self.max_steering_angle, steering_angle))
        
        # 当前速度
        speed = math.sqrt(self.vx**2 + self.vy**2)
        
        # 转弯时降低速度
        if abs(steering_angle) > math.radians(15):
            max_turn_speed = 15
            if speed > max_turn_speed:
                acceleration = min(acceleration, -1.0)  # 强制减速
        
        if speed < 1.0:  # 低速处理
            # 简化的运动学模型
            self.vx += acceleration * dt
            self.vx = max(0, min(self.desired_speed, self.vx))  # 限制速度范围
            
            self.x += self.vx * math.cos(self.angle) * dt
            self.y += self.vx * math.sin(self.angle) * dt
            
            if abs(self.vx) > 0.1:
                self.angle += (self.vx * math.tan(steering_angle) / self.wheelbase) * dt
        else:
            try:
                # 计算前后轮侧偏角
                if abs(self.vx) > 0.5:
                    alpha_f = steering_angle - (self.vy + self.lf * self.yaw_rate) / max(0.5, abs(self.vx))
                    alpha_r = -(self.vy - self.lr * self.yaw_rate) / max(0.5, abs(self.vx))
                else:
                    alpha_f = steering_angle
                    alpha_r = 0
                
                # 限制侧偏角，防止过大
                alpha_f = max(-0.5, min(0.5, alpha_f))
                alpha_r = max(-0.5, min(0.5, alpha_r))
                
                # 计算轮胎侧向力
                Fyf = self.Cf * alpha_f
                Fyr = self.Cr * alpha_r
                
                # 纵向力
                Fx = self.mass * acceleration
                
                # 动力学方程
                ax = (Fx - Fyf * math.sin(steering_angle)) / self.mass + self.vy * self.yaw_rate
                ay = (Fyf * math.cos(steering_angle) + Fyr) / self.mass - self.vx * self.yaw_rate
                yaw_acceleration = (self.lf * Fyf * math.cos(steering_angle) - self.lr * Fyr) / self.Iz
                
                # 限制加速度
                ax = max(-20, min(20, ax))
                ay = max(-20, min(20, ay))
                yaw_acceleration = max(-5, min(5, yaw_acceleration))
                
                # 更新状态
                self.vx += ax * dt
                self.vy += ay * dt
                self.yaw_rate += yaw_acceleration * dt
                
                # 限制速度和角速度
                self.vx = max(0, min(self.desired_speed * 1.1, self.vx))  # 不允许倒车
                self.vy = max(-10, min(10, self.vy))  # 限制侧向速度
                self.yaw_rate = max(-2, min(2, self.yaw_rate))  # 限制角速度
                
                # 更新位置和角度
                dx = (self.vx * math.cos(self.angle) - self.vy * math.sin(self.angle)) * dt
                dy = (self.vx * math.sin(self.angle) + self.vy * math.cos(self.angle)) * dt
                
                # 限制位移量
                dx = max(-20, min(20, dx))
                dy = max(-20, min(20, dy))
                
                self.x += dx
                self.y += dy
                self.angle += self.yaw_rate * dt
            
            except Exception as e:
                print(f"车辆 {self.id} 动力学模型计算错误: {e}")
                # 错误时使用简化模型
                self.vx = max(0, min(self.desired_speed, self.vx))
                self.vy = 0
                self.yaw_rate = 0
                self.x += self.vx * math.cos(self.angle) * dt
                self.y += self.vx * math.sin(self.angle) * dt
        
        # 角度归一化
        self.angle = self.angle % (2 * math.pi)
        
        # 检查坐标是否合理
        if (math.isnan(self.x) or math.isnan(self.y) or
            math.isinf(self.x) or math.isinf(self.y) or
            abs(self.x) > 1e6 or abs(self.y) > 1e6):
            print(f"车辆 {self.id} 坐标计算错误，重置位置")
            # 尝试恢复到路径上的合理位置
            if self.current_point_index < len(self.route_points):
                self.x, self.y = self.route_points[self.current_point_index]
            else:
                self.completed = True

    def _update_current_point_index(self):
        """更新当前路径点索引"""
        if self.current_point_index >= len(self.route_points) - 1:
            return
        
        # 检查是否接近当前目标点
        target_point = self.route_points[self.current_point_index + 1]
        distance = math.sqrt((target_point[0] - self.x)**2 + (target_point[1] - self.y)**2)
        
        # 如果接近目标点，切换到下一个点
        if distance < 10:  # 阈值距离
            self.current_point_index += 1

    def update(self, dt=0.1, front_vehicle=None):
        """更新车辆状态
        
        Args:
            dt: 时间步长
            front_vehicle: 前方车辆对象（用于IDM）
        """
        if self.completed or not self.route_points:
            return
        
        # 更新当前路径点索引
        self._update_current_point_index()
        
        # 检查是否完成路线
        if self.current_point_index >= len(self.route_points) - 1:
            current_speed = math.sqrt(self.vx**2 + self.vy**2)
            if current_speed < 1.0:  # 速度足够低时认为完成
                self.completed = True
                return
        
        # Pure Pursuit横向控制
        steering_angle = self._pure_pursuit_control()
        
        # IDM纵向控制
        acceleration = self._idm_longitudinal_control(front_vehicle)
        
        # 动态自行车模型更新
        self._dynamic_bicycle_model(steering_angle, acceleration, dt)

    def draw(self, surface, transform_func=None):
        """在surface上绘制车辆
        
        Args:
            surface: 要绘制的表面
            transform_func: 坐标转换函数，接收 (x, y) 返回转换后的 (x, y)
        """
        if self.completed:
            return
        
        # 如果未提供转换函数，则使用恒等映射
        if transform_func is None:
            transform_func = lambda x, y: (x, y)
        
        # 检查坐标是否有效，避免数值溢出
        try:
            # 如果坐标无效，则打印信息并跳过绘制
            if (not isinstance(self.x, (int, float)) or 
                not isinstance(self.y, (int, float)) or
                math.isnan(self.x) or math.isnan(self.y) or
                math.isinf(self.x) or math.isinf(self.y) or
                abs(self.x) > 1e6 or abs(self.y) > 1e6):
                print(f"车辆 {self.id} 坐标无效: x={self.x}, y={self.y}, vx={self.vx}, vy={self.vy}")
                self.completed = True  # 标记为已完成，不再更新
                return
                
            # 将世界坐标转换为屏幕坐标
            screen_x, screen_y = transform_func(self.x, self.y)
            
            # 创建车辆矩形
            car_rect = pygame.Rect(0, 0, self.length, self.width)
            car_rect.center = (int(screen_x), int(screen_y))  # 确保是整数坐标
            
            # 余下的绘制代码
            car_surf = pygame.Surface((self.length, self.width), pygame.SRCALPHA)
            pygame.draw.rect(car_surf, (0, 0, 255, 200), car_surf.get_rect(), 0, border_radius=5)
            
            angle_degrees = math.degrees(self.angle)
            rotated_car = pygame.transform.rotate(car_surf, -angle_degrees)
            rotated_rect = rotated_car.get_rect(center=(int(screen_x), int(screen_y)))
            
            surface.blit(rotated_car, rotated_rect.topleft)
            
            font = pygame.font.SysFont(None, 20)
            speed = math.sqrt(self.vx**2 + self.vy**2)
            text = f"{self.id}:{speed:.1f}"
            id_surf = font.render(text, True, (255, 255, 255))
            id_rect = id_surf.get_rect(center=(int(screen_x), int(screen_y)))
            surface.blit(id_surf, id_rect)
            
            # 可选：绘制前视点（用于调试）
            lookahead_point = self._find_lookahead_point()
            if lookahead_point:
                lx, ly = transform_func(lookahead_point[0], lookahead_point[1])
                pygame.draw.circle(surface, (255, 255, 0), (int(lx), int(ly)), 5)
            
        except Exception as e:
            print(f"绘制车辆 {self.id} 时出错: {e}")
            self.completed = True  # 出错时标记为已完成

    def visualize_route(self, surface, color=(255, 0, 0, 128), transform_func=None):
        """可视化车辆的完整路径（用于调试）
        
        Args:
            surface: 要绘制的表面
            color: 路径颜色 (R,G,B,A)
            transform_func: 坐标转换函数
        """
        if transform_func is None:
            transform_func = lambda x, y: (x, y)
            
        if not self.route_points or len(self.route_points) < 2:
            return
            
        # 创建点列表并应用转换
        points = []
        for point in self.route_points:
            screen_x, screen_y = transform_func(point[0], point[1])
            points.append((int(screen_x), int(screen_y)))
            
        # 绘制线段
        if len(points) > 1:
            # 创建临时surface支持透明度
            temp_surface = pygame.Surface((surface.get_width(), surface.get_height()), pygame.SRCALPHA)
            pygame.draw.lines(temp_surface, color, False, points, 2)
            surface.blit(temp_surface, (0, 0))
            
        # 标记当前目标点
        if self.current_point_index < len(self.route_points):
            target = self.route_points[self.current_point_index]
            tx, ty = transform_func(target[0], target[1])
            pygame.draw.circle(surface, (0, 255, 0), (int(tx), int(ty)), 5)
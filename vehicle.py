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
        self.width = 25
        self.length = 50
        self.wheelbase = 30  # 轴距
        
        # 动态自行车模型状态变量
        self.vx = 0  # 纵向速度
        self.vy = 0  # 横向速度
        self.yaw_rate = 0  # 角速度
        
        # IDM参数
        self.desired_speed = 20  # 期望速度
        self.max_acceleration = 3.0  # 最大加速度
        self.max_deceleration = -3.0  # 最大减速度
        self.time_headway = 1.5  # 时间间距
        self.min_spacing = 5.0  # 最小间距
        self.delta = 4.0  # IDM指数
        
        # 改进的Pure Pursuit参数 - 动态前视距离
        self.lookahead_base = 15      # 基础前视距离 l_0 (像素)
        self.lookahead_gain = 1.5     # 速度增益系数 k (像素/速度单位)
        self.max_lookahead = 100       # 最大前视距离 (像素)
        self.min_lookahead = 10       # 最小前视距离 (像素)
        self.max_steering_angle = math.radians(60)  # 最大转向角
        
        # 车辆动力学参数
        self.mass = 1500  # 质量 kg
        self.Iz = 2000  # 绕z轴转动惯量 kg*m²
        self.Cf = 60000  # 前轮侧偏刚度 N/rad
        self.Cr = 80000  # 后轮侧偏刚度 N/rad
        self.lf = self.wheelbase * 0.6  # 质心到前轴距离
        self.lr = self.wheelbase * 0.4  # 质心到后轴距离
        
        self.completed = False

        # 可视化控制
        self.show_bicycle_model = False  # 控制是否显示自行车模型可视化
        self.show_debug_info = False     # 控制是否显示调试信息
    
    def toggle_bicycle_visualization(self):
        """切换自行车模型可视化显示"""
        self.show_bicycle_model = not self.show_bicycle_model
    
    def toggle_debug_info(self):
        """切换调试信息显示"""
        self.show_debug_info = not self.show_debug_info
    
    def _get_initial_angle(self, direction):
        """根据方向获取初始角度（弧度）"""
        angle_map = {
            'north': math.pi * 1.5,
            'south': math.pi * 0.5,
            'east': math.pi,
            'west': 0
        }
        return angle_map.get(direction, 0)

    def _calculate_dynamic_lookahead(self):
        """计算动态前视距离: l_0 + k*v
    
        Returns:
            float: 动态前视距离
        """
        current_speed = math.sqrt(self.vx**2 + self.vy**2)
        
        # 动态前视距离公式: l_0 + k*v
        dynamic_distance = self.lookahead_base + self.lookahead_gain * current_speed
        
        # 限制在合理范围内
        dynamic_distance = max(self.min_lookahead, 
                              min(self.max_lookahead, dynamic_distance))
        
        return dynamic_distance

    def _find_lookahead_point(self):
        """改进的Pure Pursuit前视点查找算法 - 使用动态前视距离"""
        if not self.route_points or self.current_point_index >= len(self.route_points) - 1:
            return None
        
        # 使用动态前视距离
        lookahead_distance = self._calculate_dynamic_lookahead()
        
        # 确保从当前点索引开始搜索，不会后退
        start_index = max(0, self.current_point_index)
        
        # 从当前点开始寻找前视点
        for i in range(start_index, len(self.route_points)):
            point = self.route_points[i]
            distance = math.sqrt((point[0] - self.x)**2 + (point[1] - self.y)**2)
            
            if distance >= lookahead_distance:
                return point
        
        # 如果没找到足够远的点，返回最后一个点
        return self.route_points[-1]

    def _pure_pursuit_control(self):
        """简化的Pure Pursuit + 角度控制"""
        lookahead_point = self._find_lookahead_point()
        if lookahead_point is None:
            return 0
        
        # 基本Pure Pursuit计算
        dx = lookahead_point[0] - self.x
        dy = lookahead_point[1] - self.y
        local_x = dx * math.cos(self.angle) + dy * math.sin(self.angle)
        local_y = -dx * math.sin(self.angle) + dy * math.cos(self.angle)
        
        ld = math.sqrt(local_x**2 + local_y**2)
        if ld < 1e-6:
            return 0
        
        # 位置控制
        curvature = 2 * local_y / (ld**2)
        position_steering = math.atan(self.wheelbase * curvature)
        
        # 简单的角度控制
        # 计算期望航向角（基于前视点）
        desired_heading = math.atan2(dy, dx)
        heading_error = desired_heading - self.angle
        
        # 角度归一化
        while heading_error > math.pi:
            heading_error -= 2 * math.pi
        while heading_error < -math.pi:
            heading_error += 2 * math.pi
    
        # 角度P控制
        heading_correction = 1.0 * heading_error  # P增益为1.0
    
        # 根据横向误差选择控制策略
        lateral_error = abs(local_y)
        current_speed = math.sqrt(self.vx**2 + self.vy**2)
        
        if lateral_error < 5 and current_speed < 10:  # 小误差且低速时，主要用角度控制
            final_steering = 0.3 * position_steering + 0.7 * heading_correction
        else:  # 其他情况主要用位置控制
            final_steering = 0.8 * position_steering + 0.2 * heading_correction
        
        # 限制转向角
        final_steering = max(-self.max_steering_angle, 
                            min(self.max_steering_angle, final_steering))
        
        return final_steering

    def _idm_longitudinal_control(self, front_vehicle=None):
        """改进的IDM纵向控制 - 考虑转向需求"""
        current_speed = math.sqrt(self.vx**2 + self.vy**2)
        
        # 获取当前转向角
        current_steering = self._pure_pursuit_control()
        
        # 根据转向角调整期望速度
        adjusted_desired_speed = self._get_turn_adjusted_speed(current_steering)
        
        # 计算自由流加速度（使用调整后的期望速度）
        speed_ratio = current_speed / adjusted_desired_speed
        free_acceleration = self.max_acceleration * (1 - speed_ratio**self.delta)
        
        # 如果没有前车，使用自由流加速度
        if front_vehicle is None:
            acceleration = free_acceleration
        else:
            # 正常的IDM前车跟随逻辑
            distance = math.sqrt((front_vehicle.x - self.x)**2 + 
                               (front_vehicle.y - self.y)**2)
            front_speed = math.sqrt(front_vehicle.vx**2 + front_vehicle.vy**2)
            relative_speed = current_speed - front_speed
            
            desired_spacing = (self.min_spacing + 
                             self.time_headway * current_speed +
                             (current_speed * relative_speed) / 
                             (2 * math.sqrt(self.max_acceleration * abs(self.max_deceleration))))
            
            spacing_ratio = desired_spacing / max(distance, 0.1)
            interaction_acceleration = -self.max_acceleration * (spacing_ratio**2)
            acceleration = free_acceleration + interaction_acceleration
        
        # 限制加速度
        acceleration = max(self.max_deceleration, 
                         min(self.max_acceleration, acceleration))
        
        return acceleration

    def _get_turn_adjusted_speed(self, steering_angle):
        """根据转向角调整期望速度"""
        base_speed = self.desired_speed
        
        # 转向角的绝对值（弧度）
        abs_steering = abs(steering_angle)
        
        if abs_steering < math.radians(10):
            # 直线行驶，不调整
            return base_speed
        elif abs_steering < math.radians(15):
            # 轻微转向，轻微减速
            speed_factor = 0.9
        elif abs_steering < math.radians(25):
            # 中等转向，明显减速
            speed_factor = 0.7
        else:
            # 急转弯，大幅减速
            speed_factor = 0.5
        
        return base_speed * speed_factor

    def _dynamic_bicycle_model(self, steering_angle, acceleration, dt):
        """修正的动态自行车模型 - 解决低速转向问题"""
        # 限制时间步长
        dt = min(dt, 0.1)
        
        # 限制转向角
        steering_angle = max(-self.max_steering_angle, min(self.max_steering_angle, steering_angle))
        
        # 当前速度
        speed = math.sqrt(self.vx**2 + self.vy**2)
        
        # 调整低速阈值，并改进低速模型
        if abs(speed) < 3.0:  # 降低阈值到3.0
            # 低速时使用混合模型：运动学 + 简化动力学
            self._low_speed_bicycle_model(steering_angle, acceleration, dt, speed)
        else:
            # 高速时使用完整动力学模型
            self._high_speed_bicycle_model(steering_angle, acceleration, dt)
        
        # 角度归一化
        self.angle = self.angle % (2 * math.pi)
        
        # 数值健康检查
        if (math.isnan(self.x) or math.isnan(self.y) or 
            math.isinf(self.x) or math.isinf(self.y) or
            abs(self.x) > 1e6 or abs(self.y) > 1e6):
            print(f"车辆 {self.id} 坐标异常，重置")
            self._reset_to_path()

    def _low_speed_bicycle_model(self, steering_angle, acceleration, dt, speed):
        """低速自行车模型 - 确保转向响应"""
        
        # 纵向动力学
        self.vx += acceleration * dt
        self.vx = max(0.5, self.vx)  # 保持最小速度
        
        # 低速转向处理 - 关键改进！
        if abs(steering_angle) > 0.01:  # 有转向输入时
            # 使用运动学自行车模型计算角速度
            # 这确保了低速时的转向响应
            if abs(self.vx) > 0.1:
                # 基于阿克曼转向几何
                turning_radius = self.wheelbase / math.tan(steering_angle)
                target_yaw_rate = self.vx / turning_radius
            else:
                target_yaw_rate = 0
            
            # 限制角速度
            max_yaw_rate = math.radians(45)  # 最大45度/秒
            target_yaw_rate = max(-max_yaw_rate, min(max_yaw_rate, target_yaw_rate))
            
            # 平滑过渡到目标角速度（避免突变）
            yaw_rate_error = target_yaw_rate - self.yaw_rate
            yaw_rate_gain = min(5.0, max(1.0, speed + 1.0))  # 速度越低，增益越小
            yaw_acceleration = yaw_rate_gain * yaw_rate_error
            
            # 更新角速度
            self.yaw_rate += yaw_acceleration * dt
            
            # 侧向动力学（简化）
            # 在转向时产生侧向速度
            if abs(target_yaw_rate) > 0.01:
                # 根据转向角和速度计算目标侧向速度
                target_vy = -self.vx * math.sin(steering_angle) * 0.5  # 系数0.5减少侧滑
                vy_error = target_vy - self.vy
                vy_gain = 2.0
                self.vy += vy_gain * vy_error * dt
            else:
                # 无转向时，侧向速度逐渐衰减
                self.vy *= max(0.7, 1.0 - 3.0 * dt)
            
        else:  # 无转向输入时
            # 直线行驶：清零侧向运动
            self.vy *= max(0.8, 1.0 - 5.0 * dt)  # 侧向速度衰减
            self.yaw_rate *= max(0.8, 1.0 - 3.0 * dt)  # 角速度衰减
        
        # 限制状态变量
        self.vy = max(-10, min(10, self.vy))
        self.yaw_rate = max(-math.radians(60), min(math.radians(60), self.yaw_rate))
        
        # 位置和角度更新
        dx = (self.vx * math.cos(self.angle) - self.vy * math.sin(self.angle)) * dt
        dy = (self.vx * math.sin(self.angle) + self.vy * math.cos(self.angle)) * dt
        
        self.x += dx
        self.y += dy
        self.angle += self.yaw_rate * dt

    def _high_speed_bicycle_model(self, steering_angle, acceleration, dt):
        """高速动力学模型 - 保持原有实现"""
        try:
            # 使用 atan2 计算侧偏角，避免除零
            theta_vf = math.atan2(self.vy + self.lf * self.yaw_rate, abs(self.vx))
            theta_vr = math.atan2(self.vy - self.lr * self.yaw_rate, abs(self.vx))
            
            # 轮胎侧偏角
            alpha_f = steering_angle - theta_vf
            alpha_r = -theta_vr  # 后轮无转向角
            
            # 限制侧偏角
            alpha_f = max(-0.5, min(0.5, alpha_f))
            alpha_r = max(-0.5, min(0.5, alpha_r))
            
            # 轮胎侧向力
            Fyf = self.Cf * alpha_f
            Fyr = self.Cr * alpha_r
            
            # 纵向力
            Fx = self.mass * acceleration
            
            # 动力学方程
            ax = (Fx - Fyf * math.sin(steering_angle)) / self.mass + self.vy * self.yaw_rate
            ay = (Fyf * math.cos(steering_angle) + Fyr) / self.mass - self.vx * self.yaw_rate
            yaw_acceleration = (self.lf * Fyf * math.cos(steering_angle) - self.lr * Fyr) / self.Iz
            
            # 数值稳定性限制
            ax = max(-20, min(20, ax))
            ay = max(-20, min(20, ay))
            yaw_acceleration = max(-5, min(5, yaw_acceleration))
            
            # 状态更新
            self.vx += ax * dt
            self.vy += ay * dt
            self.yaw_rate += yaw_acceleration * dt
            
            # 物理约束
            self.vx = max(0.1, self.vx)  # 确保非零速度
            self.vy = max(-20, min(20, self.vy))
            self.yaw_rate = max(-5, min(5, self.yaw_rate))
            
            # 位置和角度更新
            dx = (self.vx * math.cos(self.angle) - self.vy * math.sin(self.angle)) * dt
            dy = (self.vx * math.sin(self.angle) + self.vy * math.cos(self.angle)) * dt
            
            self.x += dx
            self.y += dy
            self.angle += self.yaw_rate * dt
            
        except Exception as e:
            print(f"车辆 {self.id} 高速动力学计算错误: {e}")
            # 降级到低速模型
            self._low_speed_bicycle_model(steering_angle, acceleration, dt, 
                                         math.sqrt(self.vx**2 + self.vy**2))

    def _reset_to_path(self):
        """重置车辆到路径上的合理位置"""
        if self.current_point_index < len(self.route_points):
            self.x, self.y = self.route_points[self.current_point_index]
            self.vx = 1.0  # 设置合理的初始速度
            self.vy = 0
            self.yaw_rate = 0
            # 重新计算角度
            if self.current_point_index < len(self.route_points) - 1:
                next_point = self.route_points[self.current_point_index + 1]
                dx = next_point[0] - self.x
                dy = next_point[1] - self.y
                self.angle = math.atan2(dy, dx)
        else:
            self.completed = True

    def _update_current_point_index(self):
        """改进的路径点索引更新 - 避免频繁跳跃"""
        if self.current_point_index >= len(self.route_points) - 1:
            return
        
        current_pos = (self.x, self.y)
        
        # 使用更大的阈值，避免在路径点附近震荡
        completion_threshold = 20  # 增加完成阈值
        
        # 检查是否应该前进到下一个路径点
        while self.current_point_index < len(self.route_points) - 1:
            current_target = self.route_points[self.current_point_index]
            next_target = self.route_points[self.current_point_index + 1]
            
            # 计算到当前目标点的距离
            dist_to_current = math.sqrt((current_target[0] - self.x)**2 + 
                                       (current_target[1] - self.y)**2)
            
            # 基于距离的判断
            if dist_to_current < completion_threshold:
                self.current_point_index += 1
                continue
            
            # 基于投影的判断（更精确）
            if self._is_past_current_point(current_target, next_target):
                self.current_point_index += 1
                continue
            
            break

    def _is_past_current_point(self, current_target, next_target):
        """检查车辆是否已经超过当前路径点"""
        # 路径段向量
        path_vec_x = next_target[0] - current_target[0]
        path_vec_y = next_target[1] - current_target[1]
        
        # 车辆相对于当前目标点的向量
        vehicle_vec_x = self.x - current_target[0]
        vehicle_vec_y = self.y - current_target[1]
        
        # 如果路径段长度为0，直接返回True
        path_length_sq = path_vec_x**2 + path_vec_y**2
        if path_length_sq < 1e-6:
            return True
        
        # 计算投影系数
        projection = (vehicle_vec_x * path_vec_x + vehicle_vec_y * path_vec_y) / path_length_sq
        
        # 如果投影系数 > 0.7，说明车辆已经明显超过路径点
        return projection > 0.7  # 增加阈值，避免过早切换

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

    def draw_bicycle_model(self, surface, transform_func, scale=1.0):
        """绘制详细的自行车模型可视化"""
        if not self.show_bicycle_model:
            return
            
        try:
            # 车辆中心位置
            center_x, center_y = transform_func(self.x, self.y)
            
            # 计算前后轴位置
            front_axle_x = self.x + self.lf * math.cos(self.angle)
            front_axle_y = self.y + self.lf * math.sin(self.angle)
            rear_axle_x = self.x - self.lr * math.cos(self.angle)
            rear_axle_y = self.y - self.lr * math.sin(self.angle)
            
            # 转换到屏幕坐标
            front_screen_x, front_screen_y = transform_func(front_axle_x, front_axle_y)
            rear_screen_x, rear_screen_y = transform_func(rear_axle_x, rear_axle_y)
            
            # 绘制车身轮廓
            body_length = max(int(self.length * scale), 4)
            body_width = max(int(self.width * scale), 2)
            
            # 车身矩形
            body_surf = pygame.Surface((body_length, body_width), pygame.SRCALPHA)
            pygame.draw.rect(body_surf, (100, 100, 255, 150), body_surf.get_rect(), 2)
            
            # 旋转车身
            angle_degrees = math.degrees(self.angle)
            rotated_body = pygame.transform.rotate(body_surf, -angle_degrees)
            body_rect = rotated_body.get_rect(center=(int(center_x), int(center_y)))
            surface.blit(rotated_body, body_rect.topleft)
            
            # 车轮参数
            wheel_width = max(int(8 * scale), 2)    # 车轮宽度
            wheel_height = max(int(4 * scale), 1)   # 车轮高度
            wheel_offset = max(int(self.width * scale * 0.6), 2)  # 车轮偏移（左右轮的距离）
            
            # 绘制后轮（固定方向，绿色）
            # 左后轮
            left_rear_x = rear_axle_x - wheel_offset * math.sin(self.angle) / scale
            left_rear_y = rear_axle_y + wheel_offset * math.cos(self.angle) / scale
            self._draw_wheel(surface, transform_func, left_rear_x, left_rear_y, 
                            self.angle, wheel_width, wheel_height, (0, 255, 0), scale)
            
            # 右后轮
            right_rear_x = rear_axle_x + wheel_offset * math.sin(self.angle) / scale
            right_rear_y = rear_axle_y - wheel_offset * math.cos(self.angle) / scale
            self._draw_wheel(surface, transform_func, right_rear_x, right_rear_y, 
                            self.angle, wheel_width, wheel_height, (0, 255, 0), scale)
            
            # 绘制前轮（可转向，红色）
            current_steering = self._pure_pursuit_control()
            front_wheel_angle = self.angle + current_steering
            
            # 左前轮
            left_front_x = front_axle_x - wheel_offset * math.sin(self.angle) / scale
            left_front_y = front_axle_y + wheel_offset * math.cos(self.angle) / scale
            self._draw_wheel(surface, transform_func, left_front_x, left_front_y, 
                            front_wheel_angle, wheel_width, wheel_height, (255, 0, 0), scale)
            
            # 右前轮
            right_front_x = front_axle_x + wheel_offset * math.sin(self.angle) / scale
            right_front_y = front_axle_y - wheel_offset * math.cos(self.angle) / scale
            self._draw_wheel(surface, transform_func, right_front_x, right_front_y, 
                            front_wheel_angle, wheel_width, wheel_height, (255, 0, 0), scale)
            
            # 绘制前后轴中心点（用于参考）
            pygame.draw.circle(surface, (255, 0, 0), (int(front_screen_x), int(front_screen_y)), 2)  # 前轴中心（红）
            pygame.draw.circle(surface, (0, 255, 0), (int(rear_screen_x), int(rear_screen_y)), 2)    # 后轴中心（绿）
            
            # 绘制转向指示线（只在转向时显示）
            if abs(current_steering) > 0.01:
                # 前轮转向方向指示
                indicator_length = max(int(20 * scale), 5)
                indicator_end_x = front_axle_x + indicator_length * math.cos(front_wheel_angle) / scale
                indicator_end_y = front_axle_y + indicator_length * math.sin(front_wheel_angle) / scale
                
                indicator_end_screen = transform_func(indicator_end_x, indicator_end_y)
                pygame.draw.line(surface, (255, 255, 0), (int(front_screen_x), int(front_screen_y)), 
                               indicator_end_screen, 2)
            
            # 绘制速度矢量
            speed = math.sqrt(self.vx**2 + self.vy**2)
            if speed > 0.1:
                velocity_length = max(int(speed * scale * 2), 5)
                vel_end_x = self.x + velocity_length * math.cos(self.angle) / scale
                vel_end_y = self.y + velocity_length * math.sin(self.angle) / scale
                vel_end_screen = transform_func(vel_end_x, vel_end_y)
                pygame.draw.line(surface, (0, 255, 255), (int(center_x), int(center_y)), 
                                vel_end_screen, 2)
            
            # 绘制质心
            pygame.draw.circle(surface, (255, 0, 255), (int(center_x), int(center_y)), 3)
            
            # 绘制前视点连线
            lookahead_point = self._find_lookahead_point()
            if lookahead_point:
                lx, ly = transform_func(lookahead_point[0], lookahead_point[1])
                pygame.draw.line(surface, (255, 128, 0), (int(center_x), int(center_y)), 
                               (int(lx), int(ly)), 1)
                pygame.draw.circle(surface, (255, 128, 0), (int(lx), int(ly)), 
                                 max(int(3 * scale), 1))
            
            # 绘制前视距离圆圈（在其他元素之前绘制，作为背景）
            current_lookahead = self._calculate_dynamic_lookahead()
            lookahead_radius = max(int(current_lookahead * scale), 5)
            
            # 创建透明surface绘制前视圆圈
            circle_surf = pygame.Surface((lookahead_radius * 2 + 4, lookahead_radius * 2 + 4), pygame.SRCALPHA)
            pygame.draw.circle(circle_surf, (255, 255, 0, 30), 
                              (lookahead_radius + 2, lookahead_radius + 2), lookahead_radius, 2)
            
            circle_rect = circle_surf.get_rect(center=(int(center_x), int(center_y)))
            surface.blit(circle_surf, circle_rect.topleft)
            
        except Exception as e:
            print(f"绘制自行车模型时出错: {e}")

    def _draw_wheel(self, surface, transform_func, wheel_x, wheel_y, wheel_angle, 
                    wheel_width, wheel_height, color, scale):
        """绘制单个车轮
        
        Args:
            surface: 绘制表面
            transform_func: 坐标转换函数
            wheel_x, wheel_y: 车轮世界坐标
            wheel_angle: 车轮角度（弧度）
            wheel_width, wheel_height: 车轮尺寸
            color: 车轮颜色
            scale: 缩放比例
        """
        try:
            # 转换到屏幕坐标
            screen_x, screen_y = transform_func(wheel_x, wheel_y)
            
            # 创建车轮表面
            wheel_surf = pygame.Surface((wheel_width, wheel_height), pygame.SRCALPHA)
            
            # 绘制车轮（矩形表示轮胎）
            pygame.draw.rect(wheel_surf, color, wheel_surf.get_rect(), 0)
            pygame.draw.rect(wheel_surf, (255, 255, 255), wheel_surf.get_rect(), 1)  # 白色边框
            
            # 旋转车轮
            angle_degrees = math.degrees(wheel_angle)
            rotated_wheel = pygame.transform.rotate(wheel_surf, -angle_degrees)
            wheel_rect = rotated_wheel.get_rect(center=(int(screen_x), int(screen_y)))
            
            # 绘制到表面
            surface.blit(rotated_wheel, wheel_rect.topleft)
            
        except Exception as e:
            print(f"绘制车轮时出错: {e}")
    
    def draw_debug_info(self, surface, transform_func, scale=1.0):
        """绘制调试信息"""
        if not self.show_debug_info:
            return
        
        try:
            center_x, center_y = transform_func(self.x, self.y)
            
            # 创建调试文本
            font_size = max(int(16 * scale), 10)
            font = pygame.font.SysFont('monospace', font_size)
            
            # 计算当前状态
            speed = math.sqrt(self.vx**2 + self.vy**2)
            steering_angle = self._pure_pursuit_control()
            current_lookahead = self._calculate_dynamic_lookahead()
            
            debug_lines = [
                f"ID: {self.id}",
                f"Speed: {speed:.1f}",
                f"vx: {self.vx:.1f}, vy: {self.vy:.1f}",
                f"Steering: {math.degrees(steering_angle):.1f}°",
                f"Yaw Rate: {math.degrees(self.yaw_rate):.1f}°/s",
                f"Angle: {math.degrees(self.angle):.1f}°",
                f"Lookahead: {current_lookahead:.1f}",  # 显示当前前视距离
                f"Point: {self.current_point_index}/{len(self.route_points)-1}",
            ]
            
            # 绘制调试文本背景
            text_height = len(debug_lines) * (font_size + 2)
            text_width = 200
            
            debug_surf = pygame.Surface((text_width, text_height), pygame.SRCALPHA)
            pygame.draw.rect(debug_surf, (0, 0, 0, 180), debug_surf.get_rect())
            
            # 绘制文本
            for i, line in enumerate(debug_lines):
                text_surf = font.render(line, True, (255, 255, 255))
                debug_surf.blit(text_surf, (5, i * (font_size + 2)))
            
            # 定位调试信息（车辆右侧）
            debug_x = int(center_x) + max(int(30 * scale), 20)
            debug_y = int(center_y) - text_height // 2
            
            surface.blit(debug_surf, (debug_x, debug_y))
            
        except Exception as e:
            print(f"绘制调试信息时出错: {e}")
    
    def draw(self, surface, transform_func=None, scale=1.0):
        """在surface上绘制车辆"""
        if self.completed:
            return
    
        # 如果未提供转换函数，则使用恒等映射
        if transform_func is None:
            transform_func = lambda x, y: (x, y)
    
        # 检查坐标是否有效，避免数值溢出
        try:
            if (not isinstance(self.x, (int, float)) or 
                not isinstance(self.y, (int, float)) or
                math.isnan(self.x) or math.isnan(self.y) or
                math.isinf(self.x) or math.isinf(self.y) or
                abs(self.x) > 1e6 or abs(self.y) > 1e6):
                print(f"车辆 {self.id} 坐标无效: x={self.x}, y={self.y}, vx={self.vx}, vy={self.vy}")
                self.completed = True
                return
            
            # 绘制基本车辆
            screen_x, screen_y = transform_func(self.x, self.y)
            scaled_length = max(int(self.length * scale), 1)
            scaled_width = max(int(self.width * scale), 1)
        
            car_surf = pygame.Surface((scaled_length, scaled_width), pygame.SRCALPHA)
            pygame.draw.rect(car_surf, (0, 0, 255, 200), car_surf.get_rect(), 0, 
                           border_radius=max(1, int(5 * scale)))
        
            angle_degrees = math.degrees(self.angle)
            rotated_car = pygame.transform.rotate(car_surf, -angle_degrees)
            rotated_rect = rotated_car.get_rect(center=(int(screen_x), int(screen_y)))
            surface.blit(rotated_car, rotated_rect.topleft)
        
            # 绘制车辆ID和速度
            font_size = max(int(20 * scale), 8)
            font = pygame.font.SysFont(None, font_size)
            speed = math.sqrt(self.vx**2 + self.vy**2)
            text = f"{self.id}:{speed:.1f}"
            id_surf = font.render(text, True, (255, 255, 255))
            id_rect = id_surf.get_rect(center=(int(screen_x), int(screen_y)))
            surface.blit(id_surf, id_rect)
        
            # 绘制前视点
            lookahead_point = self._find_lookahead_point()
            if lookahead_point:
                lx, ly = transform_func(lookahead_point[0], lookahead_point[1])
                circle_radius = max(int(5 * scale), 1)
                pygame.draw.circle(surface, (255, 255, 0), (int(lx), int(ly)), circle_radius)
            
            # 绘制自行车模型可视化
            self.draw_bicycle_model(surface, transform_func, scale)
            
            # 绘制调试信息
            self.draw_debug_info(surface, transform_func, scale)
        
        except Exception as e:
            print(f"绘制车辆 {self.id} 时出错: {e}")
            self.completed = True

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
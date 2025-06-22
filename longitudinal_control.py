# longitudinal_control.py

import numpy as np
from config import * # 从配置文件中导入所有参数

class GippsModel:
    """
    实现Gipps跟驰模型，用于计算安全的跟车速度作为目标。
    """
    def __init__(self):
        # 从config加载Gipps模型自身的参数
        self.a = GIPPS_A                # 期望加速度
        self.b = GIPPS_B                # 期望减速度
        self.V_desired = GIPPS_V_DESIRED # 期望速度
        self.s0 = GIPPS_S0              # 车辆静止时的安全距离（有效车长）
        self.tau = GIPPS_TAU            # 反应时间
        self.b_hat = GIPPS_B_HAT        # 对前车减速度的估计

    def calculate_target_speed(self, ego_vehicle_state, lead_vehicle_state=None):
        """
        计算下一时刻的目标速度。

        Args:
            ego_vehicle_state (dict): 自车状态字典。
            lead_vehicle_state (dict, optional): 前车状态字典。如果为None，表示前方无车。

        Returns:
            float: 计算出的目标速度 (m/s)。
        """
        v_ego = ego_vehicle_state['vx']

        # --- 自由加速部分 ---
        # 车辆在无约束时，倾向于加速到期望速度
        v_accel = v_ego + 2.5 * self.a * self.tau * (1 - v_ego / self.V_desired) * np.sqrt(0.025 + v_ego / self.V_desired)

        # 如果没有前车，直接返回自由加速计算的速度
        if lead_vehicle_state is None:
            return min(v_accel, self.V_desired)

        # --- 跟驰刹车部分 ---
        # 为保证安全，必须满足的刹车速度约束
        x_ego = ego_vehicle_state['x']
        v_lead = lead_vehicle_state['vx']
        x_lead = lead_vehicle_state['x']
        
        gap = x_lead - x_ego - self.s0
        # 确保gap为正，避免在车辆并排或重叠时出现数学错误
        if gap < 0: gap = 0.01

        # 计算根号内的项
        sqrt_term = self.b**2 * self.tau**2 - self.b * (2 * gap - v_ego * self.tau - v_lead**2 / self.b_hat)

        # 如果根号内为负，意味着非常安全，刹车约束不起作用，可以忽略
        if sqrt_term < 0:
            v_brake = float('inf') # 设为一个极大值
        else:
            v_brake = -self.b * self.tau + np.sqrt(sqrt_term)
        
        # 最终速度是两者中的较小值，且不超过期望速度
        return min(v_accel, v_brake, self.V_desired)


class PIDController:
    """
    一个标准的PID控制器，用于追踪目标速度。
    """
    def __init__(self):
        # 从config加载PID参数
        self.Kp = PID_KP
        self.Ki = PID_KI
        self.Kd = PID_KD
        
        # 初始化误差项
        self.integral_error = 0.0
        self.last_error = 0.0
        
        # 设置一个积分抗饱和的阈值，防止积分项无限增大
        self.integral_max = PID_INTEGRAL_MAX 

    def step(self, target_speed, current_speed, dt):
        """
        执行一步PID计算。

        Args:
            target_speed (float): 目标速度 (m/s)。
            current_speed (float): 当前速度 (m/s)。
            dt (float): 时间步长 (s)。

        Returns:
            float: 计算出的控制力 (N)，正为油门，负为刹车。
        """
        error = target_speed - current_speed
        
        # 积分项，带抗饱和
        self.integral_error += error * dt
        self.integral_error = np.clip(self.integral_error, -self.integral_max, self.integral_max)
        
        # 微分项
        derivative_error = (error - self.last_error) / dt
        
        # 更新上一次误差
        self.last_error = error
        
        # 计算总输出
        output = self.Kp * error + self.Ki * self.integral_error + self.Kd * derivative_error
        
        return output
# longitudinal_control.py

import numpy as np
from config import *

class PIDController:
    """
    一个标准的PID控制器，用于追踪目标速度。
    """
    def __init__(self):
        self.Kp = PID_KP
        self.Ki = PID_KI
        self.Kd = PID_KD
        self.integral_error = 0.0
        self.last_error = 0.0
        self.integral_max = PID_INTEGRAL_MAX 

    def step(self, target_speed, current_speed, dt):
        """
        执行一步PID计算，返回控制力（正为油门，负为刹车）。
        """
        error = target_speed - current_speed
        self.integral_error += error * dt
        self.integral_error = np.clip(self.integral_error, -self.integral_max, self.integral_max)
        derivative_error = (error - self.last_error) / dt
        self.last_error = error
        output = self.Kp * error + self.Ki * self.integral_error + self.Kd * derivative_error
        return output
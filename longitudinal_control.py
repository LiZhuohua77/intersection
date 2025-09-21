"""
@file: longitudinal_control.py
@description:
该文件实现了一个标准的 PID (比例-积分-微分) 控制器，专门用于车辆的纵向速度控制。
它负责根据上层规划模块提供的目标速度，计算出驱动车辆加速或减速所需的具体控制力
（即油门或刹车力）。

主要函数:
1. PIDController.__init__(): 初始化PID控制器，设置各项参数和初始状态
2. PIDController.step(): 核心控制函数，根据目标速度和当前速度计算控制输出

核心组件与原理:

1.  **PIDController (PID控制器类):**
    该控制器通过不断减小"目标速度"与"当前速度"之间的误差来实现精确的速度追踪。
    它由三个部分协同工作：

    - **比例 (Proportional, P):**
      `Kp * error` - 核心作用是纠正**当前**的误差。误差越大，施加的控制力就越强。
      它是控制器响应的主要部分。

    - **积分 (Integral, I):**
      `Ki * integral_error` - 核心作用是消除系统的**稳态误差**。通过累积**过去**的
      所有误差，它可以补偿那些仅靠P控制器无法完全克服的持续性小误差（例如，持续的
      空气阻力或坡度阻力），确保车辆最终能精确达到目标速度。

    - **微分 (Derivative, D):**
      `Kd * derivative_error` - 核心作用是预测**未来**的误差趋势，并提供阻尼以
      **抑制过冲**。通过观察误差的变化速率，它可以在车辆速度快速接近目标值时
      提前"踩刹车"，防止速度超出目标太多，使整个调节过程更加平稳、稳定。

2.  **关键实现细节:**
    - **积分抗饱和 (Integral Anti-Windup):** 在实现中，通过 `np.clip` 对积分项
      `integral_error` 的累积值进行了限制。这是一个至关重要的工程实践，可以防止
      在长时间存在较大误差时积分项过度累积（即"饱和"），从而避免当误差反向时
      系统因巨大的积分惯性而产生剧烈的超调。

接口与交互:
- **`step` 方法:** 这是控制器唯一的执行接口，输入目标速度和当前速度，输出纵向控制力（正值代表油门，负值代表刹车）。
- **系统层级:** 该PID控制器位于车辆控制架构的最底层，将上层规划器的速度指令转化为可以直接作用于车辆的物理力。
"""

import numpy as np
from config import *

class PIDController:
    """
    一个标准的PID控制器，用于追踪目标速度。
    实现了比例-积分-微分控制策略，带有积分项抗饱和处理。
    """
    def __init__(self):
        """
        初始化PID控制器。
        
        设置PID控制器的各项参数和初始状态：
        - Kp: 比例增益，决定对当前误差的响应强度
        - Ki: 积分增益，用于消除稳态误差
        - Kd: 微分增益，提供阻尼以减少过冲和震荡
        - integral_error: 误差积分项，用于累积历史误差
        - last_error: 上一步的误差，用于计算误差变化率
        - integral_max: 积分项上限，防止积分饱和
        """
        self.Kp = PID_KP              # 比例增益
        self.Ki = PID_KI              # 积分增益
        self.Kd = PID_KD              # 微分增益
        self.integral_error = 0.0      # 误差积分累积
        self.last_error = 0.0         # 上一步的误差
        self.integral_max = PID_INTEGRAL_MAX  # 积分项饱和上限

    def step(self, target_speed, current_speed, dt):
        """
        执行一步PID计算，返回控制力（正为油门，负为刹车）。
        
        参数:
            target_speed (float): 目标速度，由上层规划器提供
            current_speed (float): 车辆当前速度
            dt (float): 时间步长，用于积分和微分计算
            
        返回:
            float: 计算得到的纵向控制力（牛顿），正值表示油门，负值表示刹车
        """
        error = target_speed - current_speed
        self.integral_error += error * dt
        self.integral_error = np.clip(self.integral_error, -self.integral_max, self.integral_max)
        derivative_error = (error - self.last_error) / dt
        self.last_error = error
        output = self.Kp * error + self.Ki * self.integral_error + self.Kd * derivative_error
        return output
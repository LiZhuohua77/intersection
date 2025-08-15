"""
@file: longitudinal_control.py
@description:
该文件实现了一个标准的 PID (比例-积分-微分) 控制器，专门用于车辆的纵向速度控制。
它的核心职责是作为一个底层的速度追踪执行器：根据上层规划模块（如 `LongitudinalPlanner`
或 `GippsModel`）给出的期望目标速度，计算出驱动车辆加速或减速所需的具体控制力
（即油门或刹车力）。

核心组件与原理:

1.  **PIDController (PID控制器类):**
    该控制器通过不断减小“目标速度”与“当前速度”之间的误差来实现精确的速度追踪。
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
      提前“踩刹车”，防止速度超出目标太多，使整个调节过程更加平稳、稳定。

2.  **关键实现细节:**
    - **积分抗饱和 (Integral Anti-Windup):** 在实现中，通过 `np.clip` 对积分项
      `integral_error` 的累积值进行了限制。这是一个至关重要的工程实践，可以防止
      在长时间存在较大误差时积分项过度累积（即“饱和”），从而避免当误差反向时
      系统因巨大的积分惯性而产生剧烈的超调。

接口与交互:
- **`step` 方法:** 这是控制器唯一的执行接口。它接收 `目标速度`、`当前速度` 和
  `时间步长 dt` 作为输入，并输出一个标量值，代表需要施加在车辆上的**纵向控制力**
  （单位：牛顿）。正值代表油门，负值代表刹车。
- **系统层级:** 该PID控制器位于车辆控制架构的最底层，它将上层规划器给出的抽象的
  “速度指令”转化为可以直接作用于车辆动力学模型的“物理力”。
"""

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
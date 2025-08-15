"""
@file: config.py
@description:
该文件包含了整个强化学习仿真项目的所有全局配置参数和常量。
通过修改此文件中的值，可以方便地调整仿真环境、车辆物理特性、控制器行为以及智能体（agent）的能力，
而无需改动核心的仿真和控制逻辑代码。

各参数块说明:
- 仿真参数 (Simulation Parameters): 控制仿真器本身的基本设置，如时间步长和渲染窗口大小。
- 车辆物理参数 (Vehicle Physical Parameters): 定义了车辆的动力学模型属性，这些是仿真真实性的基础。
- PID 控制器参数 (PID Controller Parameters): 用于车辆纵向（速度）控制的PID控制器调优参数。
- MPC 控制器参数 (MPC Controller Parameters): 用于车辆横向（路径跟踪）控制的模型预测控制器（MPC）调优参数。
- Gipps 模型参数 (Gipps' Model Parameters): 定义了背景车辆（非Agent车辆）的行为，它们遵循Gipps跟驰模型。
- Agent 参数 (Agent Parameters): 定义了强化学习智能体的行为约束和感知能力。
"""

import numpy as np

# 仿真参数
SIMULATION_DT = 0.05  # 仿真步长 (s), 建议小一些以保证MPC稳定性
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800

# 车辆物理参数
VEHICLE_MASS = 1500  # kg
VEHICLE_LF = 1.4     # 前轴到质心距离 (m)
VEHICLE_LR = 1.5     # 后轴到质心距离 (m)
VEHICLE_W = 2.0      # 车辆宽度 (m)
VEHICLE_L = 5.0      # 车辆长度 (m)
VEHICLE_IZ = 2500    # 绕Z轴的转动惯量 (kg*m^2)
VEHICLE_CF = 80000   # 前轮侧偏刚度 (N/rad)
VEHICLE_CR = 80000   # 后轮侧偏刚度 (N/rad)
LOW_SPEED_THRESHOLD = 0.1 # m/s

# PID 控制器参数
PID_KP = 1000.0  # 比例增益 (将速度误差m/s转换为力N)
PID_KI = 50.0
PID_KD = 100.0
PID_INTEGRAL_MAX = 5000.0 # 积分抗饱和上限

# MPC 控制器参数
MPC_HORIZON = 15         # 预测时域 N
MPC_CONTROL_HORIZON = 5  # 控制时域 M
MPC_Q = [10.0, 180.0]     # 状态误差权重 [y_e, psi_e]
MPC_R = [10.0]            # 控制输入权重 [delta]
MPC_RD = [150.0]          # 控制输入变化率权重 [delta_dot]
MAX_STEER_ANGLE = np.deg2rad(30.0) # 最大方向盘转角

# Gipps 模型参数
GIPPS_A = 1.8               # m/s^2, 期望加速度
GIPPS_B = -3.5              # m/s^2, 期望减速度
GIPPS_V_DESIRED = 15.0      # m/s, 期望速度 (~54 km/h)
GIPPS_S0 = VEHICLE_L        # m, 车辆静止时的安全距离（等于车长）
GIPPS_TAU = 1.0             # s, 驾驶员反应时间
GIPPS_B_HAT = -3.0          # m/s^2, 对前车减速度的估计

# agent参数
MAX_ACCELERATION = 3.0  # agent可以输出的最大加速度 (m/s^2)
MAX_STEERING_ANGLE = np.deg2rad(30) # agent可以输出的最大转向角 (弧度)
OBSERVATION_RADIUS = 50.0 # 观测周围车辆的半径 (米)
NUM_OBSERVED_VEHICLES = 5 # 最多观测周围5辆车
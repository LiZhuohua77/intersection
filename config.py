"""
@file: config.py
@description: 
本文件包含整个交通仿真与强化学习项目的全局配置参数和常量。
该文件不包含函数定义，而是通过常量组织的方式提供以下配置类别：

1. 仿真参数 (Simulation Parameters):
   控制仿真器的基本设置，如时间步长(SIMULATION_DT)和渲染窗口尺寸。

2. 车辆物理参数 (Vehicle Physical Parameters):
   定义车辆动力学模型属性，包括质量、尺寸、转动惯量、轮胎特性等。
   还包括空气动力学参数和能量转换效率参数，用于功率消耗计算。

3. PID 控制器参数 (PID Controller Parameters):
   用于纵向(速度)控制的PID控制器参数，包括比例、积分、微分增益。

4. MPC 控制器参数 (MPC Controller Parameters):
   用于横向(路径跟踪)的模型预测控制器参数，包括预测时域和权重矩阵。

5. Gipps 模型参数 (Gipps' Model Parameters):
   定义背景车辆行为的参数，基于Gipps跟驰模型实现真实的交通流仿真。

6. Agent 参数 (Agent Parameters):
   定义强化学习智能体的行为约束和感知能力，如最大加速度和观测半径。

通过修改此文件中的值，可以调整仿真环境的各个方面而无需改动核心代码逻辑。
"""

import numpy as np

# 仿真参数
SIMULATION_DT = 0.05  # 仿真步长 (s), 建议小一些以保证MPC稳定性
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800

# 车辆物理参数
VEHICLE_MASS = 1500  # kg
VEHICLE_LF = 1.8     # 前轴到质心距离 (m)
VEHICLE_LR = 1.8     # 后轴到质心距离 (m)
VEHICLE_W = 2.0      # 车辆宽度 (m)
VEHICLE_L = 5.0      # 车辆长度 (m)
VEHICLE_IZ = 2500    # 绕Z轴的转动惯量 (kg*m^2)
VEHICLE_CF = 60000   # 前轮侧偏刚度 (N/rad)
VEHICLE_CR = 60000   # 后轮侧偏刚度 (N/rad)
WHEEL_RADIUS = 0.33

GEAR_RATIO = 8.5
DRIVETRAIN_EFFICIENCY = 0.95 # 传动系统效率（从电机到车轮）
REGEN_EFFICIENCY = 0.7

LOW_SPEED_THRESHOLD = 0.5 # m/s

AIR_DENSITY = 1.225
DRAG_COEFFICIENT = 0.3
FRONTAL_AREA = 2.2
ROLLING_RESISTANCE_COEFFICIENT = 0.015
GRAVITATIONAL_ACCEL = 9.81

# PID 控制器参数
PID_KP = 12000.0  # 比例增益 (将速度误差m/s转换为力N)
PID_KI = 800
PID_KD = 100
PID_INTEGRAL_MAX = 5000.0 # 积分抗饱和上限

# MPC 控制器参数
MPC_HORIZON = 20         # 预测时域 N
MPC_CONTROL_HORIZON = 5  # 控制时域 M
MPC_Q = [10.0, 180.0]     # 状态误差权重 [y_e, psi_e]
MPC_R = [10.0]            # 控制输入权重 [delta]
MPC_RD = [150.0]          # 控制输入变化率权重 [delta_dot]
MAX_STEER_ANGLE = np.deg2rad(30.0) # 最大方向盘转角

# Gipps 模型参数
GIPPS_A = 1.8               # m/s^2, 期望加速度
GIPPS_B = -3.5              # m/s^2, 期望减速度
GIPPS_V_DESIRED = 12.5      # m/s, 期望速度 (~54 km/h)
GIPPS_S0 = VEHICLE_L        # m, 车辆静止时的安全距离（等于车长）
GIPPS_TAU = 1.0             # s, 驾驶员反应时间
GIPPS_B_HAT = -3.0          # m/s^2, 对前车减速度的估计

# agent参数
MAX_ACCELERATION = 3.0  # agent可以输出的最大加速度 (m/s^2)
MAX_STEERING_ANGLE = np.deg2rad(0) # agent可以输出的最大转向角 (弧度)
OBSERVATION_RADIUS = 80.0 # 观测周围车辆的半径 (米)
NUM_OBSERVED_VEHICLES = 1 # 最多观测周围5辆车

MAX_RELEVANT_CTE = 15.0 # 最大相关横向误差

AGGRESSIVE_PROB = 0.5 # 背景车辆中激进驾驶员的比例
SCENARIO_TREE_LENGTH = 2.4 # 场景树的长度 (时间步数)，例如2.4秒


OBSERVATION_DIM = 6 + 4 * NUM_OBSERVED_VEHICLES + 2*int(SCENARIO_TREE_LENGTH/(4 * SIMULATION_DT))# 6 (自身状态) + 4 * N (每辆车的状态)
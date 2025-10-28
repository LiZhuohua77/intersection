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
PID_KP = 12000.0 
PID_KI = 800
PID_KD = 100
PID_INTEGRAL_MAX = 5000.0 # 积分抗饱和上限

# MPC 控制器参数
MPC_HORIZON = 25         # 预测时域 N
MPC_CONTROL_HORIZON = 5  # 控制时域 M
MPC_Q = [50.0, 50.0]     # 状态误差权重 [y_e, psi_e]
MPC_R = [10.0]            # 控制输入权重 [delta]
MPC_RD = [150.0]          # 控制输入变化率权重 [delta_dot]

# IDM 驾驶员模型参数
# v0: 期望速度 (m/s)
# T: 安全时间间隔 (s) - 越小越激进
# a: 最大加速度 (m/s^2)
# b: 舒适减速度 (m/s^2) - 绝对值越大越能接受急刹车
# s0: 最小安全车头间距 (m)
IDM_PARAMS = {
    'AGGRESSIVE':   {'v0': 15.0, 'T': 1.2, 'a': 2.5, 'b': 3.0, 's0': 2.0},
    'NORMAL':       {'v0': 13.0, 'T': 1.6, 'a': 1.5, 'b': 2.0, 's0': 2.0},
    'CONSERVATIVE': {'v0': 11.0,  'T': 2.2, 'a': 1.0, 'b': 1.5, 's0': 2.5},
}

# === 5. [新增] 交叉口与环境参数 ===
INTERACTION_ZONE_RADIUS = 60.0  # 交叉口交互区域的半径 (米)

# === 6. [修改] 强化学习智能体与观测空间参数 ===
MAX_VEHICLES = 4
# Agent 物理极限
MAX_ACCELERATION = 3.0
MAX_STEERING_ANGLE = np.deg2rad(30.0) # 修正: 原有MPC和Agent参数中都有定义，统一在此

# 观测范围
OBSERVATION_RADIUS = 80.0
NUM_OBSERVED_VEHICLES = 1
MAX_RELEVANT_CTE = 15.0

# [新增] 预测性场景树参数
PREDICTION_HORIZON = 40        # 预测模块向前看的时间步数 (40步 * 0.05s/步 = 2秒)
FEATURES_PER_STEP = 3          # 每个预测时间步的特征数量 (例如: 相对x, 相对y, 速度v)

# [新增] 观测空间各部分维度定义
AV_OBS_DIM = 6  # 智能体自身状态维度 (vx, vy, psi_dot, cte, he, path_completion)
HV_OBS_DIM = 4 * NUM_OBSERVED_VEHICLES # 每个背景车辆的相对状态维度 (rel_x, rel_y, rel_vx, rel_vy)

# [修改] 自动计算最终的观测空间总维度
# 移除了旧的、手动的OBSERVATION_DIM计算方式
TOTAL_OBS_DIM = AV_OBS_DIM + HV_OBS_DIM + (2 * PREDICTION_HORIZON * FEATURES_PER_STEP)

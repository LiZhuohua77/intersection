# config.py

import numpy as np

# 仿真参数
SIMULATION_DT = 0.05  # 仿真步长 (s), 建议小一些以保证MPC稳定性
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800

# 车辆物理参数
VEHICLE_MASS = 1500  # kg
VEHICLE_LF = 2.5     # 前轴到质心距离 (m)
VEHICLE_LR = 2.5     # 后轴到质心距离 (m)
VEHICLE_W = 2.0      # 车辆宽度 (m)
VEHICLE_L = 5.0      # 车辆长度 (m)
VEHICLE_IZ = 2500    # 绕Z轴的转动惯量 (kg*m^2)
VEHICLE_CF = 80000   # 前轮侧偏刚度 (N/rad)
VEHICLE_CR = 80000   # 后轮侧偏刚度 (N/rad)

# PID 控制器参数
PID_KP = 1000.0  # 比例增益 (将速度误差m/s转换为力N)
PID_KI = 50.0
PID_KD = 100.0
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
GIPPS_V_DESIRED = 12.0      # m/s, 期望速度 (~54 km/h)
GIPPS_S0 = VEHICLE_L        # m, 车辆静止时的安全距离（等于车长）
GIPPS_TAU = 1.0             # s, 驾驶员反应时间
GIPPS_B_HAT = -3.0          # m/s^2, 对前车减速度的估计
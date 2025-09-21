"""
@file: analysis.py
@description: 
本文件用于分析和可视化车辆在不同速度和加速度工况下的功率消耗特性。
主要包含以下功能：
1. analyze_power_map(): 生成车辆速度-加速度-功率消耗热力图，用于直观展示车辆在
   不同工况下的能量需求特性，包括驱动模式和再生制动模式下的电功率计算。
"""

import numpy as np
import matplotlib.pyplot as plt
from config import *
from vehicle import calculate_spm_loss_kw

def analyze_power_map():
    """
    分析车辆在不同速度和加速度组合下的瞬时功率消耗并生成热力图。
    
    功能说明:
    1. 创建速度(0-35m/s)和加速度(-MAX_ACCELERATION到MAX_ACCELERATION)的二维网格
    2. 计算每个网格点的功率消耗:
       - 计算合成牵引力(惯性力+风阻+滚阻)
       - 转换为电机扭矩和转速
       - 根据驱动/制动模式计算电功率
    3. 绘制功率热力图，包括等高线和相关标注
    
    返回:
    无 - 直接显示生成的热力图
    """
    # --- 1. 定义分析范围 ---
    # 速度范围: 0 到 35 m/s (约 126 km/h), 分成 100 个点
    speeds_ms = np.linspace(0, 35, 100)

    # 加速度范围: 从最大制动到最大加速, 分成 100 个点
    accelerations = np.linspace(-MAX_ACCELERATION, MAX_ACCELERATION, 100)

    # 创建网格
    vx_grid, accel_grid = np.meshgrid(speeds_ms, accelerations)
    
    # 初始化用于存储功率结果的网格
    power_grid_kw = np.zeros_like(vx_grid)

    # --- 2. 遍历网格，计算每个点的功率消耗 ---
    for i in range(vx_grid.shape[0]):
        for j in range(vx_grid.shape[1]):
            vx = vx_grid[i, j]
            accel = accel_grid[i, j]

            # a. 计算总牵引力 (需要克服惯性、风阻、滚阻)
            F_inertial = VEHICLE_MASS * accel
            F_aero = 0.5 * DRAG_COEFFICIENT * FRONTAL_AREA * AIR_DENSITY * (vx**2)
            F_roll = ROLLING_RESISTANCE_COEFFICIENT * VEHICLE_MASS * GRAVITATIONAL_ACCEL
            F_tractive = F_inertial + F_aero + F_roll

            # b. 转换为电机扭矩和转速
            motor_torque_nm = (F_tractive * WHEEL_RADIUS) / (GEAR_RATIO * DRIVETRAIN_EFFICIENCY)
            
            if vx > 0:
                motor_speed_rad_s = (vx / WHEEL_RADIUS) * GEAR_RATIO
                motor_speed_rpm = motor_speed_rad_s * (60 / (2 * np.pi))
            else:
                motor_speed_rpm = 0
                motor_speed_rad_s = 0

            # c. 根据驱动/制动工况，计算电功率 P_elec
            if motor_torque_nm >= 0:
                # --- 驱动模式 ---
                P_mech_kW = (motor_torque_nm * motor_speed_rad_s) / 1000.0
                P_loss_kW = calculate_spm_loss_kw(motor_torque_nm, motor_speed_rpm)
                P_elec_kW = P_mech_kW + P_loss_kW
            else:
                # --- 再生制动模式 ---
                P_mech_kW = (motor_torque_nm * motor_speed_rad_s) / 1000.0
                # P_elec_kW 此时为负，代表能量回到电池
                P_elec_kW = P_mech_kW * REGEN_EFFICIENCY
            
            power_grid_kw[i, j] = P_elec_kW

    # --- 3. 绘图 ---
    plt.figure(figsize=(12, 8))
    
    # 使用 pcolormesh 绘制热力图
    max_abs_power = np.max(np.abs(power_grid_kw))
    contour = plt.pcolormesh(
        speeds_ms,  # <-- 修改: 使用 m/s 为横轴
        accelerations, 
        power_grid_kw, 
        shading='auto', 
        cmap='RdYlBu_r',
        vmin=-max_abs_power/2,
        vmax=max_abs_power
    )

    # 添加等高线
    levels = np.arange(round(-max_abs_power/2), round(max_abs_power), 10)
    CS = plt.contour(speeds_ms, accelerations, power_grid_kw, levels=levels, colors='black', linewidths=0.5) # <-- 修改: 使用 m/s
    plt.clabel(CS, inline=True, fontsize=8, fmt='%1.0f')

    # 添加颜色条
    cbar = plt.colorbar(contour)
    cbar.set_label('瞬时电功率 (kW)')

    # 添加坐标轴标签和标题
    plt.xlabel('车速 (m/s)') # <-- 修改: 更改坐标轴标签
    plt.ylabel('加速度 (m/s²)')
    plt.title('车辆瞬时功率消耗图 (速度 vs 加速度)')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 突出显示匀速行驶线 (加速度=0)
    plt.axhline(0, color='black', linewidth=1.5, linestyle='-')
    plt.text(2, 0.1, '匀速行驶线 (a=0)', color='black', va='bottom') # <-- 调整了文本位置以适应新坐标系

    plt.show()


if __name__ == '__main__':
    """
    主程序入口：配置matplotlib支持中文显示并执行功率地图分析
    """
    # 确保你的环境中安装了 matplotlib: pip install matplotlib
    # 设置matplotlib支持中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    analyze_power_map()
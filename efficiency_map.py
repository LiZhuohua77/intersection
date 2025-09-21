"""
@file: efficiency_map.py
@description: 
本文件用于生成和可视化永磁同步电机(SPM)的效率图和损耗图。
主要功能包括：
1. 基于多项式模型计算电机在不同工况点(转速、转矩)的功率损耗
2. 生成电机完整的效率分布图和损耗分布图
3. 考虑电机物理约束(如最大转矩限制和功率限制)
4. 将结果保存为高质量图像文件

实现方式：
1. calculate_spm_loss_kw(): 计算SPM电机在给定转矩和转速下的功率损耗
2. 主脚本部分：创建操作网格，计算效率和损耗，并绘制可视化图表
"""

import numpy as np
import matplotlib.pyplot as plt

def calculate_spm_loss_kw(T_inst_nm, n_inst_rpm):
    """
    计算SPM电机在给定工况点(转矩和转速)下的功率损耗(kW)。
    
    基于文献中的多项式损耗模型，使用归一化参数和系数计算。
    该函数支持矢量化计算，可同时处理整个操作网格。
    
    参数:
        T_inst_nm: float或ndarray, 瞬时转矩(Nm)
        n_inst_rpm: float或ndarray, 瞬时转速(rpm)
        
    返回:
        float或ndarray: 对应工况点的功率损耗(kW)，负值被裁剪为0.01
    """
    # 1. Define base values from the paper
    T_base_nm = 250.0
    n_base_rpm = 12000.0
    P_loss_base_kW = 8.0

    # 2. Normalize inputs
    n_inst_krpm = n_inst_rpm / 1000.0
    T_norm = T_inst_nm / T_base_nm
    n_norm = n_inst_krpm / (n_base_rpm / 1000.0)

    # 3. Use Table III coefficients for SPM to calculate normalized loss
    loss_norm = (
        -0.002
        + 0.175 * n_norm
        + 0.181 * (n_norm**2)
        + 0.443 * (n_norm**3)
        - 0.065 * T_norm
        + 0.577 * T_norm * n_norm
        - 0.542 * T_norm * (n_norm**2)
        + 0.697 * (T_norm**2)
        - 1.043 * (T_norm**2) * n_norm
        + 0.942 * (T_norm**3)
    )

    # 4. Denormalize loss to get kW
    loss_kW = loss_norm * P_loss_base_kW
    
    # The model can produce negative losses at very low torque/speed, which is unphysical.
    # We clip the loss at a small positive value.
    return np.maximum(0.01, loss_kW)

# --- 1. Create the operating grid ---
# Define the speed and torque ranges based on the paper's plots
speed_rpm = np.linspace(1, 12000, 100) # From 1 to avoid division by zero
torque_nm = np.linspace(1, 240, 100)

# Create meshgrid for vectorized calculations
T_grid, N_grid = np.meshgrid(torque_nm, speed_rpm)

# --- 2. Calculate Loss and Efficiency across the grid ---
# Calculate loss for every point on the grid
loss_map_kw = calculate_spm_loss_kw(T_grid, N_grid)

# Calculate output power for every point on the grid
# P_out(kW) = T(Nm) * n(rpm) * 2*pi / 60 / 1000
P_out_map_kw = T_grid * N_grid * np.pi / 30000.0

# Calculate input power and efficiency
P_in_map_kw = P_out_map_kw + loss_map_kw
efficiency_map = P_out_map_kw / P_in_map_kw

# We need to handle the motor's physical limits. The torque capability
# decreases at high speeds (constant power region). A simple approximation
# is to cap the power. Let's assume a 50 kW rated power.
power_limit_kw = 50
max_torque_at_speed = (power_limit_kw * 30000.0 / np.pi) / N_grid
# The overload torque from the paper's Table II is 150 Nm
max_torque_at_speed = np.minimum(max_torque_at_speed, 150)

# Mask out the areas where the torque is beyond the motor's capability
invalid_mask = T_grid > max_torque_at_speed
efficiency_map[invalid_mask] = np.nan # Set to Not a Number to hide it in the plot
loss_map_kw[invalid_mask] = np.nan

# --- 3. Plot the maps ---
plt.style.use('seaborn-v0_8-whitegrid')

# --- 绘制损耗图 ---
fig1, ax1 = plt.subplots(figsize=(8, 6))
loss_levels = [200, 500, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000]
loss_levels_kw = [l/1000.0 for l in loss_levels] # Convert W to kW for levels
cp1 = ax1.contourf(N_grid, T_grid, loss_map_kw, levels=loss_levels_kw, cmap='viridis_r', extend='max')
cl1 = ax1.contour(N_grid, T_grid, loss_map_kw, levels=loss_levels_kw, colors='black', linewidths=0.5)
ax1.clabel(cl1, fmt='%1.1f', fontsize=9)
fig1.colorbar(cp1, ax=ax1, label='Power Loss [kW]')
ax1.set_xlabel('Speed [rpm]', fontsize=12)
ax1.set_ylabel('Torque [Nm]', fontsize=12)
ax1.set_ylim(0, 150)
plt.tight_layout()
plt.savefig('spm_loss_map.png')


# --- 绘制效率图 ---
fig2, ax2 = plt.subplots(figsize=(8, 6))
efficiency_levels_percent = np.arange(70, 100, 2.5)
efficiency_levels = efficiency_levels_percent / 100.0
cp2 = ax2.contourf(N_grid, T_grid, efficiency_map, levels=efficiency_levels, cmap='jet', extend='max')
cl2 = ax2.contour(N_grid, T_grid, efficiency_map, levels=efficiency_levels, colors='black', linewidths=0.5)
ax2.clabel(cl2, fmt='%1.2f', fontsize=9)
fig2.colorbar(cp2, ax=ax2, label='Efficiency')
ax2.set_xlabel('Speed [rpm]', fontsize=12)
ax2.set_ylabel('Torque [Nm]', fontsize=12)
ax2.set_ylim(0, 150)
plt.tight_layout()
plt.savefig('spm_efficiency_map.png')
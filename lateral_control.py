"""
@file: lateral_control.py
@description:
该文件实现了基于模型预测控制(MPC)的车辆横向控制器，用于精确的路径跟踪。
该控制器根据当前车辆状态和参考轨迹，通过在线优化计算最佳转向角指令。

主要函数:
1. MPCController.__init__(): 初始化MPC控制器，设置参数和优化器
2. MPCController._build_state_space_model(): 根据当前纵向速度构建线性化的状态空间模型
3. MPCController._setup_optimizer(): 使用CasADi构建优化问题及约束
4. MPCController.solve(): 主接口函数，求解MPC优化问题并返回最优转向角

核心原理:
- 使用"自行车模型"作为车辆动力学模型
- 状态向量: [横向误差, 航向误差, 横向速度, 横摆角速度]
- 控制输入: 前轮转向角
- 优化目标: 最小化路径跟踪误差、控制输入及其变化率
- 采用滚动时域优化策略，每次只执行优化序列的第一步
"""

import numpy as np
import casadi as ca
from config import *

class MPCController:
    """
    模型预测控制器（MPC），用于车辆的横向轨迹跟踪。
    实现了基于线性时变(LTV)状态空间模型的模型预测控制算法。
    """
    def __init__(self):
        """
        初始化MPC控制器。
        设置预测和控制时域、权重矩阵、车辆参数，并初始化优化器。
        """
        # 加载参数
        self.N = MPC_HORIZON
        self.N_c = MPC_CONTROL_HORIZON
        self.Q = np.diag(MPC_Q)  # 状态误差权重矩阵 [y, psi]
        self.R = np.diag(MPC_R)  # 控制输入权重矩阵 [delta]
        self.Rd = np.diag(MPC_RD) # 控制输入变化率权重矩阵 [delta_dot]

        # 从config加载车辆参数
        self.lf = VEHICLE_LF     # 前轴到质心的距离
        self.lr = VEHICLE_LR     # 后轴到质心的距离
        self.Iz = VEHICLE_IZ     # 车辆绕z轴的转动惯量
        self.m = VEHICLE_MASS    # 车辆质量
        max_lateral_force = self.m * GRAVITATIONAL_ACCEL * 0.9
        alpha_slip_limit = np.deg2rad(10.0)
                
                # 这就是物理引擎实际使用的【车轴】刚度
        self.actual_axle_stiffness = max_lateral_force / alpha_slip_limit # (约 75900)

                # 我们假设前后轴刚度相同
        self.Cf_axle = self.actual_axle_stiffness
        self.Cr_axle = self.actual_axle_stiffness

        # 初始化优化器
        self._setup_optimizer()
        
        # 用于存储上一个控制指令，以计算变化率
        self.last_control = 0.0

    def _build_state_space_model(self, vx):
        """
        根据当前纵向速度vx，构建线性化的状态空间矩阵A, B。
        
        参数:
            vx (float): 车辆当前纵向速度
            
        返回:
            Ad (numpy.ndarray): 离散化后的系统矩阵A
            Bd (numpy.ndarray): 离散化后的输入矩阵B
            
        说明:
            状态向量: x = [y_e, psi_e, vy, psi_dot]^T 
            (横向误差, 航向误差, 横向速度, 横摆角速度)
            控制向量: u = [delta]^T (前轮转向角)
        """
        # 避免除以零
        if abs(vx) < 0.1:
            vx = 0.1
        
        A = np.zeros((4, 4))
        A[0, 1] = vx # y_e_dot = vy*cos(psi_e) + vx*sin(psi_e) ~= vy + vx*psi_e
        A[0, 2] = 1  # y_e_dot 实际上是 vy 在世界坐标系y方向的分量 + vx 在世界坐标系y方向分量
        A[1, 3] = 1  # psi_e_dot = psi_dot - psi_desired_dot, 假设期望路径的横摆角速度为0
        
        A[2, 2] = -(self.Cf_axle + self.Cr_axle) / (self.m * vx)
        A[2, 3] = (self.Cr_axle * self.lr - self.Cf_axle * self.lf) / (self.m * vx) - vx
        A[3, 2] = (self.Cr_axle * self.lr - self.Cf_axle * self.lf) / (self.Iz * vx)
        A[3, 3] = -(self.Cf_axle * self.lf**2 + self.Cr_axle * self.lr**2) / (self.Iz * vx)

        B = np.zeros((4, 1))
        B[2, 0] = self.Cf_axle / self.m
        B[3, 0] = self.Cf_axle * self.lf / self.Iz
        
        # 离散化: Ad = I + A*dt, Bd = B*dt (欧拉前向法)
        Ad = np.eye(4) + A * SIMULATION_DT
        Bd = B * SIMULATION_DT
        
        return Ad, Bd

    def _setup_optimizer(self):
        """
        使用CasADi构建二次规划(QP)优化问题。
        
        定义优化变量、代价函数和约束条件，配置求解器参数。
        这是MPC控制器的核心组件，负责构建整个优化问题框架。
        """
        # 定义符号变量
        self.opti = ca.Opti()
        
        # 优化变量：状态序列和控制序列
        self.X = self.opti.variable(4, self.N + 1) # 状态 [y_e, psi_e, vy, psi_dot]
        self.U = self.opti.variable(1, self.N_c)   # 控制 [delta]
        
        # 参数：初始状态、参考轨迹、上一次控制
        self.x0 = self.opti.parameter(4, 1)
        self.X_ref = self.opti.parameter(2, self.N + 1) # 参考 [y_ref, psi_ref]
        self.u_last = self.opti.parameter(1, 1)
        
        # 参数：时变的A, B矩阵序列
        self.A_series = self.opti.parameter(4, 4 * self.N)
        self.B_series = self.opti.parameter(4, 1 * self.N)

        # 定义代价函数
        cost = 0
        for k in range(self.N):
            # 跟踪误差代价 (只关心y_e和psi_e)
            state_error = self.X[:2, k] - self.X_ref[:, k]
            cost += ca.mtimes([state_error.T, self.Q, state_error])
            
            # 控制量和变化率代价，仅在控制时域内
            if k < self.N_c:
                cost += ca.mtimes([self.U[:, k].T, self.R, self.U[:, k]])
                if k == 0:
                    control_diff = self.U[:, k] - self.u_last
                else:
                    control_diff = self.U[:, k] - self.U[:, k-1]
                cost += ca.mtimes([control_diff.T, self.Rd, control_diff])

        # 最终状态的代价
        final_state_error = self.X[:2, self.N] - self.X_ref[:, self.N]
        cost += ca.mtimes([final_state_error.T, self.Q, final_state_error])

        self.opti.minimize(cost)

        # 定义动力学约束
        for k in range(self.N):
            A_k = self.A_series[:, k*4:(k+1)*4]
            B_k = self.B_series[:, k*1:(k+1)*1]
            if k < self.N_c:
                # 在控制时域内，使用优化变量 U
                self.opti.subject_to(self.X[:, k+1] == A_k @ self.X[:, k] + B_k @ self.U[:, k])
            else:
                # 在控制时域外，保持最后一个控制输入
                self.opti.subject_to(self.X[:, k+1] == A_k @ self.X[:, k] + B_k @ self.U[:, self.N_c-1])

        # 控制输入约束
        self.opti.subject_to(self.opti.bounded(-MAX_STEERING_ANGLE, self.U, MAX_STEERING_ANGLE))
        self.opti.subject_to(self.X[:, 0] == self.x0)

        # 设置求解器
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        self.opti.solver('ipopt', opts)

    def solve(self, vehicle_state, reference_path, velocity_profile=None):
        """
        求解MPC优化问题，获取最优控制输入。
        
        参数:
            vehicle_state (dict): 车辆当前状态，包含位置、姿态和速度信息
            reference_path (list): 参考路径点列表
            velocity_profile (list, optional): 规划的速度曲线，用于构建时变模型
            
        返回:
            float: 计算得到的最优前轮转向角
            
        说明:
            这是MPC控制器的主接口函数，每个控制周期被调用一次。
            函数计算当前车辆与参考路径的误差，构建时变模型，
            然后求解优化问题获得最优控制输入。
        """
        # --- 1. 计算误差和准备参考序列 ---
        current_pos = np.array([vehicle_state['x'], vehicle_state['y']])
        path_points = np.array([p[:2] for p in reference_path])
        distances = np.linalg.norm(path_points - current_pos, axis=1)
        closest_idx = np.argmin(distances)
        ref_point = reference_path[closest_idx]
        dx = vehicle_state['x'] - ref_point[0]
        dy = vehicle_state['y'] - ref_point[1]
        ref_psi = ref_point[2]
        y_e = -dx * np.sin(ref_psi) + dy * np.cos(ref_psi)
        psi_e = (vehicle_state['psi'] - ref_psi + np.pi) % (2 * np.pi) - np.pi
        x0_val = np.array([y_e, psi_e, vehicle_state['vy'], vehicle_state['psi_dot']])
        xref_val = np.zeros((2, self.N + 1))

        # --- 2. 构建时变的A, B矩阵序列 ---
        A_list, B_list = [], []
        
        # 优先使用规划好的速度曲线
        if velocity_profile and len(velocity_profile) > 0:
            for i in range(self.N):
                # 如果曲线不够长，则使用最后一个速度值
                vx = velocity_profile[i] if i < len(velocity_profile) else velocity_profile[-1]
                Ad, Bd = self._build_state_space_model(vx)
                A_list.append(Ad)
                B_list.append(Bd)
        else:
            # 如果没有提供速度曲线，则回退到使用当前速度
            vx_current = vehicle_state['vx']
            for _ in range(self.N):
                Ad, Bd = self._build_state_space_model(vx_current)
                A_list.append(Ad)
                B_list.append(Bd)
        
        A_series_val = np.hstack(A_list)
        B_series_val = np.hstack(B_list)

        # --- 3. 设置参数并求解 ---
        self.opti.set_value(self.x0, x0_val)
        self.opti.set_value(self.X_ref, xref_val)
        self.opti.set_value(self.u_last, self.last_control)
        self.opti.set_value(self.A_series, A_series_val)
        self.opti.set_value(self.B_series, B_series_val)

        try:
            sol = self.opti.solve()
            u_opt = sol.value(self.U)
            optimal_control = u_opt[0, 0] if u_opt.ndim == 2 else u_opt[0]
            self.last_control = optimal_control
            return optimal_control
        except Exception as e:
            # print(f"MPC solver failed: {e}")
            return self.last_control
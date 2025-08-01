# longitudinal_planner.py (最终版 - "减速让行"逻辑)

import numpy as np
from config import *

# GippsModel 类保持不变，此处省略以保持简洁
class GippsModel:
    def __init__(self):
        self.a = GIPPS_A
        self.b = GIPPS_B
        self.V_desired = GIPPS_V_DESIRED
        self.s0 = GIPPS_S0
        self.tau = GIPPS_TAU
        self.b_hat = GIPPS_B_HAT

    def calculate_target_speed(self, vehicle, lead_vehicle):
        ego_state = vehicle.get_state()
        lead_state = lead_vehicle.get_state()
        v_ego = ego_state['vx']
        v_accel = v_ego + 2.5 * self.a * self.tau * (1 - v_ego / self.V_desired) * np.sqrt(0.025 + v_ego / self.V_desired)
        ego_long_pos = vehicle.get_current_longitudinal_pos()
        lead_long_pos = lead_vehicle.get_current_longitudinal_pos()
        gap = lead_long_pos - ego_long_pos - lead_vehicle.length
        if gap < 0.1: gap = 0.1
        v_lead = lead_state['vx']
        sqrt_term = self.b**2 * self.tau**2 - self.b * (2 * gap - v_ego * self.tau - v_lead**2 / self.b_hat)
        if sqrt_term < 0:
            v_brake = float('inf')
        else:
            v_brake = -self.b * self.tau + np.sqrt(sqrt_term)
        return min(v_accel, v_brake, self.V_desired)

class LongitudinalPlanner:
    def __init__(self, vehicle_params, mpc_params):
        self.a_max = vehicle_params.get('a_max', GIPPS_A)
        self.b_max = abs(vehicle_params.get('b_max', GIPPS_B))
        self.v_desired = vehicle_params.get('v_desired', GIPPS_V_DESIRED)
        self.dt = SIMULATION_DT
        self.N = mpc_params.get('N', MPC_HORIZON) 
        self.gipps_model = GippsModel()


    def _generate_profile_to_target(self, current_vx, target_vx, N):
        """生成从当前速度平滑过渡到目标速度的曲线"""
        profile = []
        v = current_vx
        for _ in range(N):
            if v < target_vx:
                v = min(target_vx, v + self.a_max * self.dt)
            else:
                v = max(target_vx, v - self.b_max * self.dt)
            profile.append(v)
        return profile

    def _generate_yielding_profile(self, current_vx):
        """
        生成一个“减速-恢复”的完整让行曲线。
        """
        profile = []
        v = current_vx
        
        # 阶段一：以最大减速度减速3秒
        decel_steps = int(2.0 / self.dt)
        for _ in range(decel_steps):
            v = max(0, v - self.b_max * self.dt) # 速度不低于0
            profile.append(v)
            
        # 阶段二：以最大加速度恢复到期望速度
        # 我们生成一个足够长的恢复曲线，以确保能达到期望速度
        recovery_steps = int(10.0 / self.dt) # 假设10秒内足以恢复
        for _ in range(recovery_steps):
            if v >= self.v_desired:
                # 如果已达到或超过期望速度，则保持期望速度
                v = self.v_desired
            else:
                v = min(self.v_desired, v + self.a_max * self.dt)
            profile.append(v)
            
        return profile

    def _check_intersection_yielding(self, vehicle, all_vehicles):
        """检查是否需要为交叉口让行"""
        for v in all_vehicles:
            if v.vehicle_id == vehicle.vehicle_id or v.has_passed_intersection:
                continue
            if vehicle._does_path_conflict(v) and not vehicle.has_priority_over(v):
                return True
        return False

    # ------------------ 公开接口 (Public API) ------------------

    def generate_initial_profile(self, vehicle):
        """为新生成的车辆提供初始化接口"""
        return self._generate_profile_to_target(
            current_vx=vehicle.state['vx'], 
            target_vx=self.v_desired,
            N=vehicle.profile_buffer_size
        )

    def determine_action(self, vehicle, all_vehicles):
        """
        核心决策函数：分析情况并返回一个包含“指令”的字典。
        """
        # --- 检查交叉口让行事件 ---
        # 只有当车辆尚未承诺让行时，才进行检查，避免重复触发
        if not vehicle.has_passed_intersection and not vehicle.is_yielding:
            entry_index = vehicle.road.get_conflict_entry_index(vehicle.move_str)
            if entry_index != -1 and vehicle.get_current_longitudinal_pos() < vehicle.path_distances[entry_index]:
                dist_to_entry = vehicle.path_distances[entry_index] - vehicle.get_current_longitudinal_pos()
                if dist_to_entry < vehicle.get_safe_stopping_distance():
                    if self._check_intersection_yielding(vehicle, all_vehicles):
                        # 指令：需要让行
                        yielding_profile = self._generate_yielding_profile(vehicle.state['vx'])
                        return {'action': 'yield', 'profile': yielding_profile}

        # --- 默认：无特殊事件 ---
        return {'action': 'none'}
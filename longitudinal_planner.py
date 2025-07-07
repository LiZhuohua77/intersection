# longitudinal_planner.py

import numpy as np
from config import *

# =============================================================================
#  Gipps模型被封装在此处，作为规划器的内部工具
# =============================================================================
class GippsModel:
    """实现Gipps跟驰模型，用于计算安全的跟车速度。"""
    def __init__(self):
        self.a = GIPPS_A
        self.b = GIPPS_B
        self.V_desired = GIPPS_V_DESIRED
        self.s0 = GIPPS_S0
        self.tau = GIPPS_TAU
        self.b_hat = GIPPS_B_HAT

    def calculate_target_speed(self, vehicle, lead_vehicle):
        """计算跟驰目标速度。"""
        ego_state = vehicle.get_state()
        lead_state = lead_vehicle.get_state()
        
        v_ego = ego_state['vx']
        v_accel = v_ego + 2.5 * self.a * self.tau * (1 - v_ego / self.V_desired) * np.sqrt(0.025 + v_ego / self.V_desired)

        # 此处需要计算两车沿路径的纵向距离，而不是简单的坐标差
        ego_long_pos = vehicle.get_current_longitudinal_pos()
        lead_long_pos = lead_vehicle.get_current_longitudinal_pos()
        
        # gap是车头到车尾的距离
        gap = lead_long_pos - ego_long_pos - lead_vehicle.length
        if gap < 0: gap = 0.01

        v_lead = lead_state['vx']
        sqrt_term = self.b**2 * self.tau**2 - self.b * (2 * gap - v_ego * self.tau - v_lead**2 / self.b_hat)
        
        if sqrt_term < 0:
            v_brake = float('inf')
        else:
            v_brake = -self.b * self.tau + np.sqrt(sqrt_term)
        
        return min(v_accel, v_brake, self.V_desired)

# =============================================================================
#  主规划器类
# =============================================================================
class LongitudinalPlanner:
    """
    纵向速度规划器。
    负责根据交通规则、前车状态和交叉口冲突，生成未来N步的速度曲线。
    """
    def __init__(self, vehicle_params, mpc_params):
        self.a_max = vehicle_params.get('a_max', GIPPS_A)
        self.b_max = abs(vehicle_params.get('b_max', GIPPS_B))
        self.v_desired = vehicle_params.get('v_desired', GIPPS_V_DESIRED)
        
        self.N = mpc_params.get('N', MPC_HORIZON)
        self.dt = SIMULATION_DT
        
        # 规划器拥有自己的Gipps模型实例
        self.gipps_model = GippsModel()

    def _generate_profile_to_target(self, current_vx, target_vx):
        """生成从当前速度平滑过渡到目标速度的曲线"""
        profile = []
        v = current_vx
        for _ in range(self.N):
            if v < target_vx:
                v = min(target_vx, v + self.a_max * self.dt)
            else:
                v = max(target_vx, v - self.b_max * self.dt)
            profile.append(v)
        return profile

    def _generate_stopping_profile(self, current_vx, distance_to_stop_point):
        """生成平滑减速至0的速度曲线"""
        if distance_to_stop_point <= 1.0:
            return [0.0] * self.N
        required_deceleration = (current_vx**2) / (2 * distance_to_stop_point)
        deceleration = min(required_deceleration, self.b_max)
        profile = []
        v = current_vx
        for _ in range(self.N):
            v = max(0, v - deceleration * self.dt)
            profile.append(v)
        return profile

    def plan(self, vehicle, all_vehicles):
        """
        执行一步规划，返回最优的速度曲线。
        """
        # print(f"\n{'='*80}")
        # print(f"规划器调试 - 车辆ID: {vehicle.vehicle_id} | 移动路径: {vehicle.move_str}")
        # print(f"当前位置: ({vehicle.state['x']:.2f}, {vehicle.state['y']:.2f}) | 当前速度: {vehicle.state['vx']:.2f} m/s")
        # print(f"已通过交叉口: {vehicle.has_passed_intersection} | 纵向位置: {vehicle.get_current_longitudinal_pos():.2f}")
        # print(f"{'='*80}")
        
        # --- 1. 最高优先级：交叉口让行决策 ---
        must_yield = False
        # print("\n1. 交叉口让行决策检查...")
        
        if not vehicle.has_passed_intersection:
            # print("  车辆尚未通过交叉口，进行让行决策...")
            entry_index = vehicle.road.get_conflict_entry_index(vehicle.move_str)
            #print(f"  冲突入口点索引: {entry_index}")
            
            if entry_index != -1:
                current_pos = vehicle.get_current_longitudinal_pos()
                entry_pos = vehicle.path_distances[entry_index]
                # print(f"  当前纵向位置: {current_pos:.2f}, 入口点位置: {entry_pos:.2f}")
                
                if current_pos < entry_pos:
                    # print(f"  车辆尚未到达入口点")
                    dist_to_entry = entry_pos - current_pos
                    safe_stop_dist = vehicle.get_safe_stopping_distance()
                    # print(f"  到入口点距离: {dist_to_entry:.2f}, 安全停车距离: {safe_stop_dist:.2f}")
                    
                    if dist_to_entry < safe_stop_dist:
                        # print("  *** 距离小于安全停车距离，必须立即进行让行决策 ***")
                        must_yield, conflict_resolved = self._check_intersection_yielding(vehicle, all_vehicles)
                        # print(f"  让行决策结果 - 必须让行: {must_yield}, 冲突已解决: {conflict_resolved}")
                    else:
                        # print("  距离大于安全停车距离，尚无需决策")
                        pass
                else:
                    # print("  车辆已经通过入口点")
                    pass
            else:
                # print("  路径不通过冲突区域，无需让行")
                pass
        else:
            # print("  车辆已通过交叉口，跳过让行决策")
            pass
        
        # print(f"\n最终让行决策: {'必须让行' if must_yield else '可以通行'}")
        
        if must_yield:
            # print("\n2. 生成停车减速曲线...")
            entry_index = vehicle.road.get_conflict_entry_index(vehicle.move_str)
            stop_point_dist = vehicle.path_distances[entry_index]
            current_pos = vehicle.get_current_longitudinal_pos()
            distance_to_stop = stop_point_dist - current_pos
            # print(f"  停车点距离: {distance_to_stop:.2f} m")
            
            profile = self._generate_stopping_profile(vehicle.state['vx'], distance_to_stop)
            # print(f"  减速曲线[前10步]: {[f'{v:.2f}' for v in profile[:10]]}")
            return profile

        # --- 2. 次高优先级：同道车辆跟驰 ---
        # print("\n2. 检查前方车辆跟驰...")
        leader = vehicle.find_leader(all_vehicles)
        if leader:
            # print(f"  发现前车 ID: {leader.vehicle_id}, 距离: {leader.get_current_longitudinal_pos() - vehicle.get_current_longitudinal_pos():.2f} m")
            # print(f"  前车速度: {leader.state['vx']:.2f} m/s")
            
            # 使用Gipps模型计算瞬时目标速度
            target_speed_gipps = self.gipps_model.calculate_target_speed(vehicle, leader)
            # print(f"  Gipps模型计算的目标速度: {target_speed_gipps:.2f} m/s")
            
            # 生成一个趋向于该目标速度的平滑速度曲线
            profile = self._generate_profile_to_target(vehicle.state['vx'], target_speed_gipps)
            # print(f"  跟车速度曲线[前10步]: {[f'{v:.2f}' for v in profile[:10]]}")
            return profile

        # --- 3. 最低优先级：自由流加速 ---
        # print("\n3. 无前车约束，生成自由流加速曲线...")
        # print(f"  期望速度: {self.v_desired:.2f} m/s")
        profile = self._generate_profile_to_target(vehicle.state['vx'], self.v_desired)
        # print(f"  自由流速度曲线[前10步]: {[f'{v:.2f}' for v in profile[:10]]}")
        return profile

    def _check_intersection_yielding(self, vehicle, all_vehicles):
        """执行交叉口让行规则检查。"""
        # print(f"\n  交叉口让行规则检查 - 车辆ID: {vehicle.vehicle_id}")
        
        # 找出所有可能冲突的车辆
        conflicting_vehicles = []
        for v in all_vehicles:
            if v.vehicle_id == vehicle.vehicle_id or v.has_passed_intersection:
                continue
            
            does_conflict = vehicle._does_path_conflict(v)
            # print(f"    检查与车辆ID {v.vehicle_id} 路径冲突: {does_conflict}")
            
            if does_conflict:
                conflicting_vehicles.append(v)
                # print(f"    添加冲突车辆 ID: {v.vehicle_id}, 路径: {v.move_str}")
        
        # print(f"    冲突车辆总数: {len(conflicting_vehicles)}")
        
        # 如果没有冲突车辆，无需让行
        if not conflicting_vehicles:
            # print("    无冲突车辆，可以安全通过")
            return False, True
        
        # 检查优先级
        for other_v in conflicting_vehicles:
            has_priority = vehicle.has_priority_over(other_v)
            # print(f"    优先级检查 - 我方车辆({vehicle.vehicle_id})对比车辆({other_v.vehicle_id}): 我方优先={has_priority}")
            
            # 如果本车对任何一辆冲突车没有优先权，就必须让行
            if not has_priority:
                # print(f"    让行决策: 必须让行给车辆ID {other_v.vehicle_id}")
                return True, False
        
        # 对所有冲突车都有优先权
        # print("    让行决策: 对所有冲突车辆都有优先权，可以通行")
        return False, True
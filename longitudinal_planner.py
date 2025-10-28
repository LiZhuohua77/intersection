"""
@file: longitudinal_planner.py
@description:
[重构后]
该文件定义了背景车辆（HV）的纵向（前进速度）规划与决策逻辑。
它不再使用固定的速度曲线，而是基于一个动态的、反应式的驾驶员模型（IDM）来实时计算目标速度。
该规划器为每个HV实例赋予了持续的“驾驶个性”和在交叉口的“隐藏意图”，
从而为RL智能体提供一个更真实、更具挑战性的交互环境。
"""
import random
import numpy as np

# [新] 导入新的配置项和即将创建的模型
from config import IDM_PARAMS, INTERACTION_ZONE_RADIUS
from driver_model import IDM # 假设这个文件和IDM类未来会创建

class LongitudinalPlanner:
    """
    [重构后]
    纵向规划器，负责基于IDM模型进行高层次的速度决策。
    它根据车辆的“个性”和“意图”，动态计算出每一步的目标速度。
    """
    def __init__(self, vehicle, personality: str, intent: str):
        """
        [重构后]
        初始化纵向规划器，并赋予车辆个性和意图。
        
        参数:
            vehicle (Vehicle): 该规划器所属的车辆对象。
            personality (str): 驾驶员个性 ('AGGRESSIVE', 'NORMAL', 'CONSERVATIVE')。
            intent (str): 车辆在交叉口的隐藏意图 ('GO' or 'YIELD')。
        """
        self.vehicle = vehicle
        self.personality = personality
        self.intent = intent

        # 从配置中根据个性加载IDM参数
        if self.personality not in IDM_PARAMS:
            raise ValueError(f"未知的驾驶员个性: {self.personality}")
        self.idm_params = IDM_PARAMS[self.personality].copy()
        self.v0_scaling_factor = 1.0

        # 实例化IDM模型
        # 注意: IDM类需要在新的 driver_models.py 文件中实现
        self.idm_model = IDM(self.idm_params)
        self.intersection_state = 'APPROACHING'
        
        print(f"车辆 {self.vehicle.vehicle_id} 初始化规划器，个性: {self.personality}, 意图: {self.intent}")

    def update_speed_scaling(self, scaling_factor: float):
            """更新期望速度v0的缩放因子，并重新配置IDM模型"""
            self.v0_scaling_factor = np.clip(scaling_factor, 0.1, 1.0) # 限制最小为0.1
            
            # 获取原始的 v0
            original_v0 = IDM_PARAMS[self.personality]['v0']
            
            # 计算缩放后的 v0
            scaled_v0 = original_v0 * self.v0_scaling_factor
            
            # 更新当前使用的 IDM 参数
            self.idm_params['v0'] = scaled_v0
            
            # [关键] 重新配置 IDM 模型以使用新的 v0
            self.idm_model.update_parameters(self.idm_params) 
            # (假设您的 IDM 类有一个 update_parameters 方法，或者重新实例化 self.idm_model = IDM(...) )
            
            # print(f"  -> HV #{self.vehicle.vehicle_id} ({self.personality}) v0 scaled to: {scaled_v0:.2f} (Factor: {self.v0_scaling_factor:.2f})") # 用于调试

    def _get_default_speed(self, all_vehicles, v_ego, reason_prefix=''):
            """
            [辅助函数] 执行默认的跟驰或自由流逻辑。
            """
            lead_vehicle = self.vehicle.find_leader(all_vehicles)
            if lead_vehicle:
                v_lead = lead_vehicle.state['vx']
                gap = (lead_vehicle.get_current_longitudinal_pos() - 
                    self.vehicle.get_current_longitudinal_pos() - 
                    lead_vehicle.length)
                target_speed = self.idm_model.get_target_speed(v_ego, v_lead, gap)
                return {'speed': target_speed, 'reason': f'{reason_prefix}CAR_FOLLOW'}
            else:
                target_speed = self.idm_model.get_target_speed(v_ego, None, None)
                return {'speed': target_speed, 'reason': f'{reason_prefix}FREE_FLOW'}

    def get_target_speed(self, all_vehicles) -> dict:
            """
            [V4 修正版] 移除了锁定的状态机，允许车辆在冲突消失后重新启动。
            决策逻辑在感知区内的每一步都会重新评估。
            """
            v_ego = self.vehicle.state['vx']
            
            # --- 1. 获取交叉口相关信息 ---
            perception_radius = self.vehicle.road.extended_conflict_zone_radius
            trigger_radius = self.vehicle.road.conflict_zone_radius + 15.0
            dist_to_entry = self.vehicle.dist_to_intersection_entry
            
            is_in_perception_zone = dist_to_entry < perception_radius
            is_at_trigger_point = dist_to_entry < trigger_radius
            has_passed = self.vehicle.has_passed_intersection

            # --- 2. 状态机逻辑 ---
            
            # 状态 1: 已经通过交叉口
            if has_passed:
                if self.intersection_state != 'PASSED':
                    print(f"  -> HV #{self.vehicle.vehicle_id} PASSED intersection. Resetting state.")
                    self.intersection_state = 'PASSED'
                # 恢复默认行为
                return self._get_default_speed(all_vehicles, v_ego, 'PASSED_')

            # 状态 2: 在感知区之外
            if not is_in_perception_zone:
                self.intersection_state = 'APPROACHING' # 重置状态
                # 执行默认行为
                return self._get_default_speed(all_vehicles, v_ego, 'OUTSIDE_')

            # 状态 3: 在感知区之内 (尚未通过)
            # 此时，我们不再关心 'APPROACHING', 'DECIDING', 'YIELDING' 等旧状态
            # 而是每一步都重新决策
            
            # 3a. 寻找冲突对象
            conflicting_rl_agent = self._find_conflicting_rl_agent(all_vehicles)
            conflicting_hv = self._find_conflicting_priority_hv(all_vehicles)

            # 3b. 决策
            needs_to_yield = False
            yield_reason = ""

            if conflicting_rl_agent and self.intent == 'YIELD':
                needs_to_yield = True
                yield_reason = 'YIELD_VIRTUAL_WALL (AV)'
            
            if not needs_to_yield and conflicting_hv:
                if not self.vehicle.has_priority_over(conflicting_hv):
                    needs_to_yield = True
                    yield_reason = 'YIELD_HV'

            # 3c. 执行
            # 场景A: 决定让行 并且 已经到达触发点
            if needs_to_yield and is_at_trigger_point:
                if self.intersection_state != 'YIELDING':
                    print(f"  -> HV #{self.vehicle.vehicle_id} starting to YIELD ({yield_reason}).")
                    self.intersection_state = 'YIELDING'
                
                # 执行让行逻辑 (减速至0)
                v_lead = 0.0
                gap = dist_to_entry
                target_speed = self.idm_model.get_target_speed(v_ego, v_lead, gap)
                return {'speed': target_speed, 'reason': 'YIELDING_ACTIVE'}

            # 场景B: 决定通过 (或正在通过), 
            # 或者 需要让行但尚未到达触发点
            else:
                if not needs_to_yield:
                    if self.intersection_state != 'GOING':
                        print(f"  -> HV #{self.vehicle.vehicle_id} DECIDED to GO.")
                        self.intersection_state = 'GOING'
                    reason_prefix = 'GOING_'
                else: # (needs_to_yield but not at trigger point)
                    self.intersection_state = 'DECIDING_TO_YIELD'
                    reason_prefix = 'DECIDING_'

                # 执行默认行为 (跟驰/自由流)
                return self._get_default_speed(all_vehicles, v_ego, reason_prefix)


    def _find_conflicting_rl_agent(self, all_vehicles):
            """
            [V2 - 修正后] 寻找一个路径冲突且尚未通过交叉口的RL智能体。
            
            这是触发HV特殊决策（GO/YIELD）的关键。
            """
            for v in all_vehicles:
                # 一个有效的冲突对象，必须同时满足三个条件：
                # 1. 是RL智能体
                # 2. 与HV的路径有物理交叉
                # 3. 它还没有驶离交叉口
                if getattr(v, 'is_rl_agent', False) and \
                self.vehicle._does_path_conflict(v) and \
                not v.has_passed_intersection:
                    return v  # 找到了需要对其做出反应的冲突对象
                    
            return None # 没有发现任何需要反应的冲突对象
    
    def _find_conflicting_priority_hv(self, all_vehicles):
            """寻找一个路径冲突且比自己有优先权的HV"""
            for v in all_vehicles:
                if not getattr(v, 'is_rl_agent', False) and \
                v.vehicle_id != self.vehicle.vehicle_id and \
                self.vehicle._does_path_conflict(v) and \
                not v.has_passed_intersection and \
                not self.vehicle.has_priority_over(v): # 检查路权
                    return v
            return None
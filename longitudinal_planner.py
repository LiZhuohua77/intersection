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

    def get_target_speed(self, all_vehicles) -> dict:
        """
        [重构版]
        实现基于“虚拟壁障”的让行逻辑。

        决策优先级:
        1.  最高优先级: 处理与RL Agent的交互。如果意图为YIELD，则将交叉口冲突点
            视为一个速度为0的静止障碍物进行跟驰。
        2.  第二优先级: 处理与其他HV的交互。
        3.  默认行为: 标准的车辆跟驰或自由流。
        """
        v_ego = self.vehicle.state['vx']
        
        # --- 1. 最高优先级：与 RL Agent 的交互 ---
        conflicting_rl_agent = self._find_conflicting_rl_agent(all_vehicles)
        
        # 从Road对象读取决策区的半径 (例如 76米)
        decision_radius = self.vehicle.road.extended_conflict_zone_radius
        is_in_decision_zone = self.vehicle.dist_to_intersection_entry < decision_radius

        if conflicting_rl_agent and is_in_decision_zone:
            
            if self.intent == 'GO':
                # 意图是“抢行”，则执行自由流，无视AV
                target_speed = self.idm_model.get_target_speed(v_ego, None, None)
                return {'speed': target_speed, 'reason': 'INTENT_GO'}

            elif self.intent == 'YIELD':
                # [核心修改] “虚拟壁障”逻辑
                # 将交叉口冲突点视为一个速度为0的静止前车
                v_lead = 0.0 
                # 与这个“虚拟前车”的距离就是到交叉口入口的距离
                gap = self.vehicle.dist_to_intersection_entry
                
                target_speed = self.idm_model.get_target_speed(v_ego, v_lead, gap)
                return {'speed': target_speed, 'reason': 'YIELD_VIRTUAL_WALL'}

        # --- 2. 第二优先级：与其他HV的交互 ---
        # (这部分逻辑可以保持不变，或者也简化为虚拟壁障模型)
        # 为保持一致性，我们同样将其简化
        for other_hv in all_vehicles:
            if (not getattr(other_hv, 'is_rl_agent', False) and
                    other_hv.vehicle_id != self.vehicle.vehicle_id and
                    self.vehicle._does_path_conflict(other_hv)):
                
                if not self.vehicle.has_priority_over(other_hv):
                    # 需要让行另一辆HV，同样视为前方有静止障碍物
                    v_lead = 0.0
                    gap = self.vehicle.dist_to_intersection_entry
                    target_speed = self.idm_model.get_target_speed(v_ego, v_lead, gap)
                    return {'speed': target_speed, 'reason': 'YIELD_HV'}

        # --- 3. 默认行为：标准跟驰或自由流 ---
        lead_vehicle = self.vehicle.find_leader(all_vehicles)
        if lead_vehicle:
            v_lead = lead_vehicle.state['vx']
            gap = (lead_vehicle.get_current_longitudinal_pos() - 
                self.vehicle.get_current_longitudinal_pos() - 
                lead_vehicle.length)
            target_speed = self.idm_model.get_target_speed(v_ego, v_lead, gap)
            return {'speed': target_speed, 'reason': 'CAR_FOLLOW'}
        else:
            target_speed = self.idm_model.get_target_speed(v_ego, None, None)
            return {'speed': target_speed, 'reason': 'FREE_FLOW'}


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
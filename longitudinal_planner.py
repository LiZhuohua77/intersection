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
        self.idm_params = IDM_PARAMS[self.personality]

        # 实例化IDM模型
        # 注意: IDM类需要在新的 driver_models.py 文件中实现
        self.idm_model = IDM(self.idm_params)
        
        print(f"车辆 {self.vehicle.vehicle_id} 初始化规划器，个性: {self.personality}, 意图: {self.intent}")

    def get_target_speed(self, all_vehicles) -> float:
        """
        [最终版]
        该方法实现了分层级的、动态的实时速度决策。
        1. 最高优先级：处理与RL Agent的交互，由预设的'intent' ('GO'/'YIELD')主导。
        2. 第二优先级：处理与其他HV的交互，由基于规则的'has_priority_over'主导。
        3. 默认行为：标准跟驰或自由流。
        """
        v_ego = self.vehicle.state['vx']
        
        # 定义一个安全的、用于“滚动让行”的目标速度（单位: m/s）
        YIELD_TARGET_SPEED = 3.0 

        # --- 1. 最高优先级：与RL Agent的交互 ---
        conflicting_rl_agent = self._find_conflicting_rl_agent(all_vehicles)
        is_in_interaction_zone = self.vehicle.dist_to_intersection_entry < INTERACTION_ZONE_RADIUS

        if conflicting_rl_agent and is_in_interaction_zone:
            if self.intent == 'GO':
                # “GO”意图：无视AV，执行自由流。
                # 此时 v_lead 和 gap 都为 None，IDM只计算自由流项。
                return self.idm_model.get_target_speed(v_ego, None, None)

            elif self.intent == 'YIELD':
                # “YIELD”意图：将AV视为虚拟前车，并为其设置速度下限。
                
                # [核心修正] v_lead 是AV的真实速度和YIELD_TARGET_SPEED中的较大值
                v_lead = max(conflicting_rl_agent.state['vx'], YIELD_TARGET_SPEED)
                
                gap = self.vehicle.dist_to_intersection_entry
                return self.idm_model.get_target_speed(v_ego, v_lead, gap)

        # --- 2. 第二优先级：与其他HV的交互 (仅在不与AV交互时触发) ---
        for other_hv in all_vehicles:
            if not getattr(other_hv, 'is_rl_agent', False) and other_hv.vehicle_id != self.vehicle.vehicle_id and self.vehicle._does_path_conflict(other_hv):
                if not self.vehicle.has_priority_over(other_hv):
                    # 如果需要让行另一辆HV，同样采用“滚动让行”策略
                    v_lead = max(other_hv.state['vx'], YIELD_TARGET_SPEED)
                    gap = self.vehicle.dist_to_intersection_entry
                    return self.idm_model.get_target_speed(v_ego, v_lead, gap)

        # --- 3. 默认行为：标准跟驰或自由流 ---
        lead_vehicle = self.vehicle.find_leader(all_vehicles)
        if lead_vehicle:
            v_lead = lead_vehicle.state['vx']
            gap = lead_vehicle.get_current_longitudinal_pos() - self.vehicle.get_current_longitudinal_pos() - lead_vehicle.length
            return self.idm_model.get_target_speed(v_ego, v_lead, gap)
        else:
            return self.idm_model.get_target_speed(v_ego, None, None)

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
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
        lead_vehicle = self.vehicle.find_leader(all_vehicles)
        conflicting_rl_agent = self._find_conflicting_rl_agent(all_vehicles)
        is_in_interaction_zone = self.vehicle.dist_to_intersection_entry < INTERACTION_ZONE_RADIUS
        
        v_ego = self.vehicle.state['vx']
        
        if conflicting_rl_agent and is_in_interaction_zone and self.intent == 'YIELD':
            # --- 情景A: 交叉口让行 ---
            v_lead = conflicting_rl_agent.state['vx']
            gap = self.vehicle.dist_to_intersection_entry
        elif lead_vehicle:
            # --- 情景B: 标准跟驰 ---
            v_lead = lead_vehicle.state['vx']
            gap = lead_vehicle.get_current_longitudinal_pos() - self.vehicle.get_current_longitudinal_pos() - lead_vehicle.length
        else:
            # --- 情景C: 自由流 ---
            v_lead = None
            gap = None
            
        # [修改] 调用新的、统一的IDM接口
        return self.idm_model.get_target_speed(v_ego, v_lead, gap)

    def _find_conflicting_rl_agent(self, all_vehicles):
        """
        [新] 辅助函数，专门用于寻找路径冲突且在交互区内的RL智能体。
        """
        for v in all_vehicles:
            # 检查对方是否是RL智能体，且路径冲突
            if getattr(v, 'is_rl_agent', False) and self.vehicle._does_path_conflict(v):
                # 额外检查对方是否也进入了交互区，避免对远处的车辆做出反应
                if v.dist_to_intersection_entry < INTERACTION_ZONE_RADIUS:
                    return v
        return None
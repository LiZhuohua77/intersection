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
        [V4 - Refined Yield]
        该方法融合了基于规则的路权判断和基于意图的行为调整。
        YIELD行为被修改为“减速观察”而非“完全停止”。
        """
        lead_vehicle = self.vehicle.find_leader(all_vehicles)
        conflicting_rl_agent = self._find_conflicting_rl_agent(all_vehicles)
        is_in_interaction_zone = self.vehicle.dist_to_intersection_entry < INTERACTION_ZONE_RADIUS
        
        v_ego = self.vehicle.state['vx']
        v_lead = None
        gap = None
        
        # 定义一个安全的目标让行速度（单位: m/s）
        YIELD_TARGET_SPEED = 3.0 

        # --- 核心决策逻辑: 判断是否在交叉口决策区内 ---
        if conflicting_rl_agent and is_in_interaction_zone and not self.vehicle.is_in_intersection:
            
            # 1. 规则判断
            rule_based_decision_to_yield = not self.vehicle.has_priority_over(conflicting_rl_agent)

            # 2. 意图调整
            final_decision_to_yield = False
            if self.intent == 'YIELD':
                final_decision_to_yield = True
            elif self.intent == 'GO':
                final_decision_to_yield = False

            # 3. 执行决策
            if final_decision_to_yield:
                # [核心修正]
                # 执行“减速让行”而非“停车让行”。
                # 车辆将目标速度调整为YIELD_TARGET_SPEED，以低速接近交叉口等待时机。
                v_lead = YIELD_TARGET_SPEED
                gap = self.vehicle.dist_to_intersection_entry
            else: # 决定通行
                v_lead = None
                gap = None

        elif lead_vehicle:
            # --- 次要逻辑: 标准跟驰 ---
            v_lead = lead_vehicle.state['vx']
            gap = lead_vehicle.get_current_longitudinal_pos() - self.vehicle.get_current_longitudinal_pos() - lead_vehicle.length
        
        return self.idm_model.get_target_speed(v_ego, v_lead, gap)

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
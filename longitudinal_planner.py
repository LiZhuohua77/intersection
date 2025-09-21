"""
@file: longitudinal_planner.py
@description:
该文件定义了背景车辆（NPC, Non-Player Character）的纵向（前进速度）规划与决策逻辑。
它负责处理两种核心的驾驶行为：常规的车辆跟驰和复杂的交叉口让行决策，并为
车辆生成相应的速度规划曲线(Speed Profile)以供执行。

主要函数:
1. GippsModel.calculate_target_speed(): 计算基于Gipps模型的目标跟车速度
2. LongitudinalPlanner._generate_profile_to_target(): 生成从当前速度平滑过渡到目标速度的曲线
3. LongitudinalPlanner._generate_cruise_profile(): 生成以期望速度巡航的速度曲线
4. LongitudinalPlanner._generate_yielding_profile(): 生成减速-等待-加速的让行速度曲线
5. LongitudinalPlanner.get_potential_speed_profiles(): 为RL Agent提供不同交互策略的速度规划
6. LongitudinalPlanner._check_intersection_yielding(): 检查是否需要在交叉口让行
7. LongitudinalPlanner.generate_initial_profile(): 为新生成的车辆提供初始速度规划
8. LongitudinalPlanner.determine_action(): 核心决策函数，分析情况并返回控制指令

核心组件:

1.  **GippsModel (Gipps跟驰模型):**
    - **目的:** 实现一个经典的、基于规则的微观交通流模型，用于模拟车辆在单一车道上
      跟随前车的行为。
    - **原理:** 该模型通过分别计算"自由加速下的期望速度"和"为避免与前车碰撞所需
      的安全刹车速度"，并取两者中的较小值作为当前时刻的目标速度。这使得车辆能在
      **追求个人期望速度**和**保证行车安全**这两个目标之间做出动态平衡。

2.  **LongitudinalPlanner (纵向规划器):**
    - **目的:** 作为更高层次的决策模块，它整合了底层的跟驰模型，并加入了更复杂的、
      基于路权和冲突的交叉口交互逻辑。
    - **核心决策逻辑:** 通过determine_action方法实现，基于当前场景状态和预定义规则做出行为决策。
    - **交互建模:** 支持保守(cooperative)和激进(aggressive)两种不同的交互行为模式。
"""
import random
import numpy as np
from config import *

class GippsModel:
    """
    Gipps跟驰模型，用于模拟车辆跟随前车的行为。
    该模型平衡了追求个人期望速度和保证行车安全这两个目标。
    """
    def __init__(self):
        """
        初始化Gipps跟驰模型参数。
        
        参数说明:
        - a: 最大加速度(m/s^2)
        - b: 舒适减速度(m/s^2)
        - V_desired: 期望速度(m/s)
        - s0: 最小安全车头间距(m)
        - tau: 反应时间(s)
        - b_hat: 估计的前车最大减速度(m/s^2)
        """
        self.a = GIPPS_A             # 最大加速度
        self.b = GIPPS_B             # 舒适减速度
        self.V_desired = GIPPS_V_DESIRED  # 期望速度
        self.s0 = GIPPS_S0           # 最小安全车头间距
        self.tau = GIPPS_TAU         # 反应时间
        self.b_hat = GIPPS_B_HAT     # 估计的前车最大减速度

    def calculate_target_speed(self, vehicle, lead_vehicle):
        """
        计算基于Gipps模型的目标跟车速度。
        
        该函数分别计算自由加速和安全制动下的速度，取两者最小值。
        
        参数:
            vehicle (Vehicle): 当前车辆对象
            lead_vehicle (Vehicle): 前方车辆对象
            
        返回:
            float: 计算得到的目标速度(m/s)
        """
        ego_state = vehicle.get_state()
        lead_state = lead_vehicle.get_state()
        v_ego = ego_state['vx']
        
        # 计算自由加速情况下的速度
        v_accel = v_ego + 2.5 * self.a * self.tau * (1 - v_ego / self.V_desired) * np.sqrt(0.025 + v_ego / self.V_desIred)
        
        # 计算当前车头距离
        ego_long_pos = vehicle.get_current_longitudinal_pos()
        lead_long_pos = lead_vehicle.get_current_longitudinal_pos()
        gap = lead_long_pos - ego_long_pos - lead_vehicle.length
        if gap < 0.1: gap = 0.1
        
        # 计算安全制动情况下的速度
        v_lead = lead_state['vx']
        sqrt_term = self.b**2 * self.tau**2 - self.b * (2 * gap - v_ego * self.tau - v_lead**2 / self.b_hat)
        if sqrt_term < 0:
            v_brake = float('inf')
        else:
            v_brake = -self.b * self.tau + np.sqrt(sqrt_term)
        
        # 取最小值作为最终目标速度
        return min(v_accel, v_brake, self.V_desired)

class LongitudinalPlanner:
    """
    纵向规划器，负责高层次的速度决策和规划。
    整合了跟驰模型并加入了交叉口交互逻辑。
    """
    def __init__(self, vehicle_params, mpc_params):
        """
        初始化纵向规划器。
        
        参数:
            vehicle_params (dict): 车辆参数字典
            mpc_params (dict): MPC控制器参数字典
        """
        self.a_max = vehicle_params.get('a_max', GIPPS_A)          # 最大加速度
        self.b_max = abs(vehicle_params.get('b_max', GIPPS_B))     # 最大减速度
        self.v_desired = vehicle_params.get('v_desired', GIPPS_V_DESIRED)  # 期望速度
        self.dt = SIMULATION_DT                                    # 仿真时间步长
        self.N = mpc_params.get('N', MPC_HORIZON)                  # 预测时域长度
        self.gipps_model = GippsModel()                            # 内置的Gipps跟驰模型
        self.aggressive_prob = AGGRESSIVE_PROB                     # 选择激进行为的概率


    def _generate_profile_to_target(self, current_vx, target_vx, N):
        """
        生成从当前速度平滑过渡到目标速度的曲线。
        
        参数:
            current_vx (float): 当前纵向速度(m/s)
            target_vx (float): 目标纵向速度(m/s)
            N (int): 速度曲线的长度(步数)
            
        返回:
            list: 长度为N的速度曲线列表
        """
        profile = []
        v = current_vx
        for _ in range(N):
            if v < target_vx:
                v = min(target_vx, v + self.a_max * self.dt)
            else:
                v = max(target_vx, v - self.b_max * self.dt)
            profile.append(v)
        return profile

    def _generate_cruise_profile(self, current_vx):
        """
        生成一个以期望速度巡航的曲线。
        
        参数:
            current_vx (float): 当前纵向速度(m/s)
            
        返回:
            list: 速度曲线列表，长度为预测时域的两倍
        """
        # 生成一个足够长的巡航曲线，以便后续截取
        return self._generate_profile_to_target(current_vx, self.v_desired, self.N * 2)

    def _generate_yielding_profile(self, current_vx):
        """
        生成一个"减速-恢复"的完整让行曲线。
        
        包含三个阶段：减速、等待、加速恢复。
        
        参数:
            current_vx (float): 当前纵向速度(m/s)
            
        返回:
            list: 完整的让行速度曲线列表
        """
        profile = []
        v = current_vx

        # 阶段一：以最大减速度减速2.5秒
        decel_steps = int(SCENARIO_TREE_LENGTH / self.dt)
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

    def get_potential_speed_profiles(self, vehicle, all_vehicles):
        """
        为RL Agent提供场景树：预测并返回两种可能的未来速度规划。
        
        提供保守(让行)和激进(不让行/巡航)两种不同的交互策略对应的速度规划。
        
        参数:
            vehicle (Vehicle): 当前车辆对象
            all_vehicles (list): 场景中所有车辆对象的列表
            
        返回:
            dict: 包含'cooperative'和'aggressive'两个键的字典，值为对应的速度曲线列表
        """
        # 默认情况下，两种规划都是巡航
        cooperative_profile = self._generate_cruise_profile(vehicle.state['vx'])
        aggressive_profile = self._generate_cruise_profile(vehicle.state['vx'])

        # 检查是否需要让行（与 determine_action 逻辑类似）
        if not vehicle.has_passed_intersection:
            entry_index = vehicle.road.get_conflict_entry_index(vehicle.move_str)
            if entry_index != -1 and vehicle.get_current_longitudinal_pos() < vehicle.path_distances[entry_index]:
                dist_to_entry = vehicle.path_distances[entry_index] - vehicle.get_current_longitudinal_pos()
                # 只有在决策点内，才考虑生成不同的规划
                if dist_to_entry < vehicle.get_safe_stopping_distance():
                    # 检查是否存在需要让行的、更高优先级的RL Agent
                    # 注意：这里我们只关心与RL Agent的交互
                    for v in all_vehicles:
                        # 使用 getattr 确保安全访问 is_rl_agent 属性
                        if getattr(v, 'is_rl_agent', False) and vehicle._does_path_conflict(v) and not vehicle.has_priority_over(v):
                            # 找到了需要交互的RL Agent，生成两种截然不同的规划
                            cooperative_profile = self._generate_yielding_profile(vehicle.state['vx'])
                            # 激进规划保持巡航不变
                            aggressive_profile = self._generate_cruise_profile(vehicle.state['vx'])
                            break # 只考虑第一个冲突的RL Agent

        return {
            'cooperative': cooperative_profile,
            'aggressive': aggressive_profile
        }

    def _check_intersection_yielding(self, vehicle, all_vehicles):
        """
        检查是否需要为交叉口让行。
        
        参数:
            vehicle (Vehicle): 当前车辆对象
            all_vehicles (list): 场景中所有车辆对象的列表
            
        返回:
            bool: 如果需要让行返回True，否则返回False
        """
        for v in all_vehicles:
            # 忽略自身和已经通过交叉口的车辆
            if v.vehicle_id == vehicle.vehicle_id or v.has_passed_intersection:
                continue
            
            # 检查路径冲突
            if vehicle._does_path_conflict(v):
                # 如果对方车辆已经在让行，则我方可以大胆通过
                if hasattr(v, 'is_yielding') and v.is_yielding:
                    print(f"Vehicle {vehicle.vehicle_id} detects that Vehicle {v.vehicle_id} is already yielding, proceeding without slowing down")
                    continue
                    
                # 否则，按照原有逻辑判断优先权
                if not vehicle.has_priority_over(v):
                    return True
        return False

    # ------------------ 公开接口 (Public API) ------------------

    def generate_initial_profile(self, vehicle):
        """
        为新生成的车辆提供初始化速度规划。
        
        参数:
            vehicle (Vehicle): 需要初始化的车辆对象
            
        返回:
            list: 初始速度规划列表
        """
        return self._generate_profile_to_target(
            current_vx=vehicle.state['vx'], 
            target_vx=self.v_desired,
            N=vehicle.profile_buffer_size
        )

    def determine_action(self, vehicle, all_vehicles):
        """
        核心决策函数：分析当前情况并返回一个包含"指令"的字典。
        
        这是规划器的主要接口，根据当前交通状况做出行为决策，
        包括是否需要在交叉口让行以及相应的速度规划。
        
        参数:
            vehicle (Vehicle): 当前车辆对象
            all_vehicles (list): 场景中所有车辆对象的列表
            
        返回:
            dict: 包含'action'键的字典，可能的值有：
                - {'action': 'yield', 'profile': [...]} 表示需要让行并附带速度规划
                - {'action': 'none'} 表示无特殊行为，继续跟随前车或巡航
        """
        # --- 检查交叉口让行事件 ---
        if not vehicle.has_passed_intersection and not vehicle.is_yielding:
            entry_index = vehicle.road.get_conflict_entry_index(vehicle.move_str)
            if entry_index != -1 and vehicle.get_current_longitudinal_pos() < vehicle.path_distances[entry_index]:
                dist_to_entry = vehicle.path_distances[entry_index] - vehicle.get_current_longitudinal_pos()
                if dist_to_entry < vehicle.get_safe_stopping_distance():
                    
                    # --- 步骤1: 优先处理与RL Agent的交互 ---
                    conflicting_rl_agent = None
                    for v in all_vehicles:
                        if getattr(v, 'is_rl_agent', False) and vehicle._does_path_conflict(v):
                            conflicting_rl_agent = v
                            break
                    
                    if conflicting_rl_agent:
                        # --- 检查是否已经做过决策 ---
                        if vehicle.interaction_decision is None:
                            # 尚未决策，现在进行一次性"掷骰子"
                            a = random.random()
                            print(f"Vehicle {vehicle.vehicle_id} ROLLS a {a:.2f}")
                            if a > self.aggressive_prob:
                                vehicle.interaction_decision = 'cooperative'
                                print(f"Vehicle {vehicle.vehicle_id} DECIDES to be COOPERATIVE.")
                            else:
                                vehicle.interaction_decision = 'aggressive'
                                print(f"Vehicle {vehicle.vehicle_id} DECIDES to be AGGRESSIVE.")

                        # --- 根据已经做出的、被记住的决策来行动 ---
                        if vehicle.interaction_decision == 'cooperative':
                            yielding_profile = self._generate_yielding_profile(vehicle.state['vx'])
                            return {'action': 'yield', 'profile': yielding_profile}
                        elif vehicle.interaction_decision == 'aggressive':
                            return {'action': 'none'}

                    # --- 步骤2: 如果不与RL Agent交互，则按常规规则处理 ---
                    if self._check_intersection_yielding(vehicle, all_vehicles):
                        yielding_profile = self._generate_yielding_profile(vehicle.state['vx'])
                        return {'action': 'yield', 'profile': yielding_profile}

        # --- 默认：无特殊事件 ---
        return {'action': 'none'}
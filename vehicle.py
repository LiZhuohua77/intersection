"""
@file: vehicle.py
@description:
该文件是整个仿真项目中最为核心的动态实体定义文件。它包含了两个关键的类：
- `Vehicle`: 代表由预设规则驱动的背景交通车辆 (NPC)。
- `RLVehicle`: 继承自`Vehicle`，代表由强化学习算法驱动的、需要学习决策的智能体。
这两个类封装了车辆的所有属性和行为，包括物理模型、感知、决策、控制和可视化。

---
### `Vehicle` 类 (背景车辆 / NPC)
---
这是一个高度集成的类，模拟了一辆具有自主决策和控制能力的常规车辆。

**核心架构: “大脑”与“身体”的分离**

1.  **物理实体 (“身体”):**
    - 包含一个 `state` 字典，用于存储车辆的完整动力学状态（位置x, y, 航向角psi,
      纵向/横向速度vx/vy, 横摆角速度psi_dot）。
    - 包含 `_update_physics` 方法，该方法实现了基于“自行车模型”的车辆动力学方程，
      负责根据控制指令来更新车辆状态。

2.  **决策与控制系统 (“大脑”):**
    - `Vehicle` 类通过组合的方式，集成并协调了多个独立的规划与控制模块，形成一个
      完整的分层式“决策-控制”栈：
        - `LongitudinalPlanner`: 作为**高级纵向决策模块**，负责处理复杂的交叉口让行等
          需要长远规划的驾驶场景。
        - `MPCController`: 作为**底层横向控制器**，负责根据参考路径精确计算转向角，
          实现高精度的路径跟踪。
        - `PIDController`: 作为**底层纵向控制器**，负责根据目标速度精确计算油门/刹车力，
          实现高精度的速度跟踪。
        - `GippsModel`: 作为默认的**跟驰模型**，处理常规的跟车行驶行为。

**核心工作流程 (`update` 方法):**
在每个仿真步，车辆都遵循一个清晰的“规划-控制-执行”循环：
1.  **规划 (Plan):** 调用 `LongitudinalPlanner` 判断当前是否需要执行特殊机动（如让行），
    并更新其内部的**速度规划缓冲区 (Speed Profile Buffer)**。
2.  **控制 (Control):** 从速度规划缓冲区中提取当前的目标速度，并分别调用 `PIDController`
    和 `MPCController` 计算出所需的纵向力和转向角。
3.  **执行 (Act):** 将计算出的控制指令输入到 `_update_physics` 物理模型中，更新车辆
    的下一时刻状态。

**交通规则 (`has_priority_over` 方法):**
`Vehicle` 类内置了一套明确的、分层级的**路权判断逻辑**，用于在交叉口进行决策。
规则优先级为：1. 时间优先（先到先行） > 2. 行驶意图优先（直行>右转>左转） > 3. 让右原则。
这使得NPC车辆的行为符合常规交通习惯，表现得更加真实和可预测。

---
### `RLVehicle` 类 (强化学习智能体)
---
这是一个为强化学习训练而特化的车辆类。

**继承与差异:**
- `RLVehicle` 继承自 `Vehicle`，因此它自动复用了所有的物理模型、状态表示和可视化代码。
- **核心区别**在于，它在初始化时**禁用了所有基于规则的规划器和控制器** (`planner`,
  `pid_controller`, `mpc_controller` 均设为 `None`)。它的驾驶行为完全由外部的
  强化学习算法（例如DDPG）通过 `step` 方法来驱动。

**Gymnasium 接口实现:**
该类的主要作用是为 `TrafficEnv` 提供与 `Gymnasium` 环境标准兼容的核心交互接口：

- **`get_observation` (状态观测):**
  定义了**“智能体如何感知世界”**。该方法负责收集仿真世界中的原始信息（如自身速度、
  与路径的偏差、周围车辆的相对位置和速度等），将其处理并**归一化**为一个固定长度的
  特征向量，作为RL策略神经网络的输入。

- **`calculate_reward` (奖励函数):**
  定义了**“学习的目标是什么”**。该方法通过一个精心设计的、对多种驾驶目标（如行驶效率、
  安全性、舒适度、路径跟踪精度）进行加权的函数来评估智能体每个动作的好坏。奖励信号是
  引导RL算法朝着期望目标学习的唯一依据。

- **`step` (动作执行与交互):**
  定义了**“智能体如何与世界交互”**。它接收来自RL算法输出的抽象`动作`（例如，归一化
  的油门和转向值），将其转化为物理世界中的力和转角，并调用物理引擎更新自身状态。
  随后，它计算这一步的`奖励`和新的`观测`，并判断回合是否`终止`，最终将这些信息作为
  一个标准元组 `(observation, reward, terminated, truncated, info)` 返回给环境。
"""

import pygame
import numpy as np
import time
import matplotlib.pyplot as plt
from config import *
from longitudinal_control import PIDController
from lateral_control import MPCController
from longitudinal_planner import LongitudinalPlanner
from utils import check_obb_collision
from prediction import TrajectoryPredictor

def calculate_spm_loss_kw(T_inst_nm, n_inst_rpm):
    """
    Calculates the power loss in kW for the SPM motor based on the polynomial model.
    This is a vectorized version for grid calculations.
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

class Vehicle:
    """
    人工车辆代理。
    它拥有自己的物理属性、状态、路径以及一套集成的规划器与控制器。
    """
    def __init__(self, road, start_direction, end_direction, vehicle_id):
        # --- 基本属性初始化 ---
        self.road = road
        self.start_direction = start_direction
        self.end_direction = end_direction
        self.vehicle_id = vehicle_id
        self.move_str = f"{start_direction[0].upper()}_{end_direction[0].upper()}"

        # --- 路径与几何信息 ---
        path_data = self.road.get_route_points(start_direction, end_direction)
        if not path_data or not path_data["smoothed"]:
            raise ValueError(f"无法为车辆 {self.vehicle_id} 生成从 {self.move_str} 的路径")
        

        self.raw_path = path_data["raw"]
        
        self.reference_path = path_data["smoothed"] 
        self.path_points_np = np.array([[p[0], p[1]] for p in self.reference_path])
        self.path_distances = np.insert(np.cumsum(np.sqrt(np.sum(np.diff(self.path_points_np, axis=0)**2, axis=1))), 0, 0)
        self.dist_to_intersection_entry = float('inf')
        # --- 车辆状态 ---
        initial_pos = self.reference_path[0]
        initial_psi = self.reference_path[1][2]
        self.state = {
            'x': initial_pos[0], 'y': initial_pos[1], 'psi': initial_psi,
            'vx': 12.5, 'vy': 0.0, 'psi_dot': 0.0,
        }

        # --- 物理属性 ---
        self.m, self.Iz, self.lf, self.lr, self.Cf, self.Cr = VEHICLE_MASS, VEHICLE_IZ, VEHICLE_LF, VEHICLE_LR, VEHICLE_CF, VEHICLE_CR
        self.width, self.length = VEHICLE_W, VEHICLE_L

        # --- 规划器与控制器 ---
        self.planner = None  # 将在环境重置时由 initialize_planner 动态创建
        self.pid_controller = PIDController()
        self.mpc_controller = MPCController()
        
        # --- 仿真状态标志 ---
        self.completed = False
        self.color = (200, 200, 200)
        self.last_steering_angle = 0.0
        
        # 交叉口相关状态
        self.is_in_intersection = False
        self.has_passed_intersection = False
        self.interaction_decision = None
        self.debug_info = {}
        self.priority_log = {}
        self.decision_lock = {}

        # 可视化开关
        self.show_path_visualization = False
        self.show_debug = True
        self.show_bicycle_model = False

        # 记录速度
        self.speed_history = []

    def initialize_planner(self, personality: str, intent: str):
        """
        [新] 这是由 TrafficEnv 调用的新接口。
        根据分配的个性和意图，创建并初始化车辆的纵向规划器。
        """
        self.planner = LongitudinalPlanner(self, personality, intent)

    def _update_intersection_status(self):
        """根据车辆位置更新交叉口相关的状态标志"""
        if self.has_passed_intersection:
            return

        # 找到当前在路径上的投影点索引
        current_pos = np.array([self.state['x'], self.state['y']])
        distances_to_ego = np.linalg.norm(self.path_points_np - current_pos, axis=1)
        current_path_index = np.argmin(distances_to_ego)

        # --- 修改开始 ---
        # 计算点到冲突区圆心的距离的平方
        cz_center = self.road.conflict_zone['center']
        cz_radius_sq = self.road.conflict_zone['radius'] ** 2
        dist_sq_to_center = (current_pos[0] - cz_center[0])**2 + (current_pos[1] - cz_center[1])**2
        is_currently_in = dist_sq_to_center <= cz_radius_sq
        # --- 修改结束 ---

        # 更新状态机
        if not is_currently_in:
            # 在交叉口外
            if hasattr(self, 'decision_point_index') and current_path_index > self.decision_point_index and self.decision_point_index != -1:
                self.is_approaching_decision_point = True
            if self.is_in_intersection: # 刚开出交叉口
                self.has_passed_intersection = True
                self.is_in_intersection = False
                self.interaction_decision = None
        else:
            # 在交叉口内
            self.is_in_intersection = True
            self.is_approaching_decision_point = False
            self.is_yielding = False # 进入后不再让行，必须快速通过
            self.decision_made = True

    def find_leader(self, all_vehicles, lane_width=4):
        """
        查找本车前方同一路径上的前车
        考虑车辆实际行驶路径，只在相同路径上查找前车
        """
        leader = None
        min_longitudinal_dist = float('inf')

        ego_pos = np.array([self.state['x'], self.state['y']])
        
        # 计算车辆在自身路径上的投影位置
        distances_to_ego = np.linalg.norm(self.path_points_np - ego_pos, axis=1)
        ego_proj_index = np.argmin(distances_to_ego)
        ego_longitudinal_pos = self.path_distances[ego_proj_index]

        for other_vehicle in all_vehicles:
            if other_vehicle.vehicle_id == self.vehicle_id:
                continue
                
            # 只考虑相同路径的车辆作为潜在前车
            # 在交叉口场景，车辆需要有相同的起点和终点才被视为同路径
            if hasattr(other_vehicle, 'move_str') and other_vehicle.move_str != self.move_str:
                continue
                
            other_pos = np.array([other_vehicle.state['x'], other_vehicle.state['y']])
            
            # 将其他车辆投影到本车的路径上
            distances_to_other = np.linalg.norm(self.path_points_np - other_pos, axis=1)
            other_proj_index = np.argmin(distances_to_other)
            
            # 检查横向距离是否在可接受范围内（确认确实在同一车道上）
            lateral_dist = distances_to_other[other_proj_index]
            if lateral_dist > lane_width:
                continue

            # 检查纵向位置是否在前方
            other_longitudinal_pos = self.path_distances[other_proj_index]
            if other_longitudinal_pos <= ego_longitudinal_pos:
                continue
                
            # 更新最近前车
            longitudinal_dist = other_longitudinal_pos - ego_longitudinal_pos
            if longitudinal_dist < min_longitudinal_dist:
                min_longitudinal_dist = longitudinal_dist
                leader = other_vehicle
                
        return leader
    
    def update(self, dt, all_vehicles):
        """
        [重构后] 背景车辆(HV)的主更新循环。
        流程简化为：实时规划 -> 控制 -> 执行。
        """
        if self.completed or not self.planner:
            return

        # 1. 规划 (Planning): 实时计算当前的目标速度
        plan_info = self.planner.get_target_speed(all_vehicles)
        target_speed = plan_info['speed']

        # 2. 控制 (Control): 计算纵向力和转向角
        throttle_brake = self.pid_controller.step(target_speed, self.state['vx'], dt)
        
        # 为了MPC的稳定性，我们需要一个未来速度的估计。
        # 简单起见，我们假设未来将保持当前的目标速度。
        future_speed_profile = [target_speed] * MPC_HORIZON
        steering_angle = self.mpc_controller.solve(self.state, self.reference_path, future_speed_profile)
        self.last_steering_angle = steering_angle
        
        # 3. 执行 (Act): 更新物理状态
        self._update_physics(throttle_brake, steering_angle, dt)

        # 4. 更新辅助状态
        self.speed_history.append(self.get_current_speed())
        self._update_intersection_status()
        self._check_completion()
        self._update_visual_feedback(plan_info)

    def get_local_observation(self) -> np.ndarray:
        """
        [新] 提供一个标准的局部状态向量，供TrafficEnv的_get_observation调用。
        这个方法对 HV 和 AV 都适用。
        """
        # 计算横向和航向误差
        current_pos = np.array([self.state['x'], self.state['y']])
        distances_to_path = np.linalg.norm(self.path_points_np - current_pos, axis=1)
        current_path_index = np.argmin(distances_to_path)
        cross_track_error = distances_to_path[current_path_index]
        path_angle = self.reference_path[current_path_index][2]
        heading_error = (self.state['psi'] - path_angle + np.pi) % (2 * np.pi) - np.pi
        path_completion = self.get_current_longitudinal_pos() / self.path_distances[-1]
        
        # 返回一个归一化的向量，维度需与config.py中定义的AV_OBS_DIM或HV_OBS_DIM匹配
        return np.array([
            self.state['vx'] / 15.0, # 归一化vx
            self.state['vy'] / 5.0,  # 归一化vy
            self.state['psi_dot'] / 2.0, # 归一化psi_dot
            np.tanh(cross_track_error / MAX_RELEVANT_CTE),
            heading_error / np.pi,
            path_completion
        ])

    def check_collision_with(self, other_vehicle) -> bool:
        """[新] 将碰撞检测作为一个独立的、可被外部调用的方法"""
        return check_obb_collision(self, other_vehicle)

    # --------------------------------------------------------------------------
    #  辅助方法 (Helper Methods)
    # --------------------------------------------------------------------------

    def get_speed_history(self):
        """返回车辆从仿真开始到现在的完整速度历史记录。"""
        return self.speed_history


    def get_current_longitudinal_pos(self):
        """获取车辆在自身路径上的纵向投影距离"""
        current_pos = np.array([self.state['x'], self.state['y']])
        distances_to_ego = np.linalg.norm(self.path_points_np - current_pos, axis=1)
        current_path_index = np.argmin(distances_to_ego)
        return self.path_distances[current_path_index]


    def _update_visual_feedback(self, plan_info: dict):
        """
        [最终更新版] 根据规划器返回的详细决策原因来更新车辆颜色。
        """
        if not hasattr(self, 'planner') or self.planner is None:
            self.color = (200, 200, 200)  # 灰色: 无规划器
            return

        reason = plan_info.get('reason', 'UNKNOWN')

        # 规则1: 只要在交叉口物理冲突区内，就优先显示为绿色
        if self.is_in_intersection:
            self.color = (0, 255, 100)  # 亮绿: 正在穿越交叉口
            return

        # 规则2: 根据具体的决策原因设置颜色
        if reason in ["VIRTUAL_FOLLOW", "YIELD_FORCED", "YIELD_HV"]:
            # 所有“让行”相关的决策都显示为橙色
            self.color = (255, 165, 0)  # 橙色: 正在减速让行
        
        elif reason == "INTENT_GO":
            # 坚决执行“抢行”意图
            self.color = (220, 20, 60)   # 深红/品红: 正在意图抢行
            
        elif reason == "GO_OVERRIDE":
            # 本来想让行，但判断安全后决定正常通行
            self.color = (138, 43, 226)  # 紫罗兰色: 确认安全后通行
            
        else: # reason in ["CAR_FOLLOW", "FREE_FLOW", "DEFAULT", "UNKNOWN"]
            # 其他所有默认情况
            self.color = (200, 200, 200)  # 灰色: 正常行驶

    def _check_completion(self):
        """检查是否到达路径终点"""
        dist_to_end = np.linalg.norm([self.state['x'] - self.reference_path[-1][0], self.state['y'] - self.reference_path[-1][1]])
        if dist_to_end < 10.0:
            self.completed = True

    # --------------------------------------------------------------------------
    #  交通规则判断 (Traffic Rule Logic - Called by Planner)
    #  这些方法被规划器调用，以判断路权归属
    # --------------------------------------------------------------------------

    def _get_maneuver_type(self):
        """根据起止方向判断车辆的行驶意图"""
        start, end = self.start_direction, self.end_direction
        if (start, end) in [('north', 'south'), ('south', 'north'), ('east', 'west'), ('west', 'east')]: return 'straight'
        if (start, end) in [('north', 'west'), ('south', 'east'), ('east', 'north'), ('west', 'south')]: return 'right'
        return 'left'

    def _does_path_conflict(self, other_vehicle):
        my_route = self.move_str
        other_route = other_vehicle.move_str

        is_conflict = self.road.do_paths_conflict(my_route, other_route)
        #print(f"DEBUG CONFLICT CHECK: Is there a conflict between {my_route} (me, #{self.vehicle_id}) and {other_route} (#{other_vehicle.vehicle_id})? -> {is_conflict}")

        return is_conflict

    def has_priority_over(self, other_vehicle):
        """
        判断本车(self)是否比另一辆车(other_vehicle)具有更高通行优先级。
        增加决策锁定机制以防止高频振荡。
        """
        # [调试增强] 打印决策锁的状态
        if other_vehicle.vehicle_id in self.decision_lock:
            lock_info = self.decision_lock[other_vehicle.vehicle_id]
            if lock_info['steps_left'] > 0:
                lock_info['steps_left'] -= 1
                # print(f"[DEBUG] Vehicle {self.vehicle_id}: Lock ACTIVE for {other_vehicle.vehicle_id}. Decision: {lock_info['decision']}. Steps left: {lock_info['steps_left']}")
                return lock_info['decision']
            else:
                print(f"[DEBUG] Vehicle {self.vehicle_id}: Lock EXPIRED for {other_vehicle.vehicle_id}.")
                del self.decision_lock[other_vehicle.vehicle_id]

        def log_lock_and_return(decision, rule_id, message):
            # ... (内部逻辑不变) ...
            # [调试增强] 在设置新锁时打印信息
            print(f"[DEBUG] Vehicle {self.vehicle_id}: SETTING NEW LOCK for {other_vehicle.vehicle_id}. Decision: {decision}. Rule: {rule_id}")
            self.decision_lock[other_vehicle.vehicle_id] = {
                'decision': decision,
                'steps_left': 100  # [建议修改] 增加锁定时长
            }
            return decision

        # 规则1: Time-based FCFS（先进入冲突区的先通过）
        my_entry_time = self._estimate_entry_time()
        other_entry_time = other_vehicle._estimate_entry_time()
        time_diff = my_entry_time - other_entry_time

        # [调试增强] 打印决策的关键输入值
        print(f"[DEBUG] V:{self.vehicle_id} vs V:{other_vehicle.vehicle_id} | MyTime: {my_entry_time:.2f}, OtherTime: {other_entry_time:.2f}, Diff: {time_diff:.2f}")

        if time_diff < -0.1: # 我方明显更快
            return log_lock_and_return(True, "R1_Win", f"Priority: Vehicle {self.vehicle_id} has priority over {other_vehicle.vehicle_id} using Rule 1 (Time-based FCFS)")
        if time_diff > 0.1: # 对方明显更快
            return log_lock_and_return(False, "R1_Lose", f"Priority: Vehicle {self.vehicle_id} yields to {other_vehicle.vehicle_id} using Rule 1 (Time-based FCFS)")
        
        # 如果时间差在容差范围内，则进入下一条规则
        
        # 规则2: Direction Priority（直 > 右 > 左）
        priority_map = {'straight': 3, 'right': 2, 'left': 1}
        my_priority = priority_map[self._get_maneuver_type()]
        other_priority = priority_map[other_vehicle._get_maneuver_type()]
        
        if my_priority > other_priority:
            return log_lock_and_return(True, "R2_Win", f"Priority: Vehicle {self.vehicle_id} has priority over {other_vehicle.vehicle_id} using Rule 2 (Direction Priority)")
        if my_priority < other_priority:
            return log_lock_and_return(False, "R2_Lose", f"Priority: Vehicle {self.vehicle_id} yields to {other_vehicle.vehicle_id} using Rule 2 (Direction Priority)")
        
        # 规则3: Right-of-Way（让右）
        right_of_map = {'west': 'south', 'south': 'east', 'east': 'north', 'north': 'west'}
        if right_of_map.get(other_vehicle.start_direction) == self.start_direction:
            return log_lock_and_return(True, "R3_Win", f"Priority: Vehicle {self.vehicle_id} has priority over {other_vehicle.vehicle_id} using Rule 3 (Right-of-Way)")
        if right_of_map.get(self.start_direction) == other_vehicle.start_direction:
            return log_lock_and_return(False, "R3_Lose", f"Priority: Vehicle {self.vehicle_id} yields to {other_vehicle.vehicle_id} using Rule 3 (Right-of-Way)")
        
        # 平级处理，默认不具备优先权
        return log_lock_and_return(False, "Default_Lose", f"Priority: Vehicle {self.vehicle_id} yields to {other_vehicle.vehicle_id} using default rule")

    def _estimate_entry_time(self) -> float:
            """
            [新增] 估算车辆到达交叉口入口点所需的时间（秒）。
            
            这是一个简单的物理估算，用于路权判断。
            
            返回:
                float: 预计到达时间（秒）。如果车辆静止，则返回一个极大值。
            """
            # 获取到交叉口入口的剩余路径距离
            distance = self.dist_to_intersection_entry
            
            # 获取车辆当前的纵向速度
            current_speed = self.state['vx']
            
            # 为避免除以零的错误，如果车辆基本静止，我们认为它需要无穷长时间才能到达
            if current_speed < 0.1:
                return float('inf')
                
            # 核心估算：时间 = 距离 / 速度
            estimated_time = distance / current_speed
            
            return estimated_time

    def _update_physics(self, fx_total, delta, dt):
        """
        混合物理引擎：
        - 高速时使用动力学自行车模型。
        - 低速时使用运动学自行车模型以保证转向的真实性。
        """
        psi, vx, vy, psi_dot = self.state['psi'], self.state['vx'], self.state['vy'], self.state['psi_dot']
        
        # 1. 计算纵向力和纵向加速度 (所有速度下通用)
        F_air_drag = 0.5 * AIR_DENSITY * DRAG_COEFFICIENT * FRONTAL_AREA * vx**2 * np.sign(vx)
        F_rolling_resistance = ROLLING_RESISTANCE_COEFFICIENT * self.m * GRAVITATIONAL_ACCEL * np.sign(vx)
        fx_net = fx_total - F_air_drag - F_rolling_resistance
        ax = fx_net / self.m
        
        # 2. 根据速度选择模型，并计算相应的横向加速度(ay)和航向角加速度(psi_ddot_deriv)
        if abs(vx) < LOW_SPEED_THRESHOLD:
            # --- 低速区: 使用运动学模型 (Kinematic Model) ---
            L = self.lf + self.lr  # 车辆轴距

            if abs(np.cos(delta)) > 1e-6:
                target_psi_dot = (vx * np.tan(delta)) / L
            else:
                target_psi_dot = 0.0

            self.state['vy'] = 0
            
            # 【核心修改】ay 和 psi_ddot_deriv 在此代码块内被定义
            psi_ddot_deriv = (target_psi_dot - psi_dot) / dt
            ay = target_psi_dot * vx

        else:
            # --- 高速区: 使用动力学模型 (Dynamic Model) ---
            max_lateral_force = self.m * GRAVITATIONAL_ACCEL * 0.9
            alpha_slip_limit = np.deg2rad(10.0)
            effective_cornering_stiffness = max_lateral_force / alpha_slip_limit

            alpha_f = np.arctan((vy + self.lf * psi_dot) / vx) - delta
            alpha_r = np.arctan((vy - self.lr * psi_dot) / vx)
            
            if abs(alpha_f) < alpha_slip_limit:
                Fyf = -effective_cornering_stiffness * alpha_f
            else:
                Fyf = -max_lateral_force * np.sign(alpha_f)

            if abs(alpha_r) < alpha_slip_limit:
                Fyr = -effective_cornering_stiffness * alpha_r
            else:
                Fyr = -max_lateral_force * np.sign(alpha_r)
            
            # 【核心修改】将 ay 和 psi_ddot_deriv 的计算移入 else 块内
            original_vx = self.state['vx']
            ay = (Fyf + Fyr) / self.m - original_vx * psi_dot 
            psi_ddot_deriv = (self.lf * Fyf - self.lr * Fyr) / self.Iz
            
        # 3. 统一的状态积分 (此时 ax, ay, psi_ddot_deriv 都有明确定义)
        self.state['vx'] += ax * dt
        self.state['vy'] += ay * dt
        self.state['psi_dot'] += psi_ddot_deriv * dt
        
        # 4. 更新全局位置和姿态
        # 为简单起见，我们使用更新前的值
        original_vx = self.state['vx']
        self.state['x'] += (vx * np.cos(psi) - vy * np.sin(psi)) * dt
        self.state['y'] += (vx * np.sin(psi) + vy * np.cos(psi)) * dt
        self.state['psi'] += psi_dot * dt
        
    def draw(self, surface, transform_func, small_font, scale=1.0):
        """
        以更精细的方式绘制车辆，包括车轮、风挡和调试信息。
        """
        x, y, psi = self.state['x'], self.state['y'], self.state['psi']
        delta = self.last_steering_angle

        # --- 绘制车身 ---
        w, l = self.width, self.length
        body_corners = [(-l/2, -w/2), (l/2, -w/2), (l/2, w/2), (-l/2, w/2)]
        rotated_body = self._rotate_and_transform(body_corners, x, y, psi, transform_func)
        pygame.draw.polygon(surface, self.color, rotated_body)
        pygame.draw.polygon(surface, (0, 0, 0), rotated_body, 1) # 黑色边框

        # --- 绘制风挡 ---
        windshield_corners = [(l/4, -w/2*0.9), (l/2*0.9, -w/2*0.8), (l/2*0.9, w/2*0.8), (l/4, w/2*0.9)]
        rotated_windshield = self._rotate_and_transform(windshield_corners, x, y, psi, transform_func)
        pygame.draw.polygon(surface, (100, 150, 200, 150), rotated_windshield) # 半透明蓝色

        # --- 绘制车轮 ---
        wheel_l, wheel_w = 0.6, 0.3
        wheel_positions = [
            (self.lf, -self.width / 2 * 0.9),  # 前右轮
            (self.lf, self.width / 2 * 0.9),   # 前左轮
            (-self.lr, -self.width / 2 * 0.9), # 后右轮
            (-self.lr, self.width / 2 * 0.9),  # 后左轮
        ]
        
        # 后轮 (无转向)
        rear_right_wheel = self._create_wheel_poly(wheel_positions[2], wheel_l, wheel_w, x, y, psi, 0, transform_func)
        rear_left_wheel = self._create_wheel_poly(wheel_positions[3], wheel_l, wheel_w, x, y, psi, 0, transform_func)
        pygame.draw.polygon(surface, (30, 30, 30), rear_right_wheel)
        pygame.draw.polygon(surface, (30, 30, 30), rear_left_wheel)

        # 前轮 (带转向)
        front_right_wheel = self._create_wheel_poly(wheel_positions[0], wheel_l, wheel_w, x, y, psi, delta, transform_func)
        front_left_wheel = self._create_wheel_poly(wheel_positions[1], wheel_l, wheel_w, x, y, psi, delta, transform_func)
        pygame.draw.polygon(surface, (30, 30, 30), front_right_wheel)
        pygame.draw.polygon(surface, (30, 30, 30), front_left_wheel)

        # 绘制自行车模型可视化
        if self.show_bicycle_model:
            self._draw_bicycle_model_visualization(surface, transform_func, scale)

        if self.show_path_visualization:
            # 绘制原始路径 (红色)
            if len(self.raw_path) > 1:
                transformed_raw_path = [transform_func(p[0], p[1]) for p in self.raw_path]
                pygame.draw.lines(surface, (255, 100, 100, 100), False, transformed_raw_path, 1)

            # 绘制平滑并重采样后的路径 (绿色)
            if len(self.reference_path) > 1:
                transformed_ref_path = [transform_func(p[0], p[1]) for p in self.reference_path]
                pygame.draw.lines(surface, (100, 255, 100), False, transformed_ref_path, 2)

        # --- 绘制调试信息 ---
        if self.show_debug:
            info_text = f"ID:{self.vehicle_id} V:{self.state['vx']:.1f}m/s"
            text_surface = small_font.render(info_text, True, (255, 255, 255))
            # 将文本放在车辆上方
            screen_pos = transform_func(x, y)
            text_rect = text_surface.get_rect(center=(screen_pos[0], screen_pos[1] - 20 * scale))
            surface.blit(text_surface, text_rect)

    def _draw_bicycle_model_visualization(self, surface, transform_func, scale):
        """在车辆上叠加绘制自行车模型的关键元素"""
        x, y, psi = self.state['x'], self.state['y'], self.state['psi']
        vx, vy, psi_dot = self.state['vx'], self.state['vy'], self.state['psi_dot']
        delta = self.last_steering_angle
        
        # 车辆中心点
        center_screen = transform_func(x, y)
        
        # 绘制底盘 (连接前后轴的线)
        rear_axle_world = (x - self.lr * np.cos(psi), y - self.lr * np.sin(psi))
        front_axle_world = (x + self.lf * np.cos(psi), y + self.lf * np.sin(psi))
        rear_axle_screen = transform_func(*rear_axle_world)
        front_axle_screen = transform_func(*front_axle_world)
        pygame.draw.line(surface, (255, 255, 0), rear_axle_screen, front_axle_screen, 2)
        
        # 绘制前后轮方向线
        wheel_len = self.lf * 0.8 # 可视化线长度
        # 后轮
        rear_wheel_end_world = (rear_axle_world[0] + wheel_len * np.cos(psi), rear_axle_world[1] + wheel_len * np.sin(psi))
        rear_wheel_end_screen = transform_func(*rear_wheel_end_world)
        pygame.draw.line(surface, (0, 255, 255), rear_axle_screen, rear_wheel_end_screen, 2)
        # 前轮 (考虑了转向角)
        front_wheel_angle = psi + delta
        front_wheel_end_world = (front_axle_world[0] + wheel_len * np.cos(front_wheel_angle), front_axle_world[1] + wheel_len * np.sin(front_wheel_angle))
        front_wheel_end_screen = transform_func(*front_wheel_end_world)
        pygame.draw.line(surface, (255, 0, 255), front_axle_screen, front_wheel_end_screen, 2)

        # 绘制速度矢量 (在车辆坐标系下)
        speed_angle_body = np.arctan2(vy, vx) # 车辆坐标系下的速度方向
        speed_angle_world = psi + speed_angle_body # 转换到世界坐标系
        speed_magnitude = np.sqrt(vx**2 + vy**2)
        speed_vector_end_world = (x + speed_magnitude * np.cos(speed_angle_world), y + speed_magnitude * np.sin(speed_angle_world))
        speed_vector_end_screen = transform_func(*speed_vector_end_world)
        pygame.draw.line(surface, (0, 255, 0), center_screen, speed_vector_end_screen, 2)
        pygame.draw.circle(surface, (0, 255, 0), speed_vector_end_screen, 4)

    def _rotate_and_transform(self, points, x, y, angle, transform_func):
        """辅助函数，旋转点集并变换到屏幕坐标"""
        rotated_points = []
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        for px, py in points:
            rx = px * cos_a - py * sin_a + x
            ry = px * sin_a + py * cos_a + y
            rotated_points.append(transform_func(rx, ry))
        return rotated_points

    def _create_wheel_poly(self, pos, l, w, x, y, psi, delta, transform_func):
        """辅助函数，创建并变换单个车轮的多边形"""
        wheel_corners = [(-l/2, -w/2), (l/2, -w/2), (l/2, w/2), (-l/2, w/2)]
        
        # 先进行车轮自身的转向
        rotated_wheel = []
        cos_d, sin_d = np.cos(delta), np.sin(delta)
        for wx, wy in wheel_corners:
            rx = wx * cos_d - wy * sin_d + pos[0]
            ry = wx * sin_d + wy * cos_d + pos[1]
            rotated_wheel.append((rx, ry))
            
        # 再随车身一起旋转和平移
        return self._rotate_and_transform(rotated_wheel, x, y, psi, transform_func)
    
    def get_current_speed(self):
        """获取车辆当前的总速度（标量速度，基于 vx 和 vy）"""
        return np.sqrt(self.state['vx']**2 + self.state['vy']**2)

    def toggle_debug_info(self):
        """切换调试信息的显示"""
        self.show_debug = not self.show_debug

    def get_state(self):
        """返回当前车辆状态的字典"""
        return self.state

    def toggle_bicycle_visualization(self):
        """切换自行车模型的可视化显示"""
        self.show_bicycle_model = not self.show_bicycle_model

    def toggle_path_visualization(self):
        self.show_path_visualization = not self.show_path_visualization


class RLVehicle(Vehicle):
    """
    一个专门为强化学习设计的车辆代理。
    """
    def __init__(self, road, start_direction, end_direction, vehicle_id):
        super().__init__(road, start_direction, end_direction, vehicle_id)
        
        self.target_route_str = f"{start_direction}_{end_direction}"

        # 禁用基于规则的控制器
        self.planner = None
        self.pid_controller = None
        self.mpc_controller = None
        self.predictor = TrajectoryPredictor(IDM_PARAMS)
        
        # RL Agent特定属性
        self.is_rl_agent = True
        self.color = (0, 100, 255) # 
        
        # 用于计算奖励和判断超时的状态
        self.steps_since_spawn = 0
        self.max_episode_steps = 1000 
        self.last_longitudinal_pos = self.get_current_longitudinal_pos()
        self.last_action = np.array([0.0, 0.0]) # 上一帧的动作，用于计算平滑度
        self.cross_track_error = 0.0
        self.heading_error = 0.0
        self.debug_log = []

    def plot_reward_history(self, save_path_base=None):
        """
        [修改后] 绘制智能体在单次回合中的奖励历史。
        如果提供了 save_path_base，则将图表保存到文件；否则，直接显示。
        """
        if not self.debug_log:
            print("调试日志为空，无法绘制奖励历史。")
            return

        # --- 1. 从日志中提取数据 ---
        steps = [log['step'] for log in self.debug_log]
        
        # 加权奖励分量
        rewards = {
            'Velocity Tracking': [log.get('reward_velocity_tracking', 0) for log in self.debug_log],
            'Path Following': [log.get('reward_path_following', 0) for log in self.debug_log],
            'Action Smoothness': [log.get('reward_action_smoothness_penalty', 0) for log in self.debug_log],
            'Energy Consumption': [log.get('reward_energy_consumption', 0) for log in self.debug_log],
            'Cost Penalty': [log.get('reward_cost_penalty', 0) for log in self.debug_log],
            'Total Reward': [log.get('total_reward', 0) for log in self.debug_log],
        }

        # 未加权的原始指标
        raw_metrics = {
            'Speed (m/s)': [log.get('raw_speed', 0) for log in self.debug_log],
            'CTE (m)': [log.get('raw_cte', 0) for log in self.debug_log],
            'Heading Error (rad)': [log.get('raw_he', 0) for log in self.debug_log],
            'Action Smoothness Penalty': [log.get('raw_smoothness_penalty', 0) for log in self.debug_log],
            'Power (kW)': [log.get('raw_power', 0) for log in self.debug_log],
            'Cost Potential': [log.get('raw_cost_potential', 0) for log in self.debug_log],
        }

        # --- 2. 绘制加权奖励图 ---
        plt.figure(figsize=(15, 10))
        plt.suptitle(f'Weighted Reward Components for Vehicle {self.vehicle_id}', fontsize=16)
        
        num_rewards = len(rewards)
        for i, (name, data) in enumerate(rewards.items()):
            ax = plt.subplot(num_rewards, 1, i + 1)
            ax.plot(steps, data, label=name)
            ax.set_ylabel(name)
            ax.grid(True)
            if i == num_rewards - 1:
                ax.set_xlabel('Simulation Steps')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if save_path_base:
            save_path = f"{save_path_base}_rewards_weighted.png"
            plt.savefig(save_path)
            plt.close()
            print(f"加权奖励图已保存到: {save_path}")
        else:
            plt.show()

        # --- 3. 绘制未加权指标图 ---
        plt.figure(figsize=(15, 10))
        plt.suptitle(f'Unweighted Performance Metrics for Vehicle {self.vehicle_id}', fontsize=16)

        num_metrics = len(raw_metrics)
        for i, (name, data) in enumerate(raw_metrics.items()):
            ax = plt.subplot(num_metrics, 1, i + 1)
            ax.plot(steps, data, label=name, color='tab:orange')
            ax.set_ylabel(name)
            ax.grid(True)
            if i == num_metrics - 1:
                ax.set_xlabel('Simulation Steps')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if save_path_base:
            save_path = f"{save_path_base}_metrics_unweighted.png"
            plt.savefig(save_path)
            plt.close()
            print(f"原始指标图已保存到: {save_path}")
        else:
            plt.show()

    def get_observation(self, all_vehicles):
        """
        [核心修正] 
        此方法现在是观测的唯一来源，它将拼接所有需要的信息。
        """
        # --- 1. 自身状态 ---
        ego_observation = self._get_ego_observation()
        
        # --- 2. 周围车辆状态 ---
        surrounding_obs = self._get_surrounding_vehicles_observation(all_vehicles)

        # --- 3. 场景树状态 ---
        scenario_tree_obs = self._get_scenario_tree_observation(all_vehicles)

        # --- 4. 组合成最终的扁平化向量 ---
        final_obs = np.array(ego_observation + surrounding_obs + scenario_tree_obs, dtype=np.float32)

        # --- 5. [新增] 在这里进行维度检查 ---
        if final_obs.shape[0] != TOTAL_OBS_DIM:
            print(f"--- 维度不匹配诊断 (来自RLVehicle.get_observation) ---")
            print(f"期望总维度: {TOTAL_OBS_DIM}")
            print(f"实际总维度: {final_obs.shape[0]}")
            print(f"  - 自身状态维度: {len(ego_observation)}")
            print(f"  - 周边车辆维度: {len(surrounding_obs)}")
            print(f"  - 场景树维度: {len(scenario_tree_obs)}")
            raise ValueError("观测维度不匹配，请检查config.py和各观测函数的输出维度。")
            
        return final_obs

    def get_base_observation(self, all_vehicles):
        """
        [重命名与简化]
        原 get_observation -> get_base_observation
        现在只返回基础观测（自身+周围），不再包含场景树。
        """
        ego_observation = self._get_ego_observation()
        surrounding_obs = self._get_surrounding_vehicles_observation(all_vehicles)

        observation = np.array(ego_observation + surrounding_obs, dtype=np.float32)
        return observation
        
    def _get_ego_observation(self):
        """[新] 抽离出自身状态的获取逻辑，使其更清晰。"""
        ego_vx_norm = self.state['vx'] / 15.0 # 使用数值代替GIPPS_V_DESIRED
        ego_vy_norm = self.state['vy'] / 5.0
        ego_psi_dot_norm = self.state['psi_dot'] / 2.0
        
        current_pos = np.array([self.state['x'], self.state['y']])
        distances_to_path = np.linalg.norm(self.path_points_np - current_pos, axis=1)
        current_path_index = np.argmin(distances_to_path)
        self.cross_track_error = distances_to_path[current_path_index]
        path_angle = self.reference_path[current_path_index][2]
        self.heading_error = (self.state['psi'] - path_angle + np.pi) % (2 * np.pi) - np.pi
        path_completion = self.get_current_longitudinal_pos() / self.path_distances[-1]
        
        return [
            ego_vx_norm, ego_vy_norm, ego_psi_dot_norm,
            np.tanh(self.cross_track_error / MAX_RELEVANT_CTE),
            self.heading_error / np.pi,
            path_completion
        ]

    def _get_surrounding_vehicles_observation(self, all_vehicles):
        """
        [恢复后] 获取周围N辆最近的车辆的相对状态，并进行归一化。
        """
        ego_pos = np.array([self.state['x'], self.state['y']])
        ego_vel = np.array([self.state['vx'], self.state['vy']])
        
        neighbors = []
        for v in all_vehicles:
            if v.vehicle_id == self.vehicle_id:
                continue
            
            other_pos = np.array([v.state['x'], v.state['y']])
            dist = np.linalg.norm(ego_pos - other_pos)

            if dist < OBSERVATION_RADIUS:
                other_vel = np.array([v.state['vx'], v.state['vy']])
                relative_pos = other_pos - ego_pos
                relative_vel = other_vel - ego_vel
                neighbors.append((dist, relative_pos[0], relative_pos[1], relative_vel[0], relative_vel[1]))
        
        # 按距离排序，只取最近的N辆车
        neighbors.sort(key=lambda x: x[0])
        neighbors = neighbors[:NUM_OBSERVED_VEHICLES]
        
        # 构建观测向量
        surrounding_obs_flat = []
        for neighbor in neighbors:
            # 归一化: [rel_x, rel_y, rel_vx, rel_vy]
            surrounding_obs_flat.extend([
                neighbor[1] / OBSERVATION_RADIUS,
                neighbor[2] / OBSERVATION_RADIUS,
                neighbor[3] / 15.0, # 使用一个合理的数值（如15m/s）进行归一化
                neighbor[4] / 5.0
            ])
            
        # 如果车辆不足N辆，用0进行填充以保持观测向量长度固定
        padding_len = NUM_OBSERVED_VEHICLES * 4 - len(surrounding_obs_flat)
        if padding_len > 0:
            surrounding_obs_flat.extend([0.0] * padding_len)
            
        return surrounding_obs_flat

    def _get_scenario_tree_observation(self, all_vehicles):
        """
        [修正后] 使用新的TrajectoryPredictor来生成场景树。
        """
        # 默认情况下，两个规划都是空（用0填充）
        flat_yield_traj = np.zeros(PREDICTION_HORIZON * FEATURES_PER_STEP)
        flat_go_traj = np.zeros(PREDICTION_HORIZON * FEATURES_PER_STEP)

        # 寻找一个即将与之发生冲突的、由规则驱动的NPC车辆
        conflicting_npc = None
        for v in all_vehicles:
            if not getattr(v, 'is_rl_agent', False) and self._does_path_conflict(v) and not v.has_passed_intersection:
                conflicting_npc = v
                break
        
        if conflicting_npc:
            # 如果找到了冲突的NPC，调用我们自己的预测器来生成两种可能的未来
            # 注意：self.predictor 是我们在 RLVehicle 的 __init__ 中添加的
            yield_traj = self.predictor.predict(
                current_av_state=self.state, 
                current_hv_state=conflicting_npc.state,
                hv_intent_hypothesis='YIELD',
                hv_vehicle=conflicting_npc
            )
            go_traj = self.predictor.predict(
                current_av_state=self.state, 
                current_hv_state=conflicting_npc.state,
                hv_intent_hypothesis='GO',
                hv_vehicle=conflicting_npc
            )

            # 归一化和展平处理
            flat_yield_traj = self._flatten_and_normalize_trajectory(yield_traj)
            flat_go_traj = self._flatten_and_normalize_trajectory(go_traj)

        # 将两条曲线展平并拼接
        return list(flat_yield_traj) + list(flat_go_traj)

    def _flatten_and_normalize_trajectory(self, trajectory):
        """
        [新] 辅助函数，用于处理预测出的轨迹。
        """
        vec = np.array(trajectory).flatten()
        
        # 填充或截断以保证长度固定
        expected_len = PREDICTION_HORIZON * FEATURES_PER_STEP
        if len(vec) > expected_len:
             vec = vec[:expected_len]
        else:
             vec = np.pad(vec, (0, expected_len - len(vec)), 'constant')

        # 使用 OBSERVATION_RADIUS 进行归一化
        return vec / OBSERVATION_RADIUS


    def _calculate_energy_consumption(self, commanded_accel):
        """
        根据车辆状态和期望加速度，计算最终的电功率消耗 P_elec。
        """
        current_vx = self.state['vx']
        
        # --- 1. 计算车轮处的总牵引力 F_tractive ---
        # F_tractive 是电机需要产生并传递到车轮上的力
        # 它需要克服惯性、空气阻力和滚动阻力
        F_inertial = self.m * commanded_accel
        F_aero = 0.5 * DRAG_COEFFICIENT * FRONTAL_AREA * AIR_DENSITY * (current_vx**2)
        F_roll = ROLLING_RESISTANCE_COEFFICIENT * self.m * GRAVITATIONAL_ACCEL
        F_tractive = F_inertial + F_aero + F_roll
        
        # --- 2. 将车轮的力/速度转换为电机轴的扭矩/转速 ---
        # 电机扭矩 (Nm)
        motor_torque_nm = (F_tractive * WHEEL_RADIUS) / (GEAR_RATIO * DRIVETRAIN_EFFICIENCY)
        # 电机转速 (rpm)
        motor_speed_rad_s = (current_vx / WHEEL_RADIUS) * GEAR_RATIO
        motor_speed_rpm = motor_speed_rad_s * (60 / (2 * np.pi))

        # --- 3. 根据驱动/制动工况，计算电功率 P_elec ---
        if motor_torque_nm >= 0:
            # --- 驱动模式 ---
            # 机械功率 (kW)
            P_mech_kW = (motor_torque_nm * motor_speed_rad_s) / 1000.0
            # 从SPM模型中获取电机损耗 (kW)
            P_loss_kW = calculate_spm_loss_kw(motor_torque_nm, motor_speed_rpm)
            # 电功率 = 机械功率 + 损耗
            P_elec_kW = P_mech_kW + P_loss_kW
        else:
            # --- 再生制动模式 ---
            # 机械功率为负值
            P_mech_kW = (motor_torque_nm * motor_speed_rad_s) / 1000.0
            # 回收的电功率 = 机械功率 * 回收效率
            # P_elec_kW 此时也为负，代表能量回到电池
            P_elec_kW = P_mech_kW * REGEN_EFFICIENCY

        return P_elec_kW

    def calculate_reward(self, action, is_collision, is_baseline_agent):
        """
        计算当前状态下的奖励,在这里添加了某项奖励之后,去step函数的log_entry中增加相应的指标
        """
        # --- 建议的权重参数 ---
        #W_PROGRESS = 4       # 进度奖励权重
        W_VELOCITY = 8         # 速度跟踪奖励权重
        VELOCITY_STD = 5  
        W_TIME = -0            # 时间惩罚 (每步-0.2分)
        W_ACTION_SMOOTH = -1   # 动作平滑度惩罚权重
        W_ENERGY = -0.01        # 能量消耗惩罚权重
        R_SUCCESS = 50.0        # 成功奖励
        R_COLLISION = -50.0     # 碰撞惩罚
        W_COST_PENALTY = -1.5   # (仅用于PPO baseline) 成本惩罚权重



        # 期望速度，可以设为道路限速
        DESIRED_VELOCITY = 15.0

        # --- 1. 创建一个字典来存储所有奖励分量 ---
        reward_components = {}

        # --- 2. 计算各个基础奖励分量 ---
        #行驶进度奖励 (Progress Reward)
        current_longitudinal_pos = self.get_current_longitudinal_pos()
        progress = current_longitudinal_pos - self.last_longitudinal_pos
        #reward_components['progress'] = W_PROGRESS * progress
        # 效率奖励 (核心修改)
        current_speed = self.state['vx']
        velocity_diff_sq = (current_speed - DESIRED_VELOCITY)**2
        reward_components['velocity_tracking'] = W_VELOCITY * np.exp(-velocity_diff_sq / (2 * VELOCITY_STD**2))

        W_PATH = 1.5       # 路径跟踪奖励的整体权重
        ALPHA = 0.3       # 横向误差的敏感度系数 (调小以适应更大误差范围)
        BETA = 0.5         # 航向误差的敏感度系数

        cte_sq = self.cross_track_error**2
        he_sq = self.heading_error**2

        # 只有当误差同时很小时，这个奖励才高。任何一项误差很大，奖励都趋近于0。
        path_reward = W_PATH * np.exp(-(ALPHA * cte_sq + BETA * he_sq))
        reward_components['path_following'] = path_reward
        
        # 时间惩罚，鼓励车辆尽快完成任务
        reward_components['time_penalty'] = W_TIME
        
        # 舒适性奖励
        action_diff = action - self.last_action
        smoothness_penalty = np.sum(action_diff**2)
        reward_components['action_smoothness'] = W_ACTION_SMOOTH * smoothness_penalty

        acceleration = action[0] * MAX_ACCELERATION
        power_consumption = self._calculate_energy_consumption(acceleration)
        # 注意：对于再生制动，power_consumption 可能为负值，这会给予奖励而非惩罚
        # 如果要考虑绝对能耗（不管正负），可以使用 abs(power_consumption)
        reward_components['energy_consumption'] = W_ENERGY * power_consumption

        # --- 3. 处理PPO基准的成本惩罚 ---
        if is_baseline_agent:
            cost = self.calculate_cost()
            reward_components['cost_penalty'] = W_COST_PENALTY * cost
        else:
            cost = self.calculate_cost()
            reward_components['cost_penalty'] = 0

        # --- 4. 计算步进奖励的总和 ---
        total_reward = sum(reward_components.values())

        # --- 5. 根据终局状态，覆盖奖励值 ---
        if is_collision:
            total_reward += R_COLLISION #这里本来是直接等于R_COLLISION 但是感觉不太对 如果后面训练不出来的话再改回 total_reward = R_COLLISION
        if self.completed:
            total_reward += R_SUCCESS 
            
        reward_components['total_reward'] = total_reward
        
        return reward_components

    def calculate_cost(self):
        potential = self.road.get_potential_at_point(self.state['x'], self.state['y'], self.target_route_str)
        return potential

    def _check_collision(self, all_vehicles):
        """
        使用分离轴定理（SAT）来精确地检测OBB碰撞。
        """
        for v in all_vehicles:
            if v.vehicle_id == self.vehicle_id:
                continue
            
            # 调用我们新的OBB碰撞检测函数
            if check_obb_collision(self, v):
                print(f"!!! COLLISION DETECTED between {self.vehicle_id} and {v.vehicle_id} !!!")
                return True                
        return False

    def _check_off_track(self, max_deviation):
        """
        检查车辆是否偏离参考路径过远
        
        Args:
            max_deviation: 允许的最大横向偏差（单位：像素）
            
        Returns:
            bool: 如果偏离过远返回True，否则返回False
        """
        current_pos = np.array([self.state['x'], self.state['y']])
        
        # 计算到路径中心线的最小距离
        distances = np.linalg.norm(self.path_points_np - current_pos, axis=1)
        min_distance = np.min(distances)
        
        # 如果距离超过最大允许偏差，则判定为偏离轨道
        if min_distance > max_deviation:
            print(f"车辆 {self.vehicle_id} 偏离轨道！距离: {min_distance:.2f}，最大允许: {max_deviation:.2f}")
            return True
        
        return False
        
    def step(self, action, dt, all_vehicles, algo_name="sagi_ppo"):
        """RL Agent的核心函数，移除了越界终止。"""
        self.steps_since_spawn += 1
        #print(f"Raw action from network: accel={action[0]:.2f}, steer={action[1]:.2f}")
        # 1. 解读并执行动作
        acceleration = action[0] * MAX_ACCELERATION
        steering_angle = action[1] * MAX_STEERING_ANGLE
        #steering_angle = action[1] * 0
        self._update_physics(acceleration * self.m, steering_angle, dt)
        
        self.state['vx'] = max(0, min(self.state['vx'], 15))
        self.last_steering_angle = steering_angle
        self.speed_history.append(self.get_current_speed())

        self._update_intersection_status()

        # 2. 检查终止条件 (核心修改：移除了 is_out_of_bounds)
        is_collision = self._check_collision(all_vehicles)
        self._check_completion()

        is_off_track = self._check_off_track(max_deviation=3 * self.road.lane_width)

        terminated = self.completed or is_collision or is_off_track
        truncated = self.steps_since_spawn >= self.max_episode_steps

        # 3. 计算奖励和新的观测值
        is_baseline = (algo_name == "ppo_gru")
        reward_info = self.calculate_reward(action, is_collision, is_baseline_agent=is_baseline)
        total_reward = reward_info['total_reward'] # 直接从字典获取最终奖励

        # --- 5. 获取新的观测值 ---
        base_obs = self.get_base_observation(all_vehicles)
        # b. 获取预测性场景树观测
        scenario_tree_obs = self._get_scenario_tree_observation(all_vehicles)
        # c. 拼接成最终的完整观测
        observation = np.concatenate([base_obs, scenario_tree_obs]).astype(np.float32)

        
        # --- 6. 将本步的所有关键信息存入日志 ---
        cost = self.calculate_cost()
        log_entry = {
            'step': self.steps_since_spawn,
            'action_accel': action[0],
            'action_steer': action[1],
            
            # 从 reward_info 字典中获取所有奖励分量
            'reward_velocity_tracking': reward_info.get('velocity_tracking', 0),
            'reward_path_following': reward_info.get('path_following', 0),
            'reward_action_smoothness_penalty': reward_info.get('action_smoothness', 0),
            'reward_cost_penalty': reward_info.get('cost_penalty', 0),
            'reward_energy_consumption': reward_info.get('energy_consumption', 0),
            'total_reward': total_reward, # 记录最终（可能被终局奖励覆盖的）总奖励
            
            # [核心修正] 完整记录所有用于绘图的未加权原始指标
            'raw_speed': self.state['vx'],
            'raw_cte': self.cross_track_error,
            'raw_he': self.heading_error,
            'raw_smoothness_penalty': np.sum((action - self.last_action)**2),
            'raw_power': self._calculate_energy_consumption(action[0] * MAX_ACCELERATION),
            'raw_cost_potential': cost,

            'ego_vx': self.state['vx'],
            'ego_vy': self.state['vy'],
            'cross_track_error': self.cross_track_error, # 直接使用成员变量
            'heading_error': self.heading_error      # 直接使用成员变量
        }
        self.debug_log.append(log_entry)

        # --- 7. 更新用于下一帧计算的状态变量 ---
        self.last_longitudinal_pos = self.get_current_longitudinal_pos()
        self.last_action = action

        # --- 8. 准备并返回所有信息 ---
        info = {"cost": cost, 'failure': 'collision' if is_collision else None}
        # 将详细的奖励分量添加到info字典中，以便在训练脚本中记录
        info.update({f"reward_{k}": v for k, v in reward_info.items()})
        
        return observation, total_reward, terminated, truncated, info

    def reset(self):
        """重置Agent的状态，用于开始新回合。"""
        # 这个方法的具体实现依赖于环境的重置逻辑
        # 但它内部应重置这些'last'变量
        self.last_longitudinal_pos = self.get_current_longitudinal_pos()
        self.last_action = np.array([0.0, 0.0])
        self.steps_since_spawn = 0

        self.debug_log = []
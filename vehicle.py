"""
Module: vehicle

Overview:
该文件是整个仿真项目中最为核心的动态实体定义文件。它包含了两个关键的类：
- `Vehicle`: 代表由预设规则驱动的背景交通车辆 (NPC)。
- `RLVehicle`: 继承自`Vehicle`，代表由强化学习算法驱动的、需要学习决策的智能体。
这两个类封装了车辆的所有属性和行为，包括物理模型、感知、决策、控制和可视化。

Functions:
- calculate_spm_loss_kw: 根据多项式模型计算SPM电机的功率损耗。
- Vehicle.__init__: 初始化一个背景交通车辆（NPC）。
- Vehicle.initialize_planner: 根据个性和意图初始化车辆的纵向规划器。
- Vehicle._update_intersection_status: 更新车辆与交叉口相关的状态标志。
- Vehicle.find_leader: 查找本车前方的领头车辆。
- Vehicle.update: 背景车辆的主更新循环。
- Vehicle.get_local_observation: 提供一个标准的局部状态向量。
- Vehicle.check_collision_with: 检查与另一辆车的碰撞。
- Vehicle.get_speed_history: 返回车辆的速度历史记录。
- Vehicle.get_current_longitudinal_pos: 获取车辆在路径上的纵向位置。
- Vehicle._update_visual_feedback: 根据规划决策更新车辆的颜色。
- Vehicle._check_completion: 检查车辆是否到达终点。
- Vehicle._get_maneuver_type: 判断车辆的行驶意图（直行、左转、右转）。
- Vehicle._does_path_conflict: 检查与另一辆车的路径是否冲突。
- Vehicle.has_priority_over: 判断本车是否比另一辆车有更高优先级。
- Vehicle._estimate_entry_time: 估算车辆到达交叉口入口的时间。
- Vehicle._update_physics: 更新车辆的物理状态。
- Vehicle.draw: 在Pygame表面上绘制车辆。
- Vehicle._draw_bicycle_model_visualization: 绘制自行车模型的关键元素。
- Vehicle._rotate_and_transform: 旋转点集并变换到屏幕坐标。
- Vehicle._create_wheel_poly: 创建单个车轮的多边形。
- Vehicle.get_current_speed: 获取车辆当前的总速度。
- Vehicle.toggle_debug_info: 切换调试信息的显示。
- Vehicle.get_state: 返回当前车辆状态的字典。
- Vehicle.toggle_bicycle_visualization: 切换自行车模型的可视化显示。
- Vehicle.toggle_path_visualization: 切换路径的可视化显示。
- RLVehicle.__init__: 初始化一个强化学习智能体车辆。
- RLVehicle.plot_reward_history: 绘制智能体在单次回合中的奖励历史。
- RLVehicle.get_observation: 获取RL智能体的完整观测向量。
- RLVehicle.get_base_observation: 获取基础观测（自身+周围车辆）。
- RLVehicle._get_ego_observation: 获取智能体自身的观测状态。
- RLVehicle._get_surrounding_vehicles_observation: 获取周围车辆的观测状态。
- RLVehicle._get_scenario_tree_observation: 获取场景树（预测轨迹）的观测。
- RLVehicle._flatten_and_normalize_trajectory: 展平并归一化预测的轨迹。
- RLVehicle._calculate_energy_consumption: 计算电能消耗。
- RLVehicle._calculate_signed_cross_track_error: 计算有符号的横向跟踪误差。
- RLVehicle.calculate_reward: 计算RL智能体的奖励。
- RLVehicle.calculate_cost: 计算与安全相关的成本。
- RLVehicle._check_collision: 检查RL智能体是否发生碰撞。
- RLVehicle._check_off_track: 检查车辆是否偏离路径太远。
- RLVehicle.step: 执行一个RL动作并返回结果。
- RLVehicle.reset: 重置RL智能体的状态。
"""

import pygame
import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
from config import *
from longitudinal_control import PIDController
from lateral_control import MPCController
from longitudinal_planner import LongitudinalPlanner
from utils import check_obb_collision
from prediction import TrajectoryPredictor

def calculate_spm_loss_kw(T_inst_nm, n_inst_rpm):
    """根据多项式模型计算SPM电机的功率损耗（kW）。

    这是一个用于网格计算的矢量化版本，基于一篇学术论文中描述的
    表面贴装永磁（SPM）电机的损耗模型。

    Args:
        T_inst_nm (np.ndarray or float): 电机的瞬时扭矩 (Nm)。
        n_inst_rpm (np.ndarray or float): 电机的瞬时转速 (rpm)。

    Returns:
        np.ndarray or float: 计算出的电机功率损耗 (kW)，被裁剪为非负值。
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
    _arrival_queue = []

    def __init__(self, road, start_direction, end_direction, vehicle_id):
        """初始化一个背景交通车辆（NPC）。

        此构造函数设置车辆的基本属性，包括其在道路网络中的路径、物理状态、
        几何尺寸以及默认的控制器。它还初始化了与仿真相关的各种状态标志。

        Args:
            road (Road): 车辆行驶所在的道路对象。
            start_direction (str): 车辆的起始方向（如 'north', 'south'）。
            end_direction (str): 车辆的目标方向。
            vehicle_id (int or str): 车辆的唯一标识符。

        Raises:
            ValueError: 如果无法为给定的起止方向生成有效路径。
        """
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
        self.in_queue = False
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
        self.data_log = []
        self.current_sim_time = 0.0

    def initialize_planner(self, personality: str, intent: str):
        """根据分配的个性和意图，创建并初始化车辆的纵向规划器。

        此方法由 `TrafficEnv` 在环境重置时调用，用于动态配置NPC车辆的行为。

        Args:
            personality (str): 车辆的驾驶个性（例如 'aggressive', 'conservative'）。
            intent (str): 车辆在交叉口的意图（例如 'GO', 'YIELD'）。
        """
        self.planner = LongitudinalPlanner(self, personality, intent)

    def _update_intersection_status(self):
        """根据车辆位置更新与交叉口相关的状态标志。

        此方法检查车辆当前是否位于交叉口的物理冲突区域内，并相应地更新
        `is_in_intersection` 和 `has_passed_intersection` 等标志。
        这是一个状态机，用于跟踪车辆穿越交叉口的过程。
        """
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
                if self.in_queue:
                    if self.vehicle_id in Vehicle._arrival_queue:
                        Vehicle._arrival_queue.remove(self.vehicle_id)
                    self.in_queue = False
                    # print(f"[Queue] Vehicle {self.vehicle_id} LEFT.")
        else:
            # 在交叉口内
            self.is_in_intersection = True
            self.is_approaching_decision_point = False
            self.is_yielding = False # 进入后不再让行，必须快速通过
            self.decision_made = True

    def find_leader(self, all_vehicles, lane_width=4):
        """查找本车前方同一“逻辑车道”上的最近的前车。

        该方法通过将其他车辆投影到本车的参考路径上，来判断它们是否在前方以及
        是否在同一车道内。这比简单地比较`move_str`更具鲁棒性，特别是在路径
        交叉或合并的区域。

        Args:
            all_vehicles (list[Vehicle]): 仿真中的所有车辆列表。
            lane_width (float, optional): 用于判断是否在同一车道的横向距离阈值。
                                          Defaults to 4.

        Returns:
            Vehicle or None: 如果找到前车，则返回该车辆对象；否则返回 None。
        """
        leader = None
        min_positive_longitudinal_dist = float('inf')

        # 1. 获取自身在路径上的纵向位置
        ego_pos = np.array([self.state['x'], self.state['y']])
        #   (确保 self.path_points_np 和 self.path_distances 在 __init__ 中已正确初始化)
        if not hasattr(self, 'path_points_np') or self.path_points_np is None or len(self.path_points_np) < 2:
            return None # 自身路径无效，无法找前车
            
        distances_to_ego = np.linalg.norm(self.path_points_np - ego_pos, axis=1)
        ego_proj_index = np.argmin(distances_to_ego)
        ego_longitudinal_pos = self.path_distances[ego_proj_index]

        for other_vehicle in all_vehicles:
            if other_vehicle.vehicle_id == self.vehicle_id:
                continue
                
            other_pos = np.array([other_vehicle.state['x'], other_vehicle.state['y']])
            
            # 2. 将其他车辆投影到 *本车* 的路径上
            distances_to_other = np.linalg.norm(self.path_points_np - other_pos, axis=1)
            other_proj_index = np.argmin(distances_to_other)
            
            # 3. 检查横向距离 ( 是否在同一逻辑车道上)
            #    使用一个稍宽松的阈值，例如半个车道宽度
            lateral_dist = distances_to_other[other_proj_index]
            if lateral_dist > lane_width / 2.0: 
                continue

            # 4. 检查纵向位置 ( 是否在本车前方)
            other_longitudinal_pos = self.path_distances[other_proj_index]
            longitudinal_dist = other_longitudinal_pos - ego_longitudinal_pos
            
            # 只要在前方 (longitudinal_dist > 0)，就可能是前车
            if longitudinal_dist <= 0:
                continue
                
            # 5. 更新最近的前车
            if longitudinal_dist < min_positive_longitudinal_dist:
                min_positive_longitudinal_dist = longitudinal_dist
                leader = other_vehicle
                
        return leader
    
    def update(self, dt, all_vehicles):
        """背景车辆(HV)的主更新循环。

        此方法实现了“规划-控制-执行”的循环。在每个时间步，它首先调用纵向规划器
        获取目标速度，然后调用PID和MPC控制器计算控制指令（油门/刹车和转向），
        最后调用物理模型更新车辆状态。

        Args:
            dt (float): 仿真时间步长 (秒)。
            all_vehicles (list[Vehicle]): 仿真中的所有车辆列表。
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
        self.data_log.append({
            't': getattr(self, 'current_sim_time', 0.0),
            'vehicle_id': self.vehicle_id,
            'x': self.state['x'],
            'y': self.state['y'],
            'vx': self.state['vx'],
            'vy': self.state['vy'],
            'psi': self.state['psi'],
            'speed': self.get_current_speed(),
            'steering': self.last_steering_angle
        })

    def save_trajectory_to_csv(self, filename=None):
        # 在这里设置一个总开关来控制是否实际保存文件
        # 将此值更改为 False 即可禁用保存功能
        SAVE_ENABLED = False

        if not SAVE_ENABLED:
            # 如果开关关闭，打印一条消息并直接返回
            print(f"Vehicle {self.vehicle_id}: 轨迹保存功能已禁用，跳过保存。")
            return

        if not self.data_log:
            print(f"Vehicle {self.vehicle_id}: 没有数据可保存")
            return
            
        if filename is None:
            filename = f"trajectory_vehicle_{self.vehicle_id}.csv"
            
        df = pd.DataFrame(self.data_log)
        df.to_csv(filename, index=False)
        print(f"轨迹数据已保存至: {os.path.abspath(filename)}")

    def get_local_observation(self) -> np.ndarray:
        """提供一个标准的局部状态向量，供TrafficEnv的_get_observation调用。

        此方法对背景车辆（HV）和强化学习智能体（AV）都适用。它计算并返回
        一个归一化的状态向量，包括速度、角速度、路径跟踪误差和路径完成度。

        Returns:
            np.ndarray: 包含车辆归一化局部状态的NumPy数组。
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
        """将碰撞检测作为一个独立的、可被外部调用的方法。

        使用 `check_obb_collision` 辅助函数来检查两个车辆的定向包围盒（OBB）
        是否发生重叠。

        Args:
            other_vehicle (Vehicle): 要检查碰撞的另一辆车。

        Returns:
            bool: 如果发生碰撞，返回 True；否则返回 False。
        """
        return check_obb_collision(self, other_vehicle)

    # --------------------------------------------------------------------------
    #  辅助方法 (Helper Methods)
    # --------------------------------------------------------------------------

    def get_speed_history(self):
        """返回车辆从仿真开始到现在的完整速度历史记录。

        Returns:
            list[float]: 一个包含每个时间步速度值的列表。
        """
        return self.speed_history


    def get_current_longitudinal_pos(self):
        """获取车辆在自身路径上的纵向投影距离。

        该方法计算车辆当前位置在预定义参考路径上的累积行驶距离。

        Returns:
            float: 从路径起点开始的纵向距离。
        """
        current_pos = np.array([self.state['x'], self.state['y']])
        distances_to_ego = np.linalg.norm(self.path_points_np - current_pos, axis=1)
        current_path_index = np.argmin(distances_to_ego)
        return self.path_distances[current_path_index]


    def _update_visual_feedback(self, plan_info: dict):
        """根据规划器返回的详细决策原因来更新车辆颜色。

        此方法用于在可视化界面中直观地展示NPC车辆的当前决策状态。
        例如，让行时显示为橙色，抢行时显示为红色。

        Args:
            plan_info (dict): 从纵向规划器获取的包含决策原因的字典。
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
        """检查车辆是否到达其参考路径的终点。

        如果车辆与路径终点的距离小于一个阈值（10.0米），则将 `completed`
        标志设置为 True。
        """
        dist_to_end = np.linalg.norm([self.state['x'] - self.reference_path[-1][0], self.state['y'] - self.reference_path[-1][1]])
        if dist_to_end < 10.0:
            self.completed = True

    # --------------------------------------------------------------------------
    #  交通规则判断 (Traffic Rule Logic - Called by Planner)
    #  这些方法被规划器调用，以判断路权归属
    # --------------------------------------------------------------------------

    def _get_maneuver_type(self):
        """根据起止方向判断车辆的行驶意图（直行、右转或左转）。

        Returns:
            str: 'straight', 'right', 或 'left'。
        """
        start, end = self.start_direction, self.end_direction
        if (start, end) in [('north', 'south'), ('south', 'north'), ('east', 'west'), ('west', 'east')]: return 'straight'
        if (start, end) in [('north', 'west'), ('south', 'east'), ('east', 'north'), ('west', 'south')]: return 'right'
        return 'left'

    def _does_path_conflict(self, other_vehicle):
        """检查本车路径是否与另一辆车的路径在交叉口存在物理冲突。

        此方法查询 `road` 对象中预定义的路径冲突矩阵。

        Args:
            other_vehicle (Vehicle): 要检查的另一辆车。

        Returns:
            bool: 如果路径冲突，返回 True；否则返回 False。
        """
        my_route = self.move_str
        other_route = other_vehicle.move_str

        is_conflict = self.road.do_paths_conflict(my_route, other_route)
        #print(f"DEBUG CONFLICT CHECK: Is there a conflict between {my_route} (me, #{self.vehicle_id}) and {other_route} (#{other_vehicle.vehicle_id})? -> {is_conflict}")

        return is_conflict

    def has_priority_over(self, other_vehicle):
        """
        判断路权。
        逻辑升级：
        1. 绝对防御：如果对方已经在路口内，我必须让（防止碰撞）。
        2. 全局队列：利用 Vehicle._arrival_queue 锁定先来后到。
        3. 规则辅助：如果排队时间差不多，再用你的直行/右转规则。
        """
        
        # --- 0. [防撞铁律] 绝对防御 (解决后车无视前车的问题) ---
        # 如果对方已经越过停止线（或大半个身子进去了），必须让！
        # 无论算出来我有多少优先权，物理上撞上去就是输。
        if other_vehicle.dist_to_intersection_entry < 0.1:
            # 对方已经在里面了，我必须Yield
            return False 

        # --- 1. [新增] 全局队列管理 (Global Order Allocator) ---
        # 只要进入路口范围（例如30米），就去“拿号”排队
        # 这是一个“只进不出”的逻辑，保证顺序一旦确定就不变
        entry_threshold = 30.0
        
        # 尝试把自己加入全局队列
        if self.dist_to_intersection_entry < entry_threshold and not self.in_queue:
            Vehicle._arrival_queue.append(self.vehicle_id)
            self.in_queue = True
            
        # 尝试把对方加入全局队列（以防对方还没update）
        if other_vehicle.dist_to_intersection_entry < entry_threshold and not other_vehicle.in_queue:
            Vehicle._arrival_queue.append(other_vehicle.vehicle_id)
            other_vehicle.in_queue = True
            
        # 如果我俩都在队列里，直接看谁排在前面！
        # 这就是最强的“锁定”，完全消除了震荡
        if self.in_queue and other_vehicle.in_queue:
            try:
                my_index = Vehicle._arrival_queue.index(self.vehicle_id)
                other_index = Vehicle._arrival_queue.index(other_vehicle.vehicle_id)
                
                # 如果排队名次相差很大（比如他是第1名，我是第5名），直接服从队列
                # 只有当名次紧挨着（diff == 1）且到达时间极其接近时，才允许用 Rule 2/3 插队
                if my_index < other_index:
                    return True  # 我排前面，我有路权
                elif my_index > other_index:
                    return False # 他排前面，我让行
            except ValueError:
                pass # 此时可能有一方刚离开队列，回退到下方逻辑

        # --- 以下是原有的逻辑（作为兜底） ---
        
        # 修复后的时间估算（见下方函数）
        my_time = self._estimate_entry_time()
        other_time = other_vehicle._estimate_entry_time()
        
        # Rule 1: FCFS (基于物理时间)
        if my_time - other_time < -1.5: return True
        if my_time - other_time > 1.5: return False
        
        # Rule 2: 意图 (直 > 右 > 左)
        p_map = {'straight': 3, 'right': 2, 'left': 1}
        if p_map.get(self._get_maneuver_type(), 0) > p_map.get(other_vehicle._get_maneuver_type(), 0):
            return True
        if p_map.get(self._get_maneuver_type(), 0) < p_map.get(other_vehicle._get_maneuver_type(), 0):
            return False
            
        # Rule 3: 让右 (Right of Way)
        # 这里不需要改太复杂，因为上面的全局队列通常已经分出胜负了
        right_of_map = {'west': 'south', 'south': 'east', 'east': 'north', 'north': 'west'}
        if right_of_map.get(other_vehicle.start_direction) == self.start_direction:
            return True # 他在我左边（我在他右边）
        if right_of_map.get(self.start_direction) == other_vehicle.start_direction:
            return False # 他在我右边
            
        # Tie-Breaker
        return self.vehicle_id < other_vehicle.vehicle_id


    def _estimate_entry_time(self) -> float:
        """
        [优化版] 基于虚拟巡航速度的到达时间估算 (TTA)。
        
        原理：为了防止静止车辆(V=0)的时间变成无穷大，我们假设所有车辆
        至少拥有一个“虚拟底座速度”(例如 20km/h)。
        这样，距离越近的车，算出来的时间一定越短，严格保证了前后顺序。
        """
        dist = self.dist_to_intersection_entry
        
        # 1. 绝对优先区：已经在路口内或越过停止线
        if dist <= 0: 
            return -1.0 
        
        # 2. 获取当前物理速度
        current_v = max(0, self.state['vx'])
        
        # 3. 定义虚拟底座速度 (Virtual Floor Speed)
        # 这是路口通行的平均期望速度，例如 5 m/s (18 km/h)
        # 它的作用是：即使车停着，我也按这个速度算时间，确保它是按距离排序的
        V_VIRTUAL = 5.0 
        
        # 4. 核心计算：使用“有效速度”
        # 如果车跑得很快，用真实速度；如果车停着或蠕行，用虚拟速度
        effective_v = max(current_v, V_VIRTUAL)
        
        # 计算基础时间 TTC (Time To Collision/Arrival)
        tta = dist / effective_v
        
        # 5. [可选] 停车惩罚 (Stationary Penalty)
        # 给完全静止的车加一点点微小的延迟（反应时间），
        # 用于解决两车同时到达路口时的细微抖动，但不要加太多
        if current_v < 0.1:
            tta += 0.5  # 0.5秒的反应延迟
            
        return tta

    def _update_physics(self, fx_total, delta, dt):
        """更新车辆的物理状态。

        该方法实现了一个混合物理引擎：
        - 在高速时，使用动力学自行车模型，考虑侧滑和轮胎力。
        - 在低速时，切换到运动学自行车模型，以避免动力学模型在低速下的不稳定性，
          并保证转向行为的真实性。
        它还包括对空气阻力和滚动阻力的计算。

        Args:
            fx_total (float): 施加在车辆上的总纵向力 (牛顿)。
            delta (float): 前轮的转向角 (弧度)。
            dt (float): 仿真时间步长 (秒)。
        """
        psi, vx, vy, psi_dot = self.state['psi'], self.state['vx'], self.state['vy'], self.state['psi_dot']
        
        # 1. 计算纵向力和纵向加速度 (所有速度下通用)
        F_air_drag = 0.5 * AIR_DENSITY * DRAG_COEFFICIENT * FRONTAL_AREA * vx**2 * np.sign(vx)
        F_kinetic_rr_mag = ROLLING_RESISTANCE_COEFFICIENT * self.m * GRAVITATIONAL_ACCEL
        
        if abs(vx) < 0.1: # 低速/静止区 (使用一个小的速度阈值)
            # 简化静摩擦模型：
            # 施加的力(fx_total)必须克服最大静摩擦力(我们近似为动摩擦力 F_kinetic_rr_mag)
            # 才能产生净力。
            
            fx_net_pre_resistance = fx_total - F_air_drag
            
            if abs(fx_net_pre_resistance) < F_kinetic_rr_mag:
                # 施加的力 < 静摩擦力，净力为0
                fx_net = 0.0
            else:
                # 施加的力 > 静摩擦力，计算净力 (阻力方向与施加力相反)
                F_rolling_resistance = F_kinetic_rr_mag * np.sign(fx_net_pre_resistance)
                fx_net = fx_net_pre_resistance - F_rolling_resistance
        else:
            # 高速区：使用动摩擦 (阻力方向与速度相反)
            F_rolling_resistance = F_kinetic_rr_mag * np.sign(vx)
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
        """在Pygame表面上绘制车辆。

        此方法绘制车辆的车身、车轮、风挡，并可选择性地显示调试信息、
        参考路径和自行车模型的可视化元素。

        Args:
            surface (pygame.Surface): 用于绘制的Pygame表面。
            transform_func (callable): 将世界坐标转换为屏幕坐标的函数。
            small_font (pygame.font.Font): 用于渲染调试文本的字体。
            scale (float, optional): 绘图的缩放比例。 Defaults to 1.0.
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
        """在车辆上叠加绘制自行车模型的关键元素。

        此方法用于调试，可视化显示车辆的底盘、车轮方向和速度矢量，
        有助于理解物理模型的行为。

        Args:
            surface (pygame.Surface): 用于绘制的Pygame表面。
            transform_func (callable): 将世界坐标转换为屏幕坐标的函数。
            scale (float): 绘图的缩放比例。
        """
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
        """辅助函数，旋转点集并变换到屏幕坐标。

        Args:
            points (list[tuple[float, float]]): 相对于车辆中心的局部坐标点列表。
            x (float): 车辆的世界x坐标。
            y (float): 车辆的世界y坐标。
            angle (float): 车辆的世界航向角 (弧度)。
            transform_func (callable): 将世界坐标转换为屏幕坐标的函数。

        Returns:
            list[tuple[float, float]]: 变换到屏幕坐标系的点列表。
        """
        rotated_points = []
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        for px, py in points:
            rx = px * cos_a - py * sin_a + x
            ry = px * sin_a + py * cos_a + y
            rotated_points.append(transform_func(rx, ry))
        return rotated_points

    def _create_wheel_poly(self, pos, l, w, x, y, psi, delta, transform_func):
        """辅助函数，创建并变换单个车轮的多边形。

        此函数首先根据转向角`delta`旋转车轮，然后随车身一起进行平移和旋转。

        Args:
            pos (tuple[float, float]): 车轮中心相对于车辆中心的局部坐标。
            l (float): 车轮长度。
            w (float): 车轮宽度。
            x (float): 车辆的世界x坐标。
            y (float): 车辆的世界y坐标。
            psi (float): 车辆的世界航向角 (弧度)。
            delta (float): 车轮的转向角 (弧度)。
            transform_func (callable): 将世界坐标转换为屏幕坐标的函数。

        Returns:
            list[tuple[float, float]]: 变换到屏幕坐标系的车轮多边形顶点列表。
        """
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
        """获取车辆当前的总速度（标量速度，基于 vx 和 vy）。

        Returns:
            float: 车辆的标量速度 (m/s)。
        """
        return np.sqrt(self.state['vx']**2 + self.state['vy']**2)

    def toggle_debug_info(self):
        """切换调试信息的显示。"""
        self.show_debug = not self.show_debug

    def get_state(self):
        """返回当前车辆状态的字典。

        Returns:
            dict: 包含车辆动力学状态的字典。
        """
        return self.state

    def toggle_bicycle_visualization(self):
        """切换自行车模型的可视化显示。"""
        self.show_bicycle_model = not self.show_bicycle_model

    def toggle_path_visualization(self):
        """切换参考路径的可视化显示。"""
        self.show_path_visualization = not self.show_path_visualization


class RLVehicle(Vehicle):
    """
    一个专门为强化学习设计的车辆代理。
    """
    def __init__(self, road, start_direction, end_direction, vehicle_id):
        """初始化一个强化学习智能体车辆。

        此构造函数调用父类的构造函数，然后禁用所有基于规则的规划器和控制器，
        因为RL智能体的行为将由外部策略网络决定。它还初始化了与RL相关的
        特定属性，如奖励计算所需的状态变量和调试日志。

        Args:
            road (Road): 车辆行驶所在的道路对象。
            start_direction (str): 车辆的起始方向。
            end_direction (str): 车辆的目标方向。
            vehicle_id (int or str): 车辆的唯一标识符。
        """
        super().__init__(road, start_direction, end_direction, vehicle_id)
        
        self.target_route_str = f"{start_direction[0].upper()}_{end_direction[0].upper()}"

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
        """绘制智能体在单次回合中的奖励和性能指标历史。

        此方法会生成两个图表：一个显示加权的奖励分量，另一个显示未加权的
        原始性能指标（如速度、CTE等）。这对于调试奖励函数和分析智能体
        行为非常有用。

        Args:
            save_path_base (str, optional): 保存图表的文件路径基名。如果提供，
                图表将保存为PNG文件（例如 `base_rewards_weighted.png`），
                否则将直接显示。 Defaults to None.
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
        """获取RL智能体的完整观测向量。

        此方法是智能体感知世界的入口。它负责收集、处理并拼接所有需要的信息，
        形成一个扁平化的、固定长度的NumPy数组，作为策略网络的输入。
        观测向量包括：
        1.  智能体自身的归一化状态。
        2.  周围最近N辆车的相对状态。
        3.  与冲突NPC相关的“场景树”预测轨迹。

        Args:
            all_vehicles (list[Vehicle]): 仿真中的所有车辆列表。

        Returns:
            np.ndarray: 扁平化的、准备好输入神经网络的完整观测向量。

        Raises:
            ValueError: 如果最终生成的观测向量维度与配置中定义的总维度不匹配。
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
        """获取基础观测向量（自身状态 + 周围车辆状态）。

        此方法提供一个不包含场景树预测的简化版观测，主要用于基线模型
        或不需要预测性信息的场景。

        Args:
            all_vehicles (list[Vehicle]): 仿真中的所有车辆列表。

        Returns:
            np.ndarray: 扁平化的基础观测向量。
        """
        ego_observation = self._get_ego_observation()
        surrounding_obs = self._get_surrounding_vehicles_observation(all_vehicles)

        observation = np.array(ego_observation + surrounding_obs, dtype=np.float32)
        return observation
        
    def _get_ego_observation(self):
        """获取并计算智能体自身的归一化状态作为观测的一部分。

        此辅助方法计算车辆的速度、角速度、路径跟踪误差（CTE）、航向误差（HE）
        和路径完成度，并将它们归一化后返回。

        Returns:
            list[float]: 包含智能体自身归一化状态的列表。
        """
        ego_vx_norm = self.state['vx'] / 15.0 # 使用数值代替GIPPS_V_DESIRED
        ego_vy_norm = self.state['vy'] / 5.0
        ego_psi_dot_norm = self.state['psi_dot'] / 2.0
        
        current_pos = np.array([self.state['x'], self.state['y']])
        distances_to_path = np.linalg.norm(self.path_points_np - current_pos, axis=1)
        current_path_index = np.argmin(distances_to_path)
        self.signed_cross_track_error = self._calculate_signed_cross_track_error()
        self.cross_track_error = abs(self.signed_cross_track_error)
        path_angle = self.reference_path[current_path_index][2]
        self.heading_error = (self.state['psi'] - path_angle + np.pi) % (2 * np.pi) - np.pi
        path_completion = self.get_current_longitudinal_pos() / self.path_distances[-1]
        
        return [
            ego_vx_norm, ego_vy_norm, ego_psi_dot_norm,
            np.tanh(self.signed_cross_track_error / MAX_RELEVANT_CTE),
            self.heading_error / np.pi,
            path_completion
        ]

    def _get_surrounding_vehicles_observation(self, all_vehicles):
        """获取周围N辆最近车辆的相对状态作为观测的一部分。

        此方法扫描感知范围内的所有其他车辆，按距离排序，并提取最近的
        `NUM_OBSERVED_VEHICLES` 辆车的归一化相对位置和速度。如果车辆不足，
        则用零填充以保持观测维度固定。

        Args:
            all_vehicles (list[Vehicle]): 仿真中的所有车辆列表。

        Returns:
            list[float]: 包含周围车辆信息的扁平化列表。
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
        """生成并获取场景树（预测轨迹）作为观测的一部分。

        此方法首先寻找一个即将与智能体发生路径冲突的NPC车辆。如果找到，
        它会调用 `TrajectoryPredictor` 来生成两种假设下的未来轨迹：
        NPC“让行”和NPC“抢行”。然后将这两条轨迹归一化并展平，作为观测的一部分。

        Args:
            all_vehicles (list[Vehicle]): 仿真中的所有车辆列表。

        Returns:
            list[float]: 包含两条预测轨迹信息的扁平化列表。
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
        """辅助函数，用于展平、填充/截断并归一化预测出的轨迹。

        Args:
            trajectory (list[list[float]]): 一个轨迹，格式为
                [[x1, y1, ...], [x2, y2, ...], ...]。

        Returns:
            np.ndarray: 处理后的、一维的、归一化后的轨迹向量。
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
        """根据车辆状态和期望加速度，计算电机的瞬时电功率消耗。

        该模型考虑了车辆的惯性、空气阻力、滚动阻力，并通过电机模型
        将车轮处的机械功率需求转换回电池端的电功率消耗。它区分了驱动
        和再生制动两种工况。

        Args:
            commanded_accel (float): 由智能体动作决定的期望纵向加速度 (m/s^2)。

        Returns:
            float: 计算出的瞬时电功率 (kW)。正值代表消耗，负值代表能量回收。
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

    def _calculate_signed_cross_track_error(self):
        """计算有符号的横向跟踪误差 (sCTE)。

        误差的符号用于指示车辆偏离路径的方向（左侧或右侧）。
        根据约定，正值代表偏向危险的对向车道一侧。

        Returns:
            float: 有符号的横向误差 (米)。
        """
        current_pos = np.array([self.state['x'], self.state['y']])
        distances_to_path = np.linalg.norm(self.path_points_np - current_pos, axis=1)
        current_path_index = np.argmin(distances_to_path)
        
        if current_path_index < len(self.path_points_np) - 1:
            p1 = self.path_points_np[current_path_index]
            p2 = self.path_points_np[current_path_index + 1]
        else:
            p1 = self.path_points_np[current_path_index - 1]
            p2 = self.path_points_np[current_path_index]
            
        path_to_vehicle_vec = current_pos - p1
        path_segment_vec = p2 - p1
        
        cross_product_z = path_segment_vec[0] * path_to_vehicle_vec[1] - path_segment_vec[1] * path_to_vehicle_vec[0]
        
        # 假设 cross_product_z > 0 是左侧 (危险)，您可能需要根据坐标系定义反转这里的 np.sign
        signed_error = np.sign(cross_product_z) * distances_to_path[current_path_index]
        return -signed_error

    def calculate_reward(self, action, is_collision, is_baseline_agent, all_vehicles):
        """计算当前时间步的奖励值。

        这是一个多目标奖励函数，旨在平衡驾驶的多个方面：
        - **效率**: 奖励接近期望速度，并惩罚时间流逝。
        - **安全**: 对碰撞给予巨大惩罚。
        - **舒适度**: 惩罚剧烈的加减速和转向动作。
        - **路径跟踪**: 奖励小的横向和航向误差。
        - **能耗**: 惩罚高的电功率消耗。
        - **成本 (可选)**: 对于基线PPO算法，增加一个基于势场地图的成本惩罚。

        Args:
            action (np.ndarray): 智能体输出的归一化动作 [加速度, 转向]。
            is_collision (bool): 当前时间步是否发生碰撞。
            is_baseline_agent (bool): 是否为基线PPO智能体（以决定是否计算成本惩罚）。

        Returns:
            dict: 一个包含所有奖励分量和总奖励的字典。
        """
        # --- 经过重新平衡的权重 ---
        W_VELOCITY = 4.0          # 速度奖励权重（适当降低）
        VELOCITY_STD = 5.0
        W_TIME = -0.1             # [关键] 必须设置一个负值，制造紧迫感
        W_ACTION_SMOOTH = -1    # 动作平滑度惩罚
        W_ENERGY = -0.01          # 能量消耗惩罚
        R_SUCCESS = 100.0         # 成功时给予巨大奖励
        R_COLLISION = -100.0        # 碰撞时给予巨大惩罚

        # 期望速度
        DESIRED_VELOCITY = 15.0

        reward_components = {}

        # --- 速度奖励 ---
        current_speed = self.state['vx']
        velocity_diff_sq = (current_speed - DESIRED_VELOCITY)**2
        reward_components['velocity_tracking'] = W_VELOCITY * np.exp(-velocity_diff_sq / (2 * VELOCITY_STD**2))

        # --- 路径跟踪奖励 (核心修改) ---
        W_PATH = 1.0              # 路径跟踪权重
        ALPHA = 0.5               # [关键] 横向误差敏感度 (从5.0大幅降低到0.5，拓宽“甜点区”)
        BETA = 0.5                # 航向误差敏感度

        cte_sq = self.signed_cross_track_error**2
        he_sq = self.heading_error**2
        path_reward = W_PATH * np.exp(-(ALPHA * cte_sq + BETA * he_sq))
        reward_components['path_following'] = path_reward
        
        # --- 增加一个与速度相关的转弯惩罚 ---
        # 目标：在转弯时（he_sq大），如果速度（current_speed）还很快，就给予惩罚
        # 这比之前只用 he_sq 更精准
        TURN_PENALTY_FACTOR = -0.5 # 惩罚系数
        turn_penalty = TURN_PENALTY_FACTOR * he_sq * current_speed
        reward_components['turn_penalty'] = turn_penalty

        # --- 时间惩罚 ---
        reward_components['time_penalty'] = W_TIME
        
        # --- 其他惩罚项 ---
        action_diff = action - self.last_action
        smoothness_penalty = np.sum(action_diff**2)
        reward_components['action_smoothness'] = W_ACTION_SMOOTH * smoothness_penalty

        acceleration = action[0] * MAX_ACCELERATION
        power_consumption = self._calculate_energy_consumption(acceleration)
        reward_components['energy_consumption'] = W_ENERGY * power_consumption

        # --- PPO基准的成本惩罚 ---
        if is_baseline_agent:
            cost = self.calculate_cost(all_vehicles)
            reward_components['cost_penalty'] = -1.5 * cost
        else:
            reward_components['cost_penalty'] = 0

        # --- 计算总奖励 ---
        total_reward = sum(reward_components.values())

        # --- 终局奖励 ---
        # 使用直接赋值，让终局信号更强烈
        if is_collision:
            total_reward = R_COLLISION
        if self.completed:
            total_reward = R_SUCCESS
            
        reward_components['total_reward'] = total_reward
        return reward_components

    def _calculate_npc_potential(self, all_vehicles):
        """计算由周围 NPC 车辆产生的势函数。"""
        ego_pos = np.array([self.state['x'], self.state['y']])
        npc_potential = 0.0
        
        # 定义关键参数
        SAFE_MARGIN = 5.0       # 核心安全半径
        BOUNDARY_COST = 10.0    # 在边界处(5m)的基础成本
        COLLISION_FACTOR = 100.0 # 侵入安全区后的激增系数

        for v in all_vehicles:
            if v.vehicle_id == self.vehicle_id:
                continue

            other_pos = np.array([v.state['x'], v.state['y']])
            d = np.linalg.norm(ego_pos - other_pos)

            if d < OBSERVATION_RADIUS:
                dist_cost = 0.0
                
                if d < SAFE_MARGIN:
                    # [核心修正] 内部成本 = 边界基础成本 + 侵入深度惩罚
                    # 这样保证 d=2.99 时，成本肯定大于 10.0
                    intrusion = SAFE_MARGIN - d
                    dist_cost = BOUNDARY_COST + COLLISION_FACTOR * intrusion
                else:
                    # 外部成本：随着距离远离，从 BOUNDARY_COST 开始衰减
                    # 当 d = SAFE_MARGIN 时，分母为1，结果正好是 BOUNDARY_COST
                    dist_cost = BOUNDARY_COST / (d - SAFE_MARGIN + 1.0)

                npc_potential += dist_cost

        return npc_potential

    def calculate_cost(self, all_vehicles):
        """计算当前位置的总成本：地图势函数 + NPC 势函数。"""
        # 地图势
        map_potential = self.road.get_potential_at_point(
            self.state['x'], self.state['y'], self.target_route_str
        )
        # NPC 势
        npc_potential = self._calculate_npc_potential(all_vehicles)

        return map_potential + npc_potential

    def _check_collision(self, all_vehicles):
        """检查智能体是否与任何其他车辆发生碰撞。

        使用基于分离轴定理（SAT）的精确OBB（定向包围盒）碰撞检测。

        Args:
            all_vehicles (list[Vehicle]): 仿真中的所有车辆列表。

        Returns:
            bool: 如果发生碰撞，返回 True；否则返回 False。
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
        """检查车辆是否偏离参考路径过远。

        如果车辆当前位置到参考路径中心线的最小距离超过了给定的最大偏差，
        则认为车辆已偏离轨道。
        
        Args:
            max_deviation (float): 允许的最大横向偏差（米）。
            
        Returns:
            bool: 如果偏离过远返回True，否则返回False。
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
        """执行一个RL动作，并返回环境的响应。

        这是RL智能体与环境交互的核心接口，遵循Gymnasium的 `step` 函数规范。
        它执行以下操作：
        1.  将策略网络输出的归一化动作转换为物理世界的力和转向角。
        2.  调用物理引擎更新车辆状态。
        3.  检查是否触发终止条件（碰撞、完成、偏离轨道）或截断条件（超时）。
        4.  计算当前步的奖励和成本。
        5.  获取新的观测状态。
        6.  记录调试信息。
        7.  返回 (observation, reward, terminated, truncated, info) 元组。

        Args:
            action (np.ndarray): 来自策略网络的归一化动作 [加速度, 转向]。
            dt (float): 仿真时间步长 (秒)。
            all_vehicles (list[Vehicle]): 仿真中的所有车辆列表。
            algo_name (str, optional): 正在使用的算法名称，用于区分是否计算成本。
                                      Defaults to "sagi_ppo".

        Returns:
            tuple: 一个包含以下元素的元组：
                - observation (np.ndarray): 新的观测状态。
                - reward (float): 当前步获得的奖励。
                - terminated (bool): 回合是否因任务完成或失败而终止。
                - truncated (bool): 回合是否因超时等外部原因被截断。
                - info (dict): 包含成本、失败原因和详细奖励分量的调试信息。
        """
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
        is_baseline = algo_name.startswith("ppo")
        reward_info = self.calculate_reward(action, is_collision, is_baseline_agent=is_baseline, all_vehicles=all_vehicles)
        total_reward = reward_info['total_reward'] # 直接从字典获取最终奖励

        # --- 5. 获取新的观测值 ---
        base_obs = self.get_base_observation(all_vehicles)
        # b. 获取预测性场景树观测
        scenario_tree_obs = self._get_scenario_tree_observation(all_vehicles)
        # c. 拼接成最终的完整观测
        observation = np.concatenate([base_obs, scenario_tree_obs]).astype(np.float32)

        
        # --- 6. 将本步的所有关键信息存入日志 ---
        cost = self.calculate_cost(all_vehicles)
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
            'heading_error': self.heading_error,      # 直接使用成员变量
            'signed_cross_track_error': self.signed_cross_track_error
        }
        self.debug_log.append(log_entry)

        if hasattr(self, 'data_log'):
            self.data_log.append({
                't': getattr(self, 'current_sim_time', 0.0),
                'vehicle_id': self.vehicle_id,
                'x': self.state['x'],
                'y': self.state['y'],
                'vx': self.state['vx'],
                'vy': self.state['vy'],
                'psi': self.state['psi'],
                'speed': self.get_current_speed(),
                'steering': self.last_steering_angle
            })

        # --- 7. 更新用于下一帧计算的状态变量 ---
        self.last_longitudinal_pos = self.get_current_longitudinal_pos()
        self.last_action = action

        failure_reason = None
        if is_collision:
            failure_reason = 'collision'
        elif is_off_track:
            failure_reason = 'off_track'
        
        info = {"cost": cost, 'failure': failure_reason}
        # 将详细的奖励分量添加到info字典中，以便在训练脚本中记录
        info.update({f"reward_{k}": v for k, v in reward_info.items()})

        if terminated or truncated:
            info["episode_log"] = self.debug_log.copy()
            # 可选：清空，避免下一回合把旧数据混进来
            self.debug_log = []
        
        return observation, total_reward, terminated, truncated, info

    def reset(self):
        """重置智能体的状态，为新回合做准备。

        此方法在每个新回合开始时被调用。它重置用于计算奖励和判断超时的
        内部状态变量，例如步数计数器、上一帧动作和调试日志。
        注意：车辆的物理状态（位置、速度等）由环境的重置逻辑处理。
        """
        # 这个方法的具体实现依赖于环境的重置逻辑
        # 但它内部应重置这些'last'变量
        self.last_longitudinal_pos = self.get_current_longitudinal_pos()
        self.last_action = np.array([0.0, 0.0])
        self.steps_since_spawn = 0

        self.debug_log = []
        self.data_log = []
"""
@file: prediction.py
@description:
该文件实现了“轨迹预测器”（TrajectoryPredictor）模块。

其核心功能是为强化学习智能体（AV）生成预测性场景树。它通过在内存中
进行短暂的、前瞻性的“沙箱”模拟，来预测背景车辆（HV）在不同潜在意图
（例如“通行” vs “让行”）下的未来轨迹。

这个模块是实现AV意图感知能力的关键。
"""

import numpy as np
from config import SIMULATION_DT, PREDICTION_HORIZON, IDM_PARAMS
from driver_model import IDM

class TrajectoryPredictor:
    """
    通过前瞻性模拟，为场景中的动态实体生成未来轨迹。
    """
    def __init__(self, idm_params_library):
        """
        初始化轨迹预测器。
        
        参数:
            idm_params_library (dict): 从config.py传入的IDM_PARAMS字典，
                                     包含了所有可能的驾驶员个性。
        """
        self.idm_params_library = idm_params_library
        self.dt = SIMULATION_DT
        self.horizon = PREDICTION_HORIZON

    def predict(self, current_av_state: dict, current_hv_state: dict, 
                hv_intent_hypothesis: str, hv_vehicle) -> list:
        """
        [修正后]
        核心预测函数。根据一个假设的HV意图，模拟并返回其未来轨迹。
        现在使用统一的、基于数值输入的IDM接口。
        """
        # --- 步骤1: 创建模拟用的“幽灵”状态 (不变) ---
        ghost_av_state = current_av_state.copy()
        ghost_hv_state = current_hv_state.copy()

        # --- 步骤2: 根据假设，为“幽灵HV”选择合适的IDM模型 (不变) ---
        if hv_intent_hypothesis == 'YIELD':
            params = self.idm_params_library['CONSERVATIVE']
        elif hv_intent_hypothesis == 'GO':
            params = hv_vehicle.planner.idm_params
        else:
            raise ValueError(f"未知的意图假设: {hv_intent_hypothesis}")
        
        ghost_idm = IDM(params)
        
        # --- 步骤3: 在循环中进行前瞻模拟 ---
        predicted_trajectory = []
        for _ in range(self.horizon):
            # a. [核心修改] 准备调用新IDM接口所需的纯数值输入
            v_ego = ghost_hv_state['vx']
            v_lead = None
            gap = None

            if hv_intent_hypothesis == 'YIELD':
                # 在让行假设下，设置虚拟前车（AV）的速度和虚拟距离
                v_lead = ghost_av_state['vx']
                gap = self._estimate_dist_to_entry(ghost_hv_state, hv_vehicle.road, hv_vehicle.move_str)
            
            # b. 调用新的、统一的IDM接口计算目标速度
            target_speed = ghost_idm.get_target_speed(v_ego, v_lead, gap)

            # c. 使用简化的运动学模型更新“幽灵”车辆的状态 (逻辑不变)
            ghost_hv_state['vx'] += (target_speed - ghost_hv_state['vx']) * 0.5 # 简单的速度平滑
            ghost_hv_state['x'] += ghost_hv_state['vx'] * np.cos(ghost_hv_state['psi']) * self.dt
            ghost_hv_state['y'] += ghost_hv_state['vx'] * np.sin(ghost_hv_state['psi']) * self.dt
            
            # (可选) 更新幽灵AV的状态 (逻辑不变)
            ghost_av_state['x'] += ghost_av_state['vx'] * np.cos(ghost_av_state['psi']) * self.dt
            ghost_av_state['y'] += ghost_av_state['vx'] * np.sin(ghost_av_state['psi']) * self.dt

            # d. 计算并存储相对状态 (逻辑不变)
            rel_x = ghost_hv_state['x'] - ghost_av_state['x']
            rel_y = ghost_hv_state['y'] - ghost_av_state['y']
            hv_v = ghost_hv_state['vx']
            
            predicted_trajectory.append((rel_x, rel_y, hv_v))

        return predicted_trajectory

    def _estimate_dist_to_entry(self, ghost_hv_state: dict, road, move_str: str) -> float:
        """
        [修正后]
        一个精确的辅助函数，用于估算“幽灵车”沿着其规划路径到冲突区入口的剩余距离。
        
        参数:
            ghost_hv_state (dict): “幽灵”HV的当前状态字典 {'x': ..., 'y': ...}。
            road (Road): Road对象实例。
            move_str (str): HV的路径标识符，例如 'S_N'。

        返回:
            float: 沿路径到冲突区入口的距离(米)。如果不经过冲突区则返回无穷大。
        """
        # --- 步骤 1: 从Road对象获取路径数据 ---
        route_data = road.routes.get(move_str)
        if not route_data:
            return float('inf')

        # --- 步骤 2: [核心修正] 从 "smoothed" 键中获取路径并进行必要的计算 ---
        smoothed_path = route_data.get("smoothed")
        if not smoothed_path or len(smoothed_path) < 2:
            return float('inf')

        # a. 手动创建 vehicle.py 中的 path_points_np
        path_points_np = np.array([[p[0], p[1]] for p in smoothed_path])
        
        # b. 手动创建 vehicle.py 中的 path_distances
        path_distances = np.insert(np.cumsum(np.sqrt(np.sum(np.diff(path_points_np, axis=0)**2, axis=1))), 0, 0)
        
        # --- 步骤 3: 使用您已有的接口获取冲突区入口点的索引 ---
        entry_index = road.get_conflict_entry_index(move_str)
        if entry_index == -1:
            return float('inf')

        # --- 步骤 4: 从路径距离数组中查询入口点的纵向位置 ---
        entry_longitudinal_pos = path_distances[entry_index]
        
        # --- 步骤 5: 计算“幽灵车”当前在路径上的纵向位置 ---
        current_pos = np.array([ghost_hv_state['x'], ghost_hv_state['y']])
        distances_to_path = np.linalg.norm(path_points_np - current_pos, axis=1)
        current_path_index = np.argmin(distances_to_path)
        current_longitudinal_pos = path_distances[current_path_index]
        
        # --- 步骤 6: 计算剩余距离 ---
        dist = entry_longitudinal_pos - current_longitudinal_pos
        
        return max(0, dist)
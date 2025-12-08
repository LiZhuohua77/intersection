"""
@file: prediction.py
@description:
轨迹预测器（TrajectoryPredictor），支持 IDM 和 ACC 两种驾驶模型的 HV。
"""

import numpy as np
from config import SIMULATION_DT, PREDICTION_HORIZON, IDM_PARAMS, ACC_PARAMS
from driver_model import IDM, ACC


class TrajectoryPredictor:
    """
    通过前瞻性模拟，为场景中的动态实体生成未来轨迹。
    可以根据 HV 的 driver_type 选择 IDM 或 ACC 作为预测模型。
    """
    def __init__(self, idm_params_library=None, acc_params_library=None):
        """
        初始化轨迹预测器。
        
        Args:
            idm_params_library (dict, optional): IDM 参数字典，默认使用 IDM_PARAMS。
            acc_params_library (dict, optional): ACC 参数字典，默认使用 ACC_PARAMS。
        """
        self.idm_params_library = idm_params_library or IDM_PARAMS
        self.acc_params_library = acc_params_library or ACC_PARAMS
        self.dt = SIMULATION_DT
        self.horizon = PREDICTION_HORIZON

    # ================= 核心新增：根据 HV 类型和意图构造“幽灵”模型 ================= #
    def _build_ghost_model(self, hv_vehicle, hv_intent_hypothesis: str):
        """
        根据 HV 的 driver_type 和意图假设，构造一个用于预测的“幽灵”驾驶模型。
        返回一个实现了 get_target_speed(...) 的 IDM 或 ACC 实例。
        """
        driver_type = getattr(hv_vehicle, "driver_type", "IDM")
        driver_type = driver_type.upper()

        planner = getattr(hv_vehicle, "planner", None)

        # --- YIELD 假设：更保守的行为 ---
        if hv_intent_hypothesis == "YIELD":
            if driver_type == "ACC":
                # 如果有专门的保守 ACC 参数就用，没有就退回 DEFAULT
                if "CONSERVATIVE" in self.acc_params_library:
                    params = self.acc_params_library["CONSERVATIVE"].copy()
                else:
                    params = self.acc_params_library["DEFAULT"].copy()
                model = ACC(params)
            else:
                params = self.idm_params_library["CONSERVATIVE"].copy()
                model = IDM(params)

        # --- GO 假设：尽量用当前 planner 的参数 ---
        elif hv_intent_hypothesis == "GO":
            base_params = None

            if planner is not None:
                # 如果你按之前的建议改了 LongitudinalPlanner，会有 model_params
                if hasattr(planner, "model_params"):
                    base_params = planner.model_params.copy()
                # 兼容老版本：只有 idm_params 的情况
                elif hasattr(planner, "idm_params"):
                    base_params = planner.idm_params.copy()

            if driver_type == "ACC":
                if base_params is None:
                    # 没有就退回 ACC 的 DEFAULT
                    base_params = self.acc_params_library.get(
                        "DEFAULT", list(self.acc_params_library.values())[0]
                    ).copy()
                model = ACC(base_params)
            else:
                if base_params is None:
                    base_params = self.idm_params_library["NORMAL"].copy()
                model = IDM(base_params)
        else:
            raise ValueError(f"未知的意图假设: {hv_intent_hypothesis}")

        return model

    # ====================================================================== #
    def predict(self, current_av_state: dict, current_hv_state: dict,
                hv_intent_hypothesis: str, hv_vehicle) -> list:
        """
        根据一个假设的 HV 意图，模拟并返回其未来轨迹。

        Args:
            current_av_state (dict): AV 当前状态（x, y, vx, psi 等）。
            current_hv_state (dict): HV 当前状态（x, y, vx, psi 等）。
            hv_intent_hypothesis (str): 'GO' 或 'YIELD'。
            hv_vehicle: 场景中对应的 HV 对象，用于读取 driver_type 和 planner 信息。

        Returns:
            list: 长度为 self.horizon 的列表，每个元素是 (rel_x, rel_y, hv_v)。
        """
        # --- 1. 创建模拟用的“幽灵”状态（拷贝避免影响真实对象） ---
        ghost_av_state = current_av_state.copy()
        ghost_hv_state = current_hv_state.copy()

        # --- 2. 根据假设和 HV 类型构造“幽灵”驾驶模型 ---
        ghost_model = self._build_ghost_model(hv_vehicle, hv_intent_hypothesis)

        # --- 3. 前瞻模拟 ---
        predicted_trajectory = []
        for _ in range(self.horizon):
            v_ego = ghost_hv_state['vx']
            v_lead = None
            gap = None

            if hv_intent_hypothesis == 'YIELD':
                # 在让行假设下，假设 AV 在前方形成一个“虚拟前车”
                v_lead = ghost_av_state['vx']
                gap = self._estimate_dist_to_entry(
                    ghost_hv_state,
                    hv_vehicle.road,
                    hv_vehicle.move_str
                )

            # 用统一接口计算目标速度（不管是 IDM 还是 ACC）
            target_speed = ghost_model.get_target_speed(v_ego, v_lead, gap)

            # 简单的一阶滞后：速度向 target_speed 平滑逼近
            ghost_hv_state['vx'] += (target_speed - ghost_hv_state['vx']) * 0.5
            ghost_hv_state['x'] += ghost_hv_state['vx'] * np.cos(ghost_hv_state['psi']) * self.dt
            ghost_hv_state['y'] += ghost_hv_state['vx'] * np.sin(ghost_hv_state['psi']) * self.dt

            # 可选：AV 也向前走一点（保持相对位置合理）
            ghost_av_state['x'] += ghost_av_state['vx'] * np.cos(ghost_av_state['psi']) * self.dt
            ghost_av_state['y'] += ghost_av_state['vx'] * np.sin(ghost_av_state['psi']) * self.dt

            # 相对状态
            rel_x = ghost_hv_state['x'] - ghost_av_state['x']
            rel_y = ghost_hv_state['y'] - ghost_av_state['y']
            hv_v = ghost_hv_state['vx']

            predicted_trajectory.append((rel_x, rel_y, hv_v))

        return predicted_trajectory

    def _estimate_dist_to_entry(self, ghost_hv_state: dict, road, move_str: str) -> float:
        """
        估算“幽灵 HV”沿着其规划路径到冲突区入口的剩余距离。
        """
        route_data = road.routes.get(move_str)
        if not route_data:
            return float('inf')

        smoothed_path = route_data.get("smoothed")
        if not smoothed_path or len(smoothed_path) < 2:
            return float('inf')

        path_points_np = np.array([[p[0], p[1]] for p in smoothed_path])
        path_distances = np.insert(
            np.cumsum(np.sqrt(np.sum(np.diff(path_points_np, axis=0) ** 2, axis=1))),
            0,
            0
        )

        entry_index = road.get_conflict_entry_index(move_str)
        if entry_index == -1:
            return float('inf')

        entry_longitudinal_pos = path_distances[entry_index]

        current_pos = np.array([ghost_hv_state['x'], ghost_hv_state['y']])
        distances_to_path = np.linalg.norm(path_points_np - current_pos, axis=1)
        current_path_index = np.argmin(distances_to_path)
        current_longitudinal_pos = path_distances[current_path_index]

        dist = entry_longitudinal_pos - current_longitudinal_pos
        return max(0, dist)

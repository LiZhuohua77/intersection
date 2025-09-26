"""
@file: driver_models.py
@description:
该文件定义了用于模拟人类驾驶行为的微观交通模型。
目前，它包含了智能驾驶员模型（IDM）的实现，该模型能够
根据不同的参数集模拟从保守到激进的多种驾驶风格，并能处理
常规跟驰和交叉口冲突两种核心场景。
"""
from config import *
import numpy as np

class IDM:
    """
    智能驾驶员模型 (Intelligent Driver Model)。
    
    该模型根据与前车（或虚拟冲突对象）的距离和速度差，
    并结合自身的驾驶“个性”（由参数定义），来计算期望的加速度。
    
    参考文献:
    Treiber, M., Hennecke, A., & Helbing, D. (2000). Congested traffic states 
    in empirical observations and microscopic simulations. Physical Review E.
    """
    def __init__(self, params: dict):
        """
        初始化IDM模型。
        
        参数:
            params (dict): 一个包含IDM参数的字典，从 config.py 的 IDM_PARAMS 中获取。
                           需要包含 'v0', 'T', 'a', 'b', 's0'。
        """
        self.v0 = params['v0']  # 期望速度 (m/s)
        self.T = params['T']    # 安全时间间隔 (s)
        self.a = params['a']    # 最大加速度 (m/s^2)
        self.b = params['b']    # 舒适减速度 (m/s^2) - 注意：这里用正值
        self.s0 = params['s0']  # 最小安全车头间距 (m)
        self.delta = 4.0        # 加速指数 (通常固定为4)

    def _calculate_idm_accel(self, v: float, delta_v: float, s: float) -> float:
        """
        IDM核心加速度计算公式的私有辅助函数。
        
        参数:
            v (float): 自身车辆速度 (m/s)
            delta_v (float): 与前车的相对速度 (v_ego - v_lead) (m/s)
            s (float): 与前车的有效车头间距 (m)
            
        返回:
            float: 计算出的加速度 (m/s^2)
        """
        # 防止除以零错误
        s = max(s, 1e-6)

        # 1. 期望的动态间距 s_star
        s_star = self.s0 + max(0, v * self.T + (v * delta_v) / (2 * np.sqrt(self.a * self.b)))

        # 2. 自由流加速项 和 交互减速项
        accel_free_flow = self.a * (1 - (v / self.v0)**self.delta)
        accel_interaction = self.a * (s_star / s)**2
        
        # 3. 最终加速度是两者之差
        return accel_free_flow - accel_interaction

    def get_target_speed(self, v_ego: float, v_lead: float = None, gap: float = None) -> float:
        """
        [重构后] 统一的接口，根据输入的纯数值计算目标速度。
        
        参数:
            v_ego (float): 自身车辆速度 (m/s)。
            v_lead (float, optional): 前车/冲突车辆的速度 (m/s)。
            gap (float, optional): 与前车/冲突点的有效距离 (m)。

        返回:
            float: 计算出的目标速度(m/s)。
        """
        if v_lead is None or gap is None:
            # --- 自由流 ---
            acceleration = self.a * (1 - (v_ego / self.v0)**self.delta)
        else:
            # --- 跟驰或交叉口交互 ---
            delta_v = v_ego - v_lead
            acceleration = self._calculate_idm_accel(v_ego, delta_v, gap)

        target_speed = v_ego + acceleration * SIMULATION_DT
        return max(0, target_speed)
"""
Module: driver_model

Overview:
该文件定义了用于模拟人类驾驶行为的微观交通模型。
目前，它包含了智能驾驶员模型（IDM）的实现，该模型能够
根据不同的参数集模拟从保守到激进的多种驾驶风格，并能处理
常规跟驰和交叉口冲突两种核心场景。

Classes:
- IDM: 实现了智能驾驶员模型（Intelligent Driver Model）。

Methods:
- IDM.__init__: 初始化IDM模型。
- IDM.update_parameters: 允许在初始化后更新IDM模型的参数。
- IDM._calculate_idm_accel: IDM核心加速度计算公式的私有辅助函数。
- IDM.get_target_speed: 统一接口计算目标速度，包含启动助力逻辑。
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
        """初始化IDM模型。
        
        Args:
            params (dict): 一个包含IDM参数的字典，从 config.py 的 IDM_PARAMS 中获取。
                           需要包含 'v0', 'T', 'a', 'b', 's0'。
        """
        self.v0 = params['v0']  # 期望速度 (m/s)
        self.T = params['T']    # 安全时间间隔 (s)
        self.a = params['a']    # 最大加速度 (m/s^2)
        self.b = params['b']    # 舒适减速度 (m/s^2) - 注意：这里用正值
        self.s0 = params['s0']  # 最小安全车头间距 (m)
        self.delta = 4.0        # 加速指数 (通常固定为4)

# ==================== [新增方法] ====================
    def update_parameters(self, params: dict):
        """允许在初始化后更新IDM模型的参数。

        此方法主要用于课程学习等场景，其中驾驶员模型的行为（如期望速度）
        需要根据训练进度动态调整。它只更新在 `params` 字典中提供的键值对，
        未提供的参数将保持其现有值。

        Args:
            params (dict): 一个包含要更新的IDM参数的字典。
                           键可以是 'v0', 'T', 'a', 'b', 's0' 中的任意一个或多个。
        """
        # 使用 .get(key, self.existing_value) 来确保只更新传入字典中存在的键
        self.v0 = params.get('v0', self.v0)
        self.T = params.get('T', self.T)
        self.a = params.get('a', self.a)
        self.b = params.get('b', self.b)
        self.s0 = params.get('s0', self.s0)
        # delta 通常不需要更新
        
        # (可选) 添加一个打印语句，方便调试确认参数已更新
        # print(f"  -> IDM parameters updated: New v0 = {self.v0:.2f}") 
    # ====================================================

    def _calculate_idm_accel(self, v: float, delta_v: float, s: float) -> float:
        """IDM核心加速度计算公式的私有辅助函数。

        该函数根据当前车辆的速度、与前车的相对速度以及它们之间的有效间距，
        计算出瞬时加速度。它由自由流加速项和交互减速项两部分组成。

        Args:
            v (float): 自身车辆速度 (m/s)。
            delta_v (float): 与前车的相对速度 (v_ego - v_lead) (m/s)。
            s (float): 与前车的有效车头间距 (m)。
            
        Returns:
            float: 计算出的加速度 (m/s^2)。
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

    def get_target_speed(self, v_ego: float, v_lead: float = None, gap: float = None) -> float: # 返回类型改为 float
        """统一接口计算目标速度，包含启动助力逻辑。

        此方法首先根据标准的IDM跟驰模型计算期望加速度。如果车辆处于自由流状态
        （没有前车），它将向期望速度 `v0` 加速。如果存在前车，它将根据与前车的
        相对速度和间距进行调整。

        此外，该方法包含一个“启动助力”逻辑，用于解决车辆在低速时可能因模型
        计算出微小负加速度而“卡死”的问题。当车辆速度极低但前方有足够空间时，
        会强制施加一个小的正加速度以帮助启动。

        最后，基于计算出的最终加速度，通过短时前瞻（PID_LOOKAHEAD_TIME）来
        估算并返回目标速度。

        Args:
            v_ego (float): 自车当前速度 (m/s)。
            v_lead (float, optional): 前车速度 (m/s)。如果为 None，则按自由流处理。
                                      Defaults to None.
            gap (float, optional): 与前车的有效车头间距 (m)。如果为 None，则按自由流处理。
                                   Defaults to None.

        Returns:
            float: 计算出的非负目标速度 (m/s)。
        """
        # --- 1. 先计算标准的 IDM 加速度 ---
        if v_lead is None or gap is None:
            # 自由流
            acceleration = self.a * (1 - (v_ego / self.v0)**self.delta)
            # reason = 'FREE_FLOW' # reason 仅用于调试或更复杂的逻辑，这里不再需要返回
        else:
            # 跟驰或交叉口交互
            delta_v = v_ego - v_lead
            acceleration = self._calculate_idm_accel(v_ego, delta_v, gap)
            # reason = 'CAR_FOLLOW'

        # --- 2. 启动助力逻辑：覆盖加速度 ---
        STARTUP_THRESHOLD_SPEED = 0.5 # 低于此速度认为可能卡死 (保持您觉得合适的值)
        MIN_STARTUP_ACCEL_FACTOR = 0.3 # 确保这个值足够克服阻力 (根据计算是 > 0.1-0.15)
        
        # 条件：当前速度很低，IDM计算出的加速度要求刹车或静止，但前方确实有空间
        gap_check = gap # 使用传入的 gap 值进行判断
        if v_ego < STARTUP_THRESHOLD_SPEED and acceleration <= 0.0 and \
        gap_check is not None and gap_check > self.s0 * 1.1: 
            
            # 直接覆盖加速度为一个小的正值
            acceleration = MIN_STARTUP_ACCEL_FACTOR * self.a 
            # print(f"  -> IDM applying startup boost! Overriding accel to: {acceleration:.2f}") # 可以保留调试打印
            # reason += '_BOOST' 
        PID_LOOKAHEAD_TIME = 0.5
        # --- 3. 使用最终确定的加速度计算目标速度 ---
        target_speed = v_ego + acceleration * PID_LOOKAHEAD_TIME
        
        # --- 4. 返回非负的目标速度 (float) ---
        return max(0.0, target_speed) # 确保返回 float 且非负
    
class ACC:
    """
    自适应巡航控制 (Adaptive Cruise Control) 驾驶员模型。

    这里使用一个比较经典的“恒定时距 + 线性控制”形式：
    - 没有前车时：尽量跟随设定巡航速度 v_set
    - 有前车时：根据期望车头时距 T 和最小距离 s0 来调节加速度

    加速度结构：
        a = k_p * (gap - s_des)         # 间距误差项（太近减速、太远加速）
          + k_d * (v_lead - v_ego)      # 相对速度项（前车慢就减速）
          + k_v * (v_set - v_ego)       # 巡航速度项（没有前车时主要起作用）

    与 IDM 一样，提供:
        - __init__(params)
        - update_parameters(params)
        - get_target_speed(v_ego, v_lead=None, gap=None)
    """

    def __init__(self, params: dict):
        """初始化 ACC 模型。

        Args:
            params (dict): ACC 参数字典（建议在 config.py 里定义），需要包含：
                - 'v_set':  期望巡航速度 (m/s)
                - 'T':      期望时距 (s)
                - 'a_max':  最大加速度 (m/s^2)
                - 'b_comf': 舒适减速度 (m/s^2，正值)
                - 's0':     最小间距 (m)
                - 'k_p':    间距误差增益
                - 'k_d':    相对速度增益
                - 'k_v':    巡航速度增益
        """
        self.v_set  = params["v_set"]   # 巡航设定速度
        self.T      = params["T"]       # 期望时距
        self.a_max  = params["a_max"]   # 最大加速度
        self.b_comf = params["b_comf"]  # 舒适减速度 (正值)
        self.s0     = params["s0"]      # 最小间距

        # 控制增益可以给默认值，这样 config 里可以省略
        self.k_p = params.get("k_p", 0.4)
        self.k_d = params.get("k_d", 0.8)
        self.k_v = params.get("k_v", 0.3)

    def update_parameters(self, params: dict):
        """允许在初始化后更新 ACC 模型参数（和 IDM 一致的接口）。

        只更新 params 中给出的键，其余保持不变。
        """
        self.v_set  = params.get("v_set",  self.v_set)
        self.T      = params.get("T",      self.T)
        self.a_max  = params.get("a_max",  self.a_max)
        self.b_comf = params.get("b_comf", self.b_comf)
        self.s0     = params.get("s0",     self.s0)
        self.k_p    = params.get("k_p",    self.k_p)
        self.k_d    = params.get("k_d",    self.k_d)
        self.k_v    = params.get("k_v",    self.k_v)

    def _calculate_acc_accel(self, v_ego: float, v_lead: float, gap: float) -> float:
        """ACC 的核心加速度计算公式。

        Args:
            v_ego (float): 自车速度 (m/s)
            v_lead (float): 前车速度 (m/s)
            gap (float): 车头间距 (m)

        Returns:
            float: 计算得到的加速度 (m/s^2)，已做饱和处理。
        """
        gap = max(gap, 1e-6)

        # 期望间距（恒定时距策略）
        s_des = self.s0 + self.T * v_ego

        # 间距误差（为正：太远，应该加速）
        e_gap = gap - s_des

        # 相对速度（为负：前车比自车慢，应该减速）
        rel_v = v_lead - v_ego

        # 线性控制律
        accel = (
            self.k_p * e_gap +      # 间距误差
            self.k_d * rel_v +      # 相对速度
            self.k_v * (self.v_set - v_ego)  # 巡航速度
        )

        # 饱和：不超过最大加速度 / 舒适减速度
        accel = float(np.clip(accel, -self.b_comf, self.a_max))
        return accel

    def get_target_speed(self, v_ego: float, v_lead: float = None, gap: float = None) -> float:
        """统一接口计算 ACC 的目标速度。

        Args:
            v_ego (float): 自车当前速度 (m/s)
            v_lead (float, optional): 前车速度 (m/s)，如果为 None 视为无前车。
            gap (float, optional): 车头间距 (m)，如果为 None 视为无前车。

        Returns:
            float: ACC 计算出的非负目标速度 (m/s)
        """
        # --- 1. 计算基础加速度 ---
        if v_lead is None or gap is None:
            # 没有前车：只对齐巡航速度
            accel = self.k_v * (self.v_set - v_ego)
            accel = float(np.clip(accel, -self.b_comf, self.a_max))
        else:
            # 有前车：使用 ACC 控制律
            accel = self._calculate_acc_accel(v_ego, v_lead, gap)

        # --- 2. 启动助力逻辑（和 IDM 类似，防止低速卡死） ---
        STARTUP_THRESHOLD_SPEED = 0.5
        MIN_STARTUP_ACCEL_FACTOR = 0.3

        if (
            v_ego < STARTUP_THRESHOLD_SPEED
            and accel <= 0.0
            and gap is not None
            and gap > self.s0 * 1.1
        ):
            accel = max(accel, MIN_STARTUP_ACCEL_FACTOR * self.a_max)

        # --- 3. 前瞻一小段时间，得到目标速度 ---
        PID_LOOKAHEAD_TIME = 0.5  # 可以和 IDM 保持一致，或从 config 中读取
        target_speed = v_ego + accel * PID_LOOKAHEAD_TIME

        # --- 4. 保证非负速度 ---
        return max(0.0, float(target_speed))

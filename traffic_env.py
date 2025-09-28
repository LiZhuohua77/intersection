"""
@file: traffic_env.py
@description:
该文件定义了一个遵循 Gymnasium (前身为 Gym) 标准接口的十字路口交通仿真环境。
`TrafficEnv` 类是强化学习算法与底层仿真组件（如道路、交通管理器、车辆）进行交互的核心桥梁。
它将复杂的仿真逻辑封装成一个标准的、可供RL算法调用的环境。

核心功能与设计:

1.  **Gymnasium 接口封装:**
    - `TrafficEnv` 继承自 `gym.Env`，并实现了其核心方法：`__init__` (初始化)、
      `reset` (重置回合)、`step` (执行单步)，以及 `metadata` 属性。
    - 这种标准化的封装使得该环境可以与任何兼容 Gymnasium 的强化学习库（如
      Stable Baselines3, Tianshou, RLlib）无缝对接。

2.  **环境组件的整合:**
    - 在 `__init__` 方法中，环境整合了底层的仿真模块，包括 `Road` (定义地图几何与车辆路径)
      和 `TrafficManager` (管理所有车辆的生成、更新和移除)。`TrafficEnv` 在顶层扮演着
      “协调者”的角色。

3.  **动作与观测空间定义:**
    - `action_space`: 定义为一个形状为 (2,) 的连续空间（Box），通常代表RL智能体输出的
      归一化（-1到1）的加速度和转向指令。
    - `observation_space`: 定义为一个固定维度（在此为26）的连续空间（Box），代表了
      智能体在每个时间步能感知到的所有状态信息。

4.  **场景化重置 (Scenario-based Reset):**
    - `reset` 方法支持通过 `options` 字典传入一个 `scenario` 名称。这允许在训练开始时
      设置特定的交通场景（例如，左转避让直行车、无保护汇入等）。
    - 这个功能对于课程学习（从简单任务开始逐步增加难度）和对特定危险场景的专项测试
      至关重要。

5.  **解耦的步进逻辑 (Decoupled Step Logic):**
    - `step` 方法清晰地划分了职责：
        a. 首先，调用 `RLVehicle` 实例自身的 `step` 方法，将RL算法输出的 `action`
           应用到智能体车辆上，并由车辆自身计算奖励和是否终止。
        b. 随后，调用 `TrafficManager` 的方法来更新所有背景车辆的状态。
    - 这种将奖励计算和状态更新等逻辑部分下放到具体车辆类中的设计，使得环境层的代码
      更简洁，职责更分明。
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame 
import time
import random

from road import Road
from traffic import TrafficManager
from config import * 
from vehicle import RLVehicle 
from prediction import TrajectoryPredictor

class TrafficEnv(gym.Env):
    """一个遵循Gymnasium接口的十字路口交通仿真环境。"""
    
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, scenario: str = "random"):
        super().__init__()

        self.default_scenario = scenario
        # 1. 初始化环境组件
        self.road = Road()
        self.traffic_manager = TrafficManager(self.road, max_vehicles=2)
        self.rl_agent = None
        self.dt = SIMULATION_DT

        self.predictor = TrajectoryPredictor(IDM_PARAMS)

        # 2. [重构] 定义动作和观测空间
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # 使用 config.py 中自动计算的总维度
        observation_dim = TOTAL_OBS_DIM 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(observation_dim,), dtype=np.float32)
        print(f"环境已初始化，观测空间维度为: {observation_dim}")

        self.current_algo = 'sagi_ppo'
        
        # 是否启用场景感知模式（用于场景树生成）
        self.scenario_aware = False  # 默认关闭，可以通过reset的options参数开启

    def reset(self, seed=None, options=None):
        """重置环境到初始状态，可以根据options设置特定场景。"""
        super().reset(seed=seed)
        
        # 从options字典中获取场景名称，如果没有则默认为"random"
        if options and "scenario" in options:
            scenario = options.get("scenario")
        else:
            scenario = self.default_scenario
        
        # 检查是否启用场景感知模式
        self.scenario_aware = options.get("scenario_aware", False) if options else False
        
        self.traffic_manager.vehicle_id_counter = 1
        
        # --- 步骤1: 设置场景并生成车辆 ---
        self.rl_agent = self.traffic_manager.setup_scenario(scenario)
        
        if self.rl_agent is None:
            # (处理纯背景车流场景的逻辑保持不变)
            dummy_observation = np.zeros(self.observation_space.shape, dtype=np.float32)
            info = {"is_background_only": True, "scenario": scenario}
            return dummy_observation, info

        # --- 步骤2: [核心新增] 为HV分配个性和意图 ---
        personalities = list(IDM_PARAMS.keys())
        intents = ['GO', 'YIELD']
        
        for hv in self.traffic_manager.vehicles:
            if not getattr(hv, 'is_rl_agent', False):
                personality = random.choice(personalities)
                intent = random.choice(intents)
                hv.initialize_planner(personality, intent)

        self.current_algo = options.get("algo", "sagi_ppo") if options else "sagi_ppo"

        observation = self.rl_agent.get_observation(self.traffic_manager.vehicles)
        info = self._get_info()
        
        return observation, info

    
    def step(self, action):
        """执行一个时间步。"""
        # 检查是否为只有背景车辆的场景
        if self.rl_agent is None:
            # 对于纯背景车辆场景，我们只更新背景车辆，不涉及RL智能体
            self.traffic_manager.update_background_traffic(self.dt)
            
            # 检查是否所有背景车辆都已完成路径
            all_completed = all(vehicle.completed for vehicle in self.traffic_manager.vehicles)
            
            # 返回dummy观测、奖励和终止信号
            dummy_observation = np.zeros(self.observation_space.shape, dtype=np.float32)
            reward = 0.0
            terminated = all_completed
            truncated = False
            info = {
                "is_background_only": True,
                "background_vehicles": len(self.traffic_manager.vehicles),
                "completed_vehicles": len(self.traffic_manager.completed_vehicles_data)
            }
            
            return dummy_observation, reward, terminated, truncated, info
            
        # 1. 将所有工作交给 RL Agent 的 step 方法
        observation, reward, terminated, truncated, info = self.rl_agent.step(
            action, self.dt, self.traffic_manager.vehicles
        )

        # 2. 更新背景车辆
        self.traffic_manager.update_background_traffic(self.dt)
        
        info.update(self._get_info())
        
        return observation, reward, terminated, truncated, info


    def _get_info(self):
        return {"agent_speed": self.rl_agent.get_current_speed() if self.rl_agent else 0}

    def _get_environment_state(self):
        """
        获取当前环境的完整状态表示，用于场景树生成。
        
        返回:
            dict: 包含所有车辆状态的环境状态
        """
        env_state = {
            'time': time.time(),  # 当前时间戳
            'vehicles': []
        }
        
        # 收集所有车辆的状态
        for vehicle in self.traffic_manager.vehicles:
            vehicle_state = {
                'id': getattr(vehicle, 'vehicle_id', 'unknown'),
                'is_rl_agent': getattr(vehicle, 'is_rl_agent', False),
                'state': vehicle.state.copy() if hasattr(vehicle, 'state') else {},
                'start': vehicle.start_direction if hasattr(vehicle, 'start_direction') else 'unknown',
                'destination': vehicle.end_direction if hasattr(vehicle, 'end_direction') else 'unknown',
            }
            env_state['vehicles'].append(vehicle_state)
        
        return env_state

    def _get_info(self):
        return {"agent_speed": self.rl_agent.get_current_speed() if self.rl_agent else 0}

    @property
    def agent_vehicle(self):
        """返回场景中的RL Agent车辆对象。"""
        # 正确的做法是从 traffic_manager 中获取车辆列表
        if self.traffic_manager and self.traffic_manager.vehicles:
            for vehicle in self.traffic_manager.vehicles:
                if isinstance(vehicle, RLVehicle):
                    return vehicle

        return None
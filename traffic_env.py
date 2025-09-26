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

    def __init__(self):
        super().__init__()
        
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
        scenario = options.get("scenario", "random") if options else "random"
        
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
                # 这个新方法需要在您的Vehicle类中实现
                hv.initialize_planner(personality, intent)

        self.current_algo = options.get("algo", "sagi_ppo") if options else "sagi_ppo"

        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def _get_observation(self):
        """
        [最终架构] 
        环境负责构建完整的、包含预测的观测。
        """
        # 1. 从RL Agent获取基础观测 (不含场景树)
        base_obs = self.rl_agent.get_base_observation(self.traffic_manager.vehicles)
        
        # 2. 寻找冲突的HV
        relevant_hv = self.traffic_manager.get_relevant_hv_for(self.rl_agent)
        
        if not relevant_hv:
            # 没有冲突车，用0填充预测部分
            flat_yield_traj = np.zeros(PREDICTION_HORIZON * FEATURES_PER_STEP)
            flat_go_traj = np.zeros(PREDICTION_HORIZON * FEATURES_PER_STEP)
        else:
            # 3. 调用环境自身的预测器生成轨迹
            yield_traj = self.predictor.predict(
                current_av_state=self.rl_agent.state, 
                current_hv_state=relevant_hv.state,
                hv_intent_hypothesis='YIELD',
                hv_vehicle=relevant_hv
            )
            go_traj = self.predictor.predict(
                current_av_state=self.rl_agent.state, 
                current_hv_state=relevant_hv.state,
                hv_intent_hypothesis='GO',
                hv_vehicle=relevant_hv
            )
            
            # 4. 展平并归一化轨迹
            flat_yield_traj = self._flatten_and_normalize_trajectory(yield_traj)
            flat_go_traj = self._flatten_and_normalize_trajectory(go_traj)

        # 5. 最终拼接：基础观测 + 两条预测轨迹
        final_obs = np.concatenate([
            base_obs,
            flat_yield_traj,
            flat_go_traj
        ]).astype(np.float32)
        
        return final_obs
    
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
            
        # --- 核心流程 ---
        # 1. [暂时保留] 调用RL Agent自身的step方法
        #    RL Agent内部的 get_observation 仍会被调用，我们需要确保它能获取到预测信息
        _observation_from_agent, reward, terminated, truncated, info = self.rl_agent.step(
            action, self.dt, self.traffic_manager.vehicles, self.current_algo
        )

        # 2. 更新所有背景车辆 (现在它们会使用新的动态IDM逻辑)
        self.traffic_manager.update_background_traffic(self.dt)
        
        # 3. [重要] 调用环境自身的观测生成方法，以供下一次循环使用
        #    虽然rl_agent.step内部自己算了一次观测，但为了与最终架构对齐，我们在这里
        #    也计算一次，并用它作为最终返回的观测值。
        observation = self._get_observation()

        # ... (您原有的日志和info处理逻辑可以保留)
        if terminated or truncated:
            if hasattr(self.rl_agent, 'debug_log'):
                info['episode_log'] = self.rl_agent.debug_log
        info.update(self._get_info())
        
        return observation, reward, terminated, truncated, info

    def _get_observation(self):
        """
        [最终修正] 环境负责构建完整的观测，数据流完全清晰。
        """
        # --- 步骤1: 获取AV自身的局部状态 ---
        av_state_vec = self.rl_agent.get_base_observation(self.traffic_manager.vehicles)
        # 此时 av_state_vec 的长度应为 AV_OBS_DIM

        # --- 步骤2: 由环境来寻找并获取最相关的HV的状态 ---
        relevant_hv = None
        # (寻找 relevant_hv 的逻辑不变...)
        for v in self.traffic_manager.vehicles:
            if not getattr(v, 'is_rl_agent', False) and self.rl_agent._does_path_conflict(v):
                relevant_hv = v
                break

        if not relevant_hv:
            # 没有相关的HV，用0填充HV状态和预测部分
            hv_state_vec = np.zeros(HV_OBS_DIM)
            flat_yield_traj = np.zeros(PREDICTION_HORIZON * FEATURES_PER_STEP)
            flat_go_traj = np.zeros(PREDICTION_HORIZON * FEATURES_PER_STEP)
        else:
            # [核心修改] 环境自己获取HV的局部状态
            # 注意：这里需要一个能返回归一化相对状态的方法
            hv_state_vec = self._get_relative_state(relevant_hv)

            # --- 步骤3: 调用预测器 (逻辑不变) ---
            yield_traj = self.predictor.predict(
                current_av_state=self.rl_agent.state, 
                current_hv_state=relevant_hv.state,
                hv_intent_hypothesis='YIELD',
                hv_vehicle=relevant_hv
            )
            go_traj = self.predictor.predict(
                current_av_state=self.rl_agent.state, 
                current_hv_state=relevant_hv.state,
                hv_intent_hypothesis='GO',
                hv_vehicle=relevant_hv
            )
            flat_yield_traj = self._flatten_and_normalize_trajectory(yield_traj)
            flat_go_traj = self._flatten_and_normalize_trajectory(go_traj)

        # --- 步骤4: 最终拼接 ---
        final_obs = np.concatenate([
            av_state_vec,
            hv_state_vec,
            flat_yield_traj,
            flat_go_traj
        ]).astype(np.float32)
        
        # --- 步骤5: 维度检查 ---
        if final_obs.shape[0] != self.observation_space.shape[0]:
            # [新增] 打印详细的诊断信息
            print(f"--- 维度不匹配诊断 ---")
            print(f"期望总维度: {self.observation_space.shape[0]}")
            print(f"实际总维度: {final_obs.shape[0]}")
            print(f"  - AV状态维度: {len(av_state_vec)} (期望: {AV_OBS_DIM})")
            print(f"  - HV状态维度: {len(hv_state_vec)} (期望: {HV_OBS_DIM})")
            print(f"  - 预测轨迹维度: {len(flat_yield_traj) + len(flat_go_traj)} (期望: {2 * PREDICTION_HORIZON * FEATURES_PER_STEP})")
            raise ValueError("观测维度不匹配，请检查config.py和各观测函数的输出维度。")

        return final_obs

    def _get_relative_state(self, other_vehicle):
        """
        [新增] 一个辅助函数，用于计算并归一化与另一个车辆的相对状态。
        """
        ego_pos = np.array([self.rl_agent.state['x'], self.rl_agent.state['y']])
        ego_vel = np.array([self.rl_agent.state['vx'], self.rl_agent.state['vy']])
        
        other_pos = np.array([other_vehicle.state['x'], other_vehicle.state['y']])
        other_vel = np.array([other_vehicle.state['vx'], other_vehicle.state['vy']])
        
        relative_pos = other_pos - ego_pos
        relative_vel = other_vel - ego_vel
        
        # 返回一个归一化的向量，维度需与config.py中的HV_OBS_DIM (4) 匹配
        return np.array([
            relative_pos[0] / OBSERVATION_RADIUS,
            relative_pos[1] / OBSERVATION_RADIUS,
            relative_vel[0] / 15.0, # 使用最大速度近似值归一化
            relative_vel[1] / 15.0
        ])

    def _flatten_and_normalize_trajectory(self, trajectory):
        """[新] 辅助函数，用于处理预测出的轨迹。"""
        vec = np.array(trajectory).flatten()
        expected_len = PREDICTION_HORIZON * FEATURES_PER_STEP
        if len(vec) > expected_len:
            return vec[:expected_len] / OBSERVATION_RADIUS
        else:
            padded_vec = np.pad(vec, (0, expected_len - len(vec)), 'constant')
            return padded_vec / OBSERVATION_RADIUS

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
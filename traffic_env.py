# traffic_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame 

from road import Road
from traffic import TrafficManager
from config import * # 假设您的配置在这里
from vehicle import RLVehicle # 确保RLVehicle类存在于vehicle.py中

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

        # 2. 定义动作和观测空间
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        observation_dim = 14 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(observation_dim,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        """重置环境，返回初始观测值。"""
        super().reset(seed=seed)
        
        self.traffic_manager.vehicle_id_counter = 1
        self.traffic_manager.clear_all_vehicles()
        
        start_dir, end_dir = 'south', 'north'
        self.rl_agent = self.traffic_manager.spawn_rl_agent(start_dir, end_dir)
        
        if self.rl_agent is None:
            raise RuntimeError("环境重置失败：无法生成RL Agent。")
            
        self.traffic_manager.spawn_vehicle('west')

        observation = self.rl_agent.get_observation(self.traffic_manager.vehicles)
        info = self._get_info()
        
        return observation, info

    def step(self, action):
        """执行一个时间步。"""
        # 1. RL Agent执行动作
        observation, reward, terminated, truncated, info = self.rl_agent.step(
            action, self.dt, self.traffic_manager.vehicles
        )

        # 2. 更新所有背景车辆
        self.traffic_manager.update_background_traffic(self.dt)
        
        info.update(self._get_info())
        
        return observation, reward, terminated, truncated, info

    def _get_info(self):
        return {"agent_speed": self.rl_agent.get_current_speed() if self.rl_agent else 0}
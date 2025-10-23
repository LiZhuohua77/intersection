"""
@file: traffic.py
@description:
该文件定义了 `TrafficManager` 类，作为仿真世界中的“交通指挥官”。它全面负责
所有车辆（包括RL智能体和背景车辆）的生命周期管理、控制整体交通流的动态变化，
以及搭建用于训练和评估的特定交通场景。

核心职责:

1.  **车辆生命周期管理 (Vehicle Lifecycle Management):**
    - **生成 (Spawning):** 提供了 `spawn_vehicle` (用于背景NPC车辆) 和 `spawn_rl_agent`
      (用于RL智能体) 两个核心方法。车辆的生成受到严格的条件检查 (`can_spawn_vehicle`)，
      包括时间间隔、场景车辆总数限制以及出生点是否被占用，以防止不合理的生成。
    - **更新 (Updating):** `update_background_traffic` 方法在每个仿真步被调用，负责
      遍历并调用每个背景车辆自身的 `update` 方法，从而驱动它们的行为和决策逻辑。
    - **移除 (Removing):** 实时监控车辆状态，当车辆完成其预定路径后 (`vehicle.completed`)，
      会将其从活动车辆列表中安全移除，并可选择性地记录其行驶数据。

2.  **交通流控制 (Traffic Flow Control):**
    - **流量模式 (Patterns):** 内置了多种预设的交通流量模式（如 `rush_hour`, `normal`,
      `light`），可以通过 `set_traffic_pattern` 方法切换，动态调整背景车辆的生成
      频率，从而改变仿真环境的整体难度。
    - **转向行为 (Turning Behavior):** 通过 `turn_probabilities` 定义了从不同方向驶入的
      车辆选择直行、左转或右转的概率分布，使得交通流的行为更加真实和多样化。

3.  **场景搭建器 (Scenario Builder):**
    - 核心方法 `setup_scenario` 扮演着“场景工厂”的角色。它被 `TrafficEnv` 在每回合
      重置时调用，能够根据传入的场景名称 (`scenario_name`)，精确地布置RL智能体和
      一或多辆背景车辆的初始位置和行驶路线。
    - 这个功能是实现结构化、可复现实验的关键，能够创造出特定的、有针对性的挑战，
      例如“无保护左转”、“与直行车冲突”等，对于强化学习的训练和评估至关重要。

4.  **状态监控与调试:**
    - 提供了 `get_traffic_stats` 和 `draw_debug_info` 等辅助方法，用于在仿真过程中
      实时统计和显示交通信息（如车辆总数、平均速度等），为调试和分析提供便利。
"""

import random
import time
import numpy as np
from vehicle import Vehicle, RLVehicle
from config import *

class TrafficManager:
    """交通流量管理器"""
    
    def __init__(self, road, max_vehicles=6):
        self.road = road
        self.vehicles = []
        self.max_vehicles = max_vehicles
        self.vehicle_id_counter = 1
        self.completed_vehicles_data = []
        self.current_scenario = 'random_traffic'
        self.simulation_time = 0.0

        # 交通流量参数
        self.flow_config = {
            'base_intervals': {
                'north': 3.0, 'south': 3.0, 'east': 3.0, 'west': 3.0
            },
            'turn_probabilities': {
                'north': {'south': 0.4, 'east': 0.3, 'west': 0.3},
                'south': {'north': 0.4, 'east': 0.3, 'west': 0.3},
                'east': {'west': 0.4, 'north': 0.3, 'south': 0.3},
                'west': {'east': 0.4, 'north': 0.3, 'south': 0.3}
            }
        }
        
        self.next_spawn_times = {direction: 0 for direction in self.flow_config['base_intervals']}
        self._reschedule_all_spawns()

        self.current_training_step = 0
        self.total_training_steps = 1 # 避免除零错误
        
        self.min_v0_scaling = 0.2  # 训练开始时的最小缩放因子
        self.target_v0_scaling = 1.0 # 最终恢复到正常速度
        # 缩放因子从最低 ramping up 到目标值所占的总训练步数的比例
        self.scaling_ramp_up_ratio = 0.25
        self.current_v0_scaling = self.min_v0_scaling

    def update_curriculum_parameters(self, current_step: int, total_steps: int):
            """更新训练进度，并计算当前的v0缩放因子"""
            self.current_training_step = current_step
            self.total_training_steps = max(1, total_steps)
            
            # --- [核心修改：计算缩放因子] ---
            ramp_up_total_steps = self.total_training_steps * self.scaling_ramp_up_ratio
            progress = 0.0
            if ramp_up_total_steps > 0:
                progress = min(1.0, self.current_training_step / ramp_up_total_steps)
                
            self.current_v0_scaling = self.min_v0_scaling + progress * (self.target_v0_scaling - self.min_v0_scaling)
            # --- [核心修改结束] ---

            # [新增] 将计算出的缩放因子应用到所有现有的HV上
            # (这主要影响在 reset 时已经存在的车辆)
            for vehicle in self.vehicles:
                if not getattr(vehicle, 'is_rl_agent', False) and hasattr(vehicle, 'planner') and hasattr(vehicle.planner, 'update_speed_scaling'):
                    vehicle.planner.update_speed_scaling(self.current_v0_scaling)
    
    def set_base_interval(self, direction: str, interval: float):
        """
        [NEW] 新增的、用于直接控制交通密度的接口。
        """
        if direction in self.flow_config['base_intervals']:
            self.flow_config['base_intervals'][direction] = max(0.5, interval)
            print(f"Updated {direction} spawn interval to {interval}s.")
        else:
            print(f"Warning: Direction '{direction}' not found in flow_config.")

    def _sample_next_interval(self, direction: str) -> float:
        """
        [SIMPLIFIED] 不再需要 pattern_multiplier，直接使用 base_interval。
        """
        mean_interval = self.flow_config['base_intervals'][direction]
        u = 1.0 - random.random()
        return -mean_interval * np.log(u)

    def _reschedule_all_spawns(self):
        for direction in self.next_spawn_times.keys():
            self.next_spawn_times[direction] = self.simulation_time + self._sample_next_interval(direction)
    
    def get_random_destination(self, start_direction):
        probs = self.flow_config['turn_probabilities'][start_direction]
        valid_dirs = {k: v for k, v in probs.items() if k != start_direction}
        directions = list(valid_dirs.keys())
        weights = np.array(list(valid_dirs.values()))
        weights /= np.sum(weights)
        return np.random.choice(directions, p=weights)
    
    def can_spawn_vehicle(self, start_direction):
        """[优化] 检查生成点是否被占用。"""
        if len(self.vehicles) >= self.max_vehicles:
            return False
            
        spawn_point_data = self.road.get_route_points(start_direction, self.get_random_destination(start_direction))
        if not spawn_point_data:
            return False
        
        spawn_point = np.array(spawn_point_data["smoothed"][0][:2])
        safe_distance = 15  # 安全距离可以设小一些，只防止车辆重叠

        for vehicle in self.vehicles:
            # [FIXED] 通过 vehicle.state['x'] 和 vehicle.state['y'] 来获取位置
            other_pos = np.array([vehicle.state['x'], vehicle.state['y']])
            dist = np.linalg.norm(other_pos - spawn_point)
            
            if dist < safe_distance:
                return False
        return True
    
    def spawn_vehicle(self, start_direction, end_direction=None, personality=None, intent=None):
        """
        [最终修正版] 创建一个背景车辆(HV)。
        personality 和 intent 是可选的，如果没有提供，则会随机选择。
        """
        if len(self.vehicles) >= self.max_vehicles:
            return None
            
        if end_direction is None:
            end_direction = self.get_random_destination(start_direction)
        
        try:
            vehicle = Vehicle(self.road, start_direction, end_direction, self.vehicle_id_counter)        

            # [核心修改] 如果外部没有指定 personality 和 intent，才进行随机选择
            if personality is None:
                personality = random.choice(list(IDM_PARAMS.keys()))
            if intent is None:
                intent = random.choice(['GO', 'YIELD'])
            
            # 使用最终确定的 personality 和 intent 初始化规划器
            vehicle.initialize_planner(personality, intent)
            if hasattr(vehicle, 'planner') and hasattr(vehicle.planner, 'update_speed_scaling'):
                 vehicle.planner.update_speed_scaling(self.current_v0_scaling) # 应用缩放
            
            self.vehicles.append(vehicle)
            self.vehicle_id_counter += 1
            
            print(f"生成车辆 #{vehicle.vehicle_id}: 从 {start_direction} 到 {end_direction} (个性: {personality}, 意图: {intent})")
            return vehicle
        except ValueError as e:
            print(f"无法生成车辆: {e}")
            return None

    def spawn_rl_agent(self, start_direction, end_direction):
        """专门生成并返回一个RL Agent实例。"""
        # 在生成RL Agent前，最好清空环境，确保一个干净的开始
        self.clear_all_vehicles()
        try:
            # 使用 RLVehicle 类来创建 agent
            agent = RLVehicle(self.road, start_direction, end_direction, "RL_AGENT")
            self.vehicles.append(agent)
            # self.vehicle_id_counter += 1 # Agent ID是固定的，不需要增加计数器
            print(f"生成强化学习Agent: 从 {start_direction} 到 {end_direction}")
            return agent
        except ValueError as e:
            print(f"无法生成RL Agent: {e}")
            return None
        
    def update_background_traffic(self, dt: float):
        self.simulation_time += dt
        if self.current_scenario == "random_traffic":
            for direction, scheduled_time in self.next_spawn_times.items():
                if self.simulation_time >= scheduled_time:
                    # can_spawn_vehicle 现在只检查车辆数量和出生点是否被占用
                    if self.can_spawn_vehicle(direction):
                        self.spawn_vehicle(direction)
                    self.next_spawn_times[direction] = self.simulation_time + self._sample_next_interval(direction)
        
        for vehicle in self.vehicles:
            self._update_vehicle_context(vehicle)
            
        for vehicle in self.vehicles[:]:
            if not getattr(vehicle, 'is_rl_agent', False):
                vehicle.update(dt, self.vehicles)
            if vehicle.completed:
                self.vehicles.remove(vehicle)
            
    def clear_all_vehicles(self):
        self.vehicles.clear()
        self.completed_vehicles_data.clear()
        self.vehicle_id_counter = 1
        self.simulation_time = 0.0 # [新增] 重置仿真时钟
        self._reschedule_all_spawns() # [新增] 重置生成计划
        print("--- 已清空所有车辆和交通生成计划 ---")

    def _update_vehicle_context(self, vehicle):
        """
        [新] 辅助函数，用于更新车辆需要从环境中获取的上下文信息。
        这里我们计算并更新车辆到交叉口入口的路径距离。
        """
        # 使用您 road.py 中已有的接口
        entry_index = self.road.get_conflict_entry_index(vehicle.move_str)

        if entry_index != -1:
            # 如果路径经过交叉口
            entry_pos_longitudinal = vehicle.path_distances[entry_index]
            current_pos_longitudinal = vehicle.get_current_longitudinal_pos()
            
            # 计算并更新剩余距离
            distance = entry_pos_longitudinal - current_pos_longitudinal
            vehicle.dist_to_intersection_entry = max(0, distance)
        else:
            # 如果路径不经过交叉口
            vehicle.dist_to_intersection_entry = float('inf')    

    def get_traffic_stats(self):
        """获取交通统计信息"""
        stats = {
            'total_vehicles': len(self.vehicles),
            'by_direction': {'north': 0, 'south': 0, 'east': 0, 'west': 0},
            'average_speed': 0
        }
        
        if self.vehicles:
            total_speed = 0
            for vehicle in self.vehicles:
                # 统计起始方向
                if vehicle.start_direction in stats['by_direction']:
                    stats['by_direction'][vehicle.start_direction] += 1
                
                total_speed += vehicle.state.get('speed', 0)
            
            stats['average_speed'] = total_speed / len(self.vehicles)
        
        return stats
    
    def clear_all_vehicles(self):
        """清空所有车辆"""
        self.vehicles.clear()
        print("已清空所有车辆")
    
    def draw_debug_info(self, surface, font):
        """绘制调试信息"""
        stats = self.get_traffic_stats()
        
        debug_texts = [
            f"Total Vehicles: {stats['total_vehicles']}/{self.max_vehicles}",
            f"Average Speed: {stats['average_speed']:.1f} m/s",
            f"By Direction: N:{stats['by_direction']['north']} S:{stats['by_direction']['south']} E:{stats['by_direction']['east']} W:{stats['by_direction']['west']}"
        ]
        
        for i, text in enumerate(debug_texts):
            color = (200, 200, 200) if i == 0 else (255, 255, 255)
            text_surface = font.render(text, True, color)
            surface.blit(text_surface, (10, 40 + i * 20))
    
    def get_vehicle_speed_histories(self):
        """
        获取所有已完成车辆的速度历史数据。
        
        返回:
            dict: 按车辆类型分组的速度历史数据
        """
        result = {
            'background': [],
            'agent': []
        }
        
        for vehicle_data in self.completed_vehicles_data:
            vehicle_type = vehicle_data.get('type', 'background')
            result[vehicle_type].append({
                'id': vehicle_data['id'],
                'history': vehicle_data['history']
            })
        
        return result
            
    def setup_scenario(self, scenario_name="random"):
        """
        设置一个特定的实验场景。
        
        Args:
            scenario_name (str): 预定义的场景名称。
        """
        self.clear_all_vehicles()
        self.current_scenario = scenario_name
        print(f"--- Setting up scenario: {scenario_name} ---")

        agent = None

        # --------------------------------------------------------------------------
        # 阶段一：基础驾驶
        # --------------------------------------------------------------------------
        if scenario_name == "agent_only_simple":
            # 1a: 随机直行或右转
            routes = [
                ('south', 'north'), ('north', 'south'), ('east', 'west'), ('west', 'east'),
                ('south', 'east'), ('north', 'west'), ('east', 'north'), ('west', 'south'),
                ('south', 'west'), ('west', 'north'), ('north', 'east'), ('east', 'south')
            ]
            start_dir, end_dir = random.choice(routes)
            agent = self.spawn_rl_agent(start_dir, end_dir)

        # --------------------------------------------------------------------------
        # 阶段二：合作交互
        # --------------------------------------------------------------------------
        elif scenario_name == "cooperative_yield":
            # AV左转 vs HV直行，AV路权劣势。方向随机化。
            scenarios = [
                {'agent': ('south', 'west'), 'bg': ('north', 'south')},
                {'agent': ('west', 'north'), 'bg': ('east', 'west')},
                {'agent': ('north', 'east'), 'bg': ('south', 'north')},
                {'agent': ('east', 'south'), 'bg': ('west', 'east')},
            ]
            chosen = random.choice(scenarios)
            agent = self.spawn_rl_agent(chosen['agent'][0], chosen['agent'][1])
            bg_vehicle = self.spawn_vehicle(chosen['bg'][0], chosen['bg'][1], personality='CONSERVATIVE')

        # --------------------------------------------------------------------------
        # 阶段三：竞争博弈
        # --------------------------------------------------------------------------
        elif scenario_name == "head_on_conflict":
            # AV左转 vs HV迎头左转，路权模糊。方向随机化。
            scenarios = [
                {'agent': ('south', 'west'), 'bg': ('north', 'east')},
                {'agent': ('west', 'north'), 'bg': ('east', 'south')},
                {'agent': ('north', 'east'), 'bg': ('south', 'west')},
                {'agent': ('east', 'south'), 'bg': ('west', 'north')},
            ]
            chosen = random.choice(scenarios)
            agent = self.spawn_rl_agent(chosen['agent'][0], chosen['agent'][1])
            bg_vehicle = self.spawn_vehicle(chosen['bg'][0], chosen['bg'][1], personality='NORMAL')

        elif scenario_name == "crossing_conflict":
            # AV直行 vs HV十字交叉直行。方向随机化。
            scenarios = [
                {'agent': ('south', 'north'), 'bg': ('west', 'east')},
                {'agent': ('west', 'east'), 'bg': ('north', 'south')},
                {'agent': ('north', 'south'), 'bg': ('east', 'west')},
                {'agent': ('east', 'west'), 'bg': ('south', 'north')},
            ]
            chosen = random.choice(scenarios)
            agent = self.spawn_rl_agent(chosen['agent'][0], chosen['agent'][1])
            bg_vehicle = self.spawn_vehicle(chosen['bg'][0], chosen['bg'][1], personality='AGGRESSIVE')

        # --------------------------------------------------------------------------
        # 阶段四：泛化训练
        # --------------------------------------------------------------------------
        elif scenario_name == "random_traffic":
            # AV随机路线，并预先生成1-3辆随机HV，后续动态生成更多
            for _ in range(random.randint(1, 1)):
                start = random.choice(['north', 'south', 'east', 'west'])
                self.spawn_vehicle(start)

            # 为AV寻找一个安全的出生点
            spawn_success = False
            for _ in range(10): # 最多尝试10次
                start_dir = random.choice(['north', 'south', 'east', 'west'])
                end_dir = self.get_random_destination(start_dir)
                if self.can_spawn_vehicle(start_dir): # can_spawn_vehicle 检查出生点是否空闲
                    #agent = self.spawn_rl_agent(start_dir, end_dir)
                    spawn_success = True
                    break
            if not spawn_success:
                # 如果10次都失败（交通太拥挤），则清空后只生成Agent
                self.clear_all_vehicles()
                start_dir = random.choice(['north', 'south', 'east', 'west'])
                end_dir = self.get_random_destination(start_dir)
                #agent = self.spawn_rl_agent(start_dir, end_dir)
        
        else:
            raise ValueError(f"未知的场景名称: {scenario_name}")
        
        return agent
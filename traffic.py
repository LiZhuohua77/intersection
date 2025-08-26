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

class TrafficManager:
    """交通流量管理器"""
    
    def __init__(self, road, max_vehicles=2):
        self.road = road
        self.vehicles = []
        self.max_vehicles = max_vehicles
        self.vehicle_id_counter = 1
        self.completed_vehicles_data = []
        self.current_pattern = 'normal'
        self.current_scenario = 'random'
        

        # 交通流量参数
        self.spawn_intervals = {
            'north': 3.0,  # 从北边进入的车辆间隔时间（秒）
            'south': 3.0,  # 从南边进入的车辆间隔时间（秒）
            'east': 3.0,   # 从东边进入的车辆间隔时间（秒）
            'west': 3.0    # 从西边进入的车辆间隔时间（秒）
        }
        
        # 上次生成车辆的时间
        self.last_spawn_time = {
            'north': 0,
            'south': 0,
            'east': 0,
            'west': 0
        }
        
        # 转向概率分布
        self.turn_probabilities = {
            'north': {'north': 0.0, 'south': 0.4, 'east': 0.3, 'west': 0.3},
            'south': {'north': 0.4, 'south': 0.0, 'east': 0.3, 'west': 0.3},
            'east': {'north': 0.3, 'south': 0.3, 'east': 0.0, 'west': 0.4},
            'west': {'north': 0.3, 'south': 0.3, 'east': 0.4, 'west': 0.0}
        }
        
        # 交通流量模式
        self.traffic_patterns = {
            'rush_hour': {
            'multiplier': 0.5,  # Interval time multiplier (smaller means denser traffic)
            'description': 'Rush Hour - Dense Traffic'
            },
            'normal': {
            'multiplier': 1.0,
            'description': 'Normal - Medium Traffic'
            },
            'light': {
            'multiplier': 2.0,
            'description': 'Light - Sparse Traffic'
            },
            'night': {
            'multiplier': 3.0,
            'description': 'Night - Very Light Traffic'
            }
        }
        
        self.current_pattern = 'normal'
    
    def set_traffic_pattern(self, pattern_name):
        """设置交通流量模式"""
        if pattern_name in self.traffic_patterns:
            self.current_pattern = pattern_name
            print(f"交通模式切换为: {self.traffic_patterns[pattern_name]['description']}")
    
    def adjust_flow_rate(self, direction, interval):
        """调整特定方向的车流间隔"""
        if direction in self.spawn_intervals:
            self.spawn_intervals[direction] = max(0.5, interval)
    
    def get_random_destination(self, start_direction):
        """根据概率选择目标方向"""
        probabilities = self.turn_probabilities[start_direction]
        directions = list(probabilities.keys())
        weights = list(probabilities.values())
        
        return np.random.choice(directions, p=weights)
    
    def can_spawn_vehicle(self, start_direction):
        """检查是否可以在指定方向生成车辆"""
        current_time = time.time()
        
        # 检查时间间隔
        pattern_multiplier = self.traffic_patterns[self.current_pattern]['multiplier']
        required_interval = self.spawn_intervals[start_direction] * pattern_multiplier
        
        if current_time - self.last_spawn_time[start_direction] < required_interval:
            return False
        
        # 检查车辆数量限制
        if len(self.vehicles) >= self.max_vehicles:
            return False
        
        # 检查起始位置是否有车辆阻塞
        spawn_routes = self.road.get_route_points(start_direction, 
                                                self.get_random_destination(start_direction))
        if not spawn_routes:
            return False
        
        spawn_point = spawn_routes["smoothed"][0]
        safe_distance = 80  # 安全距离
        
        for vehicle in self.vehicles:
            if not vehicle.completed:
                dist = ((vehicle.state['x'] - spawn_point[0])**2 + (vehicle.state['y'] - spawn_point[1])**2)**0.5
                if dist < safe_distance:
                    return False
        
        return True
    
    def spawn_vehicle(self, start_direction):
        """在指定方向生成车辆"""
        if not self.can_spawn_vehicle(start_direction):
            return None
        
        end_direction = self.get_random_destination(start_direction)
        
        try:
            vehicle = Vehicle(self.road, start_direction, end_direction, self.vehicle_id_counter)
            
            self.vehicles.append(vehicle)
            self.vehicle_id_counter += 1
            self.last_spawn_time[start_direction] = time.time()
            
            print(f"生成车辆 #{vehicle.vehicle_id}: 从 {start_direction} 到 {end_direction}")
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
        
    def update_background_traffic(self, dt):
        """更新背景交通"""
        if self.current_scenario == "random":
            directions = ['north', 'south', 'east', 'west']
            if random.random() < 0.1: 
                self.spawn_vehicle(random.choice(directions))
        
        # 更新所有车辆
        for vehicle in self.vehicles[:]:
            if not getattr(vehicle, 'is_rl_agent', False):
                vehicle.update(dt, self.vehicles, self)
            
            if vehicle.completed:
                print(f"车辆 #{vehicle.vehicle_id} 已完成路径")
                self.completed_vehicles_data.append({
                    'id': vehicle.vehicle_id,
                    'type': 'background',
                    'history': vehicle.get_speed_history()
                })
                self.vehicles.remove(vehicle)
    
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
            f"Traffic Mode: {self.traffic_patterns[self.current_pattern]['description']}",
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
    
    def generate_scenario_tree(self, current_state, ego_plan=None, depth=3, branching=2):
        """
        为当前状态生成一个场景树
        
        参数:
        - current_state: 当前环境状态
        - ego_plan: 自车计划采取的动作序列
        - depth: 场景树的深度（预测多少步）
        - branching: 每个节点的分支数量（为每个车辆考虑多少种可能动作）
        
        返回:
        - scenario_tree: 场景树结构，包含可能的未来状态及其概率
        """
        # 获取当前场景中的所有车辆
        vehicles = self.vehicles.copy()
        
        # 创建场景树根节点
        scenario_tree = {
            'root': {
                'state': current_state,  # 当前环境状态
                'probability': 1.0,  # 根节点概率为1
                'children': []
            }
        }
        
        # 递归构建场景树
        self._build_scenario_tree_node(scenario_tree['root'], vehicles, depth, branching, ego_plan)
        
        return scenario_tree
    
    def _build_scenario_tree_node(self, node, vehicles, remaining_depth, branching, ego_plan=None):
        """递归构建场景树节点"""
        if remaining_depth <= 0 or not vehicles:
            return
        
        # 为每个车辆生成可能的动作和下一状态
        for vehicle in vehicles:
            if getattr(vehicle, 'is_rl_agent', False) and ego_plan:
                # 如果是RL Agent且有动作计划，使用计划中的动作
                # 注意：这里简化处理，实际应根据ego_plan的结构适配
                action = ego_plan[0] if ego_plan else None
                next_states = [(vehicle.predict_next_state(action), 1.0)]
            else:
                # 对于背景车辆，生成多种可能的动作和状态
                next_states = []
                
                # 使用车辆模型生成可能的下一状态
                # 这里可以基于历史数据或启发式规则来生成
                # 例如：正常行驶、减速、急减速等不同行为模式
                
                # 简单示例：生成branching个可能的速度变化
                current_speed = vehicle.state.get('speed', 0)
                for i in range(branching):
                    # 随机生成速度变化（-20%到+20%）
                    speed_change = (random.random() * 0.4 - 0.2) * current_speed
                    new_speed = max(0, current_speed + speed_change)
                    
                    # 创建假设的下一状态（简化版）
                    next_state = vehicle.state.copy()
                    next_state['speed'] = new_speed
                    
                    # 分配概率（简单起见，使用均匀分布）
                    probability = 1.0 / branching
                    
                    next_states.append((next_state, probability))
            
            # 为每个可能的下一状态创建子节点
            for next_state, probability in next_states:
                child_node = {
                    'state': next_state,
                    'probability': node['probability'] * probability,  # 累积概率
                    'children': []
                }
                
                node['children'].append(child_node)
                
                # 递归构建下一层
                if remaining_depth > 1:
                    # 创建下一步的车辆列表
                    next_vehicles = [v for v in vehicles if v != vehicle]
                    
                    # 如果有ego_plan，更新它
                    next_ego_plan = ego_plan[1:] if ego_plan and len(ego_plan) > 1 else None
                    
                    # 递归构建子树
                    self._build_scenario_tree_node(child_node, next_vehicles, remaining_depth-1, branching, next_ego_plan)
            
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

        if scenario_name == "protected_left_turn":
            # 场景：Agent从南向西左转，需要让行北向南直行的背景车
            agent = self.spawn_rl_agent('south', 'west')
            # 生成一辆有路权优势的背景车
            bg_vehicle = Vehicle(self.road, 'north', 'south', self.vehicle_id_counter)
            self.vehicles.append(bg_vehicle)
            self.vehicle_id_counter += 1
            print(f"背景车辆 #{bg_vehicle.vehicle_id}: 从北向南直行")

        elif scenario_name == "unprotected_left_turn":
            # 场景：Agent从南向西左转，北向东左转的背景车需要让行Agent
            agent = self.spawn_rl_agent('south', 'west')
            # 生成一辆路权劣势的背景车
            bg_vehicle = Vehicle(self.road, 'north', 'east', self.vehicle_id_counter)
            self.vehicles.append(bg_vehicle)
            self.vehicle_id_counter += 1
            print(f"背景车辆 #{bg_vehicle.vehicle_id}: 从北向东左转")

        elif scenario_name == "head_on_conflict":
             # 场景：Agent从南向西左转，对向车辆从北向东也左转，测试博弈
            agent = self.spawn_rl_agent('south', 'west')
            bg_vehicle = Vehicle(self.road, 'north', 'east', self.vehicle_id_counter)
            self.vehicles.append(bg_vehicle)
            self.vehicle_id_counter += 1
            print(f"背景车辆 #{bg_vehicle.vehicle_id}: 从北向东左转")

        elif scenario_name == "agent_only":
            # 场景：只有Agent，用于测试基本路径跟踪性能
            agent = self.spawn_rl_agent('south', 'north')
            
        elif scenario_name == "east_west_traffic":
            # 场景：只有东西方向的背景车辆，无RL智能体
            # 生成从东向西行驶的背景车辆
            bg_vehicle1 = Vehicle(self.road, 'east', 'west', self.vehicle_id_counter)
            self.vehicles.append(bg_vehicle1)
            self.vehicle_id_counter += 1
            print(f"背景车辆 #{bg_vehicle1.vehicle_id}: 从东向西直行")
            
            # 生成从西向东行驶的背景车辆
            bg_vehicle2 = Vehicle(self.road, 'west', 'east', self.vehicle_id_counter)
            self.vehicles.append(bg_vehicle2)
            self.vehicle_id_counter += 1
            print(f"背景车辆 #{bg_vehicle2.vehicle_id}: 从西向东直行")
            
            # 没有RL智能体的场景需要返回None
            return None
            
        elif scenario_name == "north_south_traffic":
            # 场景：只有南北方向的背景车辆，无RL智能体
            # 生成从北向南行驶的背景车辆
            bg_vehicle1 = Vehicle(self.road, 'north', 'south', self.vehicle_id_counter)
            self.vehicles.append(bg_vehicle1)
            self.vehicle_id_counter += 1
            print(f"背景车辆 #{bg_vehicle1.vehicle_id}: 从北向南直行")
            
            # 生成从南向北行驶的背景车辆
            """ bg_vehicle2 = Vehicle(self.road, 'west', 'east', self.vehicle_id_counter)
            self.vehicles.append(bg_vehicle2)
            self.vehicle_id_counter += 1
            print(f"背景车辆 #{bg_vehicle2.vehicle_id}: 从西向东直行") """

            bg_vehicle3 = Vehicle(self.road, 'south', 'west', self.vehicle_id_counter)
            self.vehicles.append(bg_vehicle3)
            self.vehicle_id_counter += 1
            print(f"背景车辆 #{bg_vehicle3.vehicle_id}: 从南向西直行")

            # 没有RL智能体的场景需要返回None
            return None

        else: # 默认是随机交通流
            # 如果是随机，我们只生成Agent，背景车会在后续的step中随机生成
            agent = self.spawn_rl_agent('south', 'north')
            print("默认场景：仅生成Agent，背景交通将随机出现。")
        
        return agent
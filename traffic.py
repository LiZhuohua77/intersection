# traffic.py

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
            
    def create_test_scenario(self):
        """
        创建特定的测试场景：两辆车同时生成，一辆从南到北，另一辆从东到西
        
        Returns:
            tuple: (南北车辆, 东西车辆) - 用于后续观察和调试
        """
        # 首先清空当前所有车辆，确保测试环境干净
        self.clear_all_vehicles()
        
        # 创建南到北的车辆
        south_to_north_vehicle = Vehicle(self.road, 'south', 'west', self.vehicle_id_counter)
        self.vehicles.append(south_to_north_vehicle)
        self.vehicle_id_counter += 1
        print(f"测试场景: 生成车辆 #{south_to_north_vehicle.vehicle_id} - 从南向西行驶")

        # 创建东到西的车辆
        east_to_west_vehicle = Vehicle(self.road, 'west', 'east', self.vehicle_id_counter)
        self.vehicles.append(east_to_west_vehicle)
        self.vehicle_id_counter += 1
        print(f"测试场景: 生成车辆 #{east_to_west_vehicle.vehicle_id} - 从东向西行驶")
        
        # 为了便于后续调试，返回这两辆车的引用
        return south_to_north_vehicle, east_to_west_vehicle
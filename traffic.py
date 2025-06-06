import random
import time
import numpy as np
from vehicle import Vehicle

class TrafficManager:
    """交通流量管理器"""
    
    def __init__(self, road, max_vehicles=20):
        self.road = road
        self.vehicles = []
        self.max_vehicles = max_vehicles
        self.vehicle_id_counter = 1
        
        # 交通流量参数
        self.spawn_intervals = {
            'north': 3.0,  # 从北边进入的车辆间隔时间（秒）
            'south': 4.0,  # 从南边进入的车辆间隔时间（秒）
            'east': 3.5,   # 从东边进入的车辆间隔时间（秒）
            'west': 2.5    # 从西边进入的车辆间隔时间（秒）
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
        
        # 车辆类型分布
        self.vehicle_types = {
            'car': {'probability': 1, 'speed': 10.0, 'color': (200, 50, 50)},
            'truck': {'probability': 0, 'speed': 12.0, 'color': (100, 100, 200)},
            'bus': {'probability': 0, 'speed': 10.0, 'color': (50, 200, 50)}
        }
    
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
    
    def get_random_vehicle_type(self):
        """随机选择车辆类型"""
        rand = random.random()
        cumulative = 0
        
        for vehicle_type, config in self.vehicle_types.items():
            cumulative += config['probability']
            if rand <= cumulative:
                return vehicle_type, config
        
        return 'car', self.vehicle_types['car']
    
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
        
        spawn_point = spawn_routes[0]
        safe_distance = 80  # 安全距离
        
        for vehicle in self.vehicles:
            if not vehicle.completed:
                dist = ((vehicle.x - spawn_point[0])**2 + (vehicle.y - spawn_point[1])**2)**0.5
                if dist < safe_distance:
                    return False
        
        return True
    
    def spawn_vehicle(self, start_direction):
        """在指定方向生成车辆"""
        if not self.can_spawn_vehicle(start_direction):
            return None
        
        # 选择目标方向
        end_direction = self.get_random_destination(start_direction)
        
        # 选择车辆类型
        vehicle_type, type_config = self.get_random_vehicle_type()
        
        # 创建车辆
        vehicle = Vehicle(self.road, start_direction, end_direction, self.vehicle_id_counter)
        vehicle.speed = type_config['speed'] + random.uniform(-2, 2)  # 添加速度随机性
        vehicle.color = type_config['color']
        vehicle.vehicle_type = vehicle_type
        
        self.vehicles.append(vehicle)
        self.vehicle_id_counter += 1
        self.last_spawn_time[start_direction] = time.time()
        
        print(f"生成车辆 #{vehicle.vehicle_id}: {vehicle_type} 从 {start_direction} 到 {end_direction}")
        return vehicle
    
    def update(self, dt):
        """更新交通管理器"""
        # 尝试在各个方向生成车辆
        directions = ['north', 'south', 'east', 'west']
        
        for direction in directions:
            if random.random() < 0.3:  # 30%概率尝试生成
                self.spawn_vehicle(direction)
        
        # 更新所有车辆（传递车辆列表用于IDM计算）
        for vehicle in self.vehicles[:]:  # 使用切片避免修改列表时的问题
            vehicle.update(dt, all_vehicles=self.vehicles)
            
            # 移除已完成的车辆
            if vehicle.completed:
                print(f"车辆 #{vehicle.vehicle_id} 已完成路径")
                self.vehicles.remove(vehicle)
    
    def get_traffic_stats(self):
        """获取交通统计信息"""
        stats = {
            'total_vehicles': len(self.vehicles),
            'by_direction': {'north': 0, 'south': 0, 'east': 0, 'west': 0},
            'by_type': {'car': 0, 'truck': 0, 'bus': 0},
            'average_speed': 0
        }
        
        if self.vehicles:
            total_speed = 0
            for vehicle in self.vehicles:
                # 统计起始方向
                if vehicle.start_direction in stats['by_direction']:
                    stats['by_direction'][vehicle.start_direction] += 1
                
                # 统计车辆类型
                if hasattr(vehicle, 'vehicle_type') and vehicle.vehicle_type in stats['by_type']:
                    stats['by_type'][vehicle.vehicle_type] += 1
                
                total_speed += vehicle.speed
            
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
            f"By Direction: N:{stats['by_direction']['north']} S:{stats['by_direction']['south']} E:{stats['by_direction']['east']} W:{stats['by_direction']['west']}",
            f"Vehicle Types: Car:{stats['by_type']['car']} Truck:{stats['by_type']['truck']} Bus:{stats['by_type']['bus']}"
        ]
        
        for i, text in enumerate(debug_texts):
            color = (200, 200, 200) if i == 0 else (255, 255, 255)
            text_surface = font.render(text, True, color)
            surface.blit(text_surface, (10, 40 + i * 20))
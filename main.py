from road import Road
from traffic import TrafficManager
from game_engine import GameEngine

def main():
    """主函数"""
    # 创建游戏引擎
    game_engine = GameEngine(width=800, height=800)
    
    # 创建道路对象和交通管理器
    road = Road()
    traffic_manager = TrafficManager(road, max_vehicles=2)
    
    # 主游戏循环
    try:
        while game_engine.is_running():
            game_engine.run_frame(road, traffic_manager)
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    finally:
        game_engine.quit()
        print("仿真结束")

if __name__ == "__main__":
    main()
"""
@file: PotentialField.py
@description: 该文件实现了一个势场可视化工具,用于交通路口场景中生成和可视化势场(potential field)。
势场在自动驾驶领域用于表示不同位置的"吸引力"或"排斥力",辅助车辆路径规划和行为决策。

主要功能:
1. 生成势场数据网格 - generate_potential_field()
2. 使用matplotlib可视化势场热力图 - visualize_as_matplotlib()
3. 绘制道路边界 - draw_main_road_boundaries()
4. 生成Pygame兼容的热力图 - create_pygame_heatmap()
5. 可视化势场梯度方向 - visualize_potential_gradient()
"""

import pygame
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors
from road import Road

class PotentialFieldVisualizer:
    def __init__(self, road, resolution=2):
        """
        初始化势场可视化器
        
        Args:
            road (Road): Road对象实例,包含道路几何和势场计算函数
            resolution (int): 网格分辨率(像素),值越小越精细但计算量越大
        """
        self.road = road
        self.resolution = resolution
        self.width = road.width
        self.height = road.height

    def generate_potential_field(self, target_route_str=None):
        """
        为整个场景生成势能场数据网格
        
        Args:
            target_route_str (str, optional): 目标路径字符串,如'N_S'(北到南),
                                            如果为None则生成通用势场
            
        Returns:
            tuple: (X网格, Y网格, 势能值矩阵Z),其中X和Y为二维网格坐标,Z为对应的势能值
        """
        # 创建网格
        x = np.arange(0, self.width, self.resolution)
        y = np.arange(0, self.height, self.resolution)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X, dtype=float)
        
        # 计算每个点的势能值
        for i in range(len(y)):
            for j in range(len(x)):
                Z[i, j] = self.road.get_potential_at_point(X[i, j], Y[i, j], target_route_str)
                
        return X, Y, Z
    
    def visualize_as_matplotlib(self, target_route_str=None, show_centerlines=True, 
                            show_boundaries=True, save_path=None, show=True):
        """
        使用matplotlib绘制势场热力图并显示/保存结果
        
        Args:
            target_route_str (str, optional): 目标路径字符串,如'S_N'
            show_centerlines (bool): 是否显示道路中心线
            show_boundaries (bool): 是否显示道路边界
            save_path (str, optional): 保存图像的文件路径,如果为None则不保存
            show (bool): 是否立即显示图像
        
        Returns:
            None: 函数将显示和/或保存图像,但不返回值
        """
        X, Y, Z = self.generate_potential_field(target_route_str)
        
        fig, ax = plt.figure(figsize=(10, 10)), plt.axes()
        
        # 使用自定义归一化使颜色映射更清晰
        norm = colors.Normalize(vmin=0, vmax=np.max(Z) * 0.8)
        
        # 绘制势场热力图
        c = ax.pcolormesh(X, Y, Z, cmap='viridis', norm=norm, alpha=0.7)
        fig.colorbar(c, ax=ax, label='Potential Value')
        
        # 绘制等高线
        contour = ax.contour(X, Y, Z, colors='white', alpha=0.5)
        ax.clabel(contour, inline=True, fontsize=8)
        
        # 绘制道路中心线和边界
        if show_centerlines or show_boundaries:
            # 确定要绘制的路径
            routes_to_draw = [route_name for route_name in self.road.routes.keys()]
            if target_route_str:
                if target_route_str in self.road.routes:
                    routes_to_draw = [target_route_str]
                else:
                    print(f"警告: 路径 {target_route_str} 不存在")
            
            # 绘制选定路径的中心线和边界
            for route_name in routes_to_draw:
                route_data = self.road.routes[route_name]
                
                # 绘制中心线
                if show_centerlines:
                    centerline = np.array([p[:2] for p in route_data["smoothed"]])
                    ax.plot(centerline[:, 0], centerline[:, 1], 'r-', linewidth=1, label="Centerline" if route_name == routes_to_draw[0] else "")
                
                # 绘制道路边界
                if show_boundaries and "boundaries" in route_data:
                    left_boundary = np.array(route_data["boundaries"]["left"])
                    right_boundary = np.array(route_data["boundaries"]["right"])
                    
                    ax.plot(left_boundary[:, 0], left_boundary[:, 1], 'w--', linewidth=1, 
                            label="Road Boundary" if route_name == routes_to_draw[0] else "")
                    ax.plot(right_boundary[:, 0], right_boundary[:, 1], 'w--', linewidth=1)
        
        # 绘制主道路边界
        self.draw_main_road_boundaries(ax)
        
        # 标题和标签
        title = f"Potential Field"
        if target_route_str:
            title += f" for Route {target_route_str}"
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        # 添加图例（如果需要）
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend()
        
        # 设置坐标轴等于比例
        ax.set_aspect('equal')
        
        # 保存图像
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        # 显示图像
        if show:
            plt.show()
        else:
            plt.close()

    def draw_main_road_boundaries(self, ax):
        """
        在matplotlib图表上绘制主道路边界线
        
        Args:
            ax (matplotlib.axes.Axes): matplotlib轴对象,用于在其上绘制边界线
        
        Notes:
            绘制十字路口的四个象限边界,包括水平和垂直道路的边界线
        """
        road = self.road
        lane_width = road.lane_width
        center_x, center_y = road.center_x, road.center_y
        
        # 绘制十字路口的四个象限边界
        
        # 水平道路边界
        # 上边界
        ax.plot([0, center_x - lane_width, center_x - lane_width], 
                [center_y - lane_width, center_y - lane_width, 0], 
                'white', linewidth=1)
        ax.plot([center_x + lane_width, center_x + lane_width, road.width], 
                [0, center_y - lane_width, center_y - lane_width], 
                'white', linewidth=1)

        # 下边界
        ax.plot([0, center_x - lane_width, center_x - lane_width], 
                [center_y + lane_width, center_y + lane_width, road.height], 
                'white', linewidth=1)
        ax.plot([center_x + lane_width, center_x + lane_width, road.width], 
                [road.height, center_y + lane_width, center_y + lane_width], 
                'white', linewidth=1)
        
        # 垂直道路边界
        # 左边界
        ax.plot([center_x - lane_width, center_x - lane_width], 
                [0, center_y - lane_width], 
                'white', linewidth=1)
        ax.plot([center_x - lane_width, center_x - lane_width], 
                [center_y + lane_width, road.height], 
                'white', linewidth=1)
        
        # 右边界
        ax.plot([center_x + lane_width, center_x + lane_width], 
                [0, center_y - lane_width], 
                'white', linewidth=1)
        ax.plot([center_x + lane_width, center_x + lane_width], 
                [center_y + lane_width, road.height], 
                'white', linewidth=1)

    def create_pygame_heatmap(self, target_route_str=None, alpha=150):
        """
        创建可叠加在Pygame界面上的热力图表面
        
        Args:
            target_route_str (str, optional): 目标路径字符串,用于生成特定路径的势场
            alpha (int): 透明度值(0-255),控制热力图的透明度
            
        Returns:
            pygame.Surface: 带有热力图的透明表面,可直接绘制到Pygame窗口
        
        Notes:
            热力图使用色彩梯度从深紫色(低势能)到黄色(高势能)表示势场强度
        """
        X, Y, Z = self.generate_potential_field(target_route_str)
        
        # 创建透明表面
        surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        
        # 将势能值归一化到0-1范围
        z_max = np.max(Z)
        z_min = np.min(Z)
        Z_norm = (Z - z_min) / (z_max - z_min) if z_max > z_min else Z
        
        # 绘制热力图
        for i in range(len(Y)):
            for j in range(len(X)):
                x, y = int(X[i, j]), int(Y[i, j])
                
                # 使用viridis色图的近似：从深蓝到黄色
                value = Z_norm[i, j]
                if value < 0.25:
                    r, g, b = 68, 1, 84  # 深紫色
                elif value < 0.5:
                    r, g, b = 59, 82, 139  # 蓝色
                elif value < 0.75:
                    r, g, b = 33, 144, 140  # 绿色
                else:
                    r, g, b = 253, 231, 37  # 黄色
                    
                pygame.draw.rect(
                    surface, 
                    (r, g, b, int(alpha * min(1.0, Z_norm[i, j] * 1.5))), 
                    (x, y, self.resolution, self.resolution)
                )
                
        return surface

    def visualize_potential_gradient(self, target_route_str=None, step=15, scale=50.0, 
                                    show_boundaries=True, arrow_color='yellow'):
        """
        可视化势场的梯度方向,显示车辆会被引导向哪个方向
        
        Args:
            target_route_str (str, optional): 目标路径字符串
            step (int): 箭头绘制间隔,值越大箭头越稀疏
            scale (float): 箭头大小比例因子,值越小箭头越长
            show_boundaries (bool): 是否显示道路边界
            arrow_color (str): 箭头颜色
        
        Notes:
            梯度箭头指向势场下降最快的方向,表示车辆在该位置的首选移动方向
            该方法展示了车辆在任意位置会受到的"力"的方向
        """
        X, Y, Z = self.generate_potential_field(target_route_str)
        
        # 计算梯度
        dy, dx = np.gradient(Z)
        
        # 可视化
        fig, ax = plt.figure(figsize=(10, 10)), plt.axes()
        
        # 绘制背景热力图
        c = ax.pcolormesh(X, Y, Z, cmap='viridis', alpha=0.7)
        fig.colorbar(c, ax=ax, label='Potential Value')
        
        # 梯度是势能的负梯度，所以车辆会朝着负梯度方向移动
        # 计算梯度向量的大小用于颜色映射
        magnitude = np.sqrt(dx**2 + dy**2)
        # 使用更明显的箭头
        quiver = ax.quiver(X[::step, ::step], Y[::step, ::step], 
                        -dx[::step, ::step], -dy[::step, ::step],
                        scale=scale, 
                        color=arrow_color,
                        width=0.005,  # 增加箭头宽度
                        headwidth=4,  # 增加箭头头部宽度
                        headlength=5,  # 增加箭头头部长度
                        minshaft=2)   # 确保短箭头也有可见的杆
                        
        # 添加图例箭头
        ax.quiverkey(quiver, 0.9, 0.95, 2, "Gradient Direction\n(Vehicle Movement Trend)", 
                labelpos='E', coordinates='figure', color=arrow_color)
        
        # 绘制道路中心线和边界
        routes_to_draw = [target_route_str] if target_route_str else self.road.routes.keys()
        for route_name in routes_to_draw:
            if route_name in self.road.routes:
                route_data = self.road.routes[route_name]
                
                # 绘制中心线
                centerline = np.array([p[:2] for p in route_data["smoothed"]])
                ax.plot(centerline[:, 0], centerline[:, 1], 'r-', linewidth=1, 
                    label="Centerline" if route_name == list(routes_to_draw)[0] else "")
                
                # 绘制道路边界
                if show_boundaries and "boundaries" in route_data:
                    left_boundary = np.array(route_data["boundaries"]["left"])
                    right_boundary = np.array(route_data["boundaries"]["right"])
                    
                    ax.plot(left_boundary[:, 0], left_boundary[:, 1], 'w--', linewidth=1, 
                        label="Road Boundary" if route_name == list(routes_to_draw)[0] else "")
                    ax.plot(right_boundary[:, 0], right_boundary[:, 1], 'w--', linewidth=1)
        
        # 绘制主道路边界
        if show_boundaries:
            self.draw_main_road_boundaries(ax)
        
        # 标题和标签
        title = f"Potential Field Gradient"
        if target_route_str:
            title += f" for Route {target_route_str}"
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        # 添加图例
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend()
        
        # 设置坐标轴等于比例
        ax.set_aspect('equal')
        plt.show()

if __name__ == "__main__":
    # 示例用法
    road = Road(width=400, height=400, lane_width=4)

    # 初始化可视化器
    visualizer = PotentialFieldVisualizer(road, resolution=1)

    # 生成并保存单条路径的势场热力图
    visualizer.visualize_as_matplotlib('S_N', show_centerlines=True, show_boundaries=True, save_path='sn_potential_field.png')

    # 可视化势场梯度
    visualizer.visualize_potential_gradient('S_E')
# path_smoother.py
import numpy as np
import math
from scipy.interpolate import interp1d # 使用scipy进行更精确的插值

def smooth_path(path, alpha=0.9, beta=0.5, iterations=100):
    """
    使用基于梯度下降的优化方法平滑路径。
    
    Args:
        path (list): 输入的路径点列表 [[x1, y1], [x2, y2], ...]
        alpha (float): 数据项权重，控制平滑后的路径与原始路径的接近程度
                      值越大，路径越接近原始路径；值越小，平滑效果越明显
        beta (float): 平滑项权重，控制路径的平滑程度
                     值越大，路径越平滑；值越小，保持更多原始形状
        iterations (int): 迭代次数，控制优化的充分程度
                         次数越多，平滑效果越稳定，但计算时间越长
    
    Returns:
        list: 平滑后的路径点列表
    """
    path = np.array(path, dtype=float)
    smoothed_path = np.copy(path)
    
    for _ in range(iterations):
        for i in range(1, len(path) - 1):
            # 数据项梯度：使平滑路径接近原始路径
            grad_data = alpha * (smoothed_path[i] - path[i])
            # 平滑项梯度：使相邻点之间更加平滑
            grad_smooth = beta * (2 * smoothed_path[i] - smoothed_path[i-1] - smoothed_path[i+1])
            # 梯度下降更新
            smoothed_path[i] -= (grad_data + grad_smooth)
            
    return smoothed_path.tolist()

def resample_path(path, segment_length=1.0):
    """
    对路径进行等距重采样。

    Args:
        path (list): 路径点列表 [[x1, y1], ...]
        segment_length (float): 重采样后的点之间的距离。

    Returns:
        list: 重采样后的路径点列表。
    """
    path = np.array(path)
    # 计算沿路径的累积距离
    distances = np.cumsum(np.sqrt(np.sum(np.diff(path, axis=0)**2, axis=1)))
    distances = np.insert(distances, 0, 0) # 在开头插入0

    if distances[-1] < segment_length: # 路径太短
        return path.tolist()

    # 创建插值函数
    fx = interp1d(distances, path[:, 0], kind='cubic') # 使用三次样条插值
    fy = interp1d(distances, path[:, 1], kind='cubic')

    # 生成新的采样点
    new_distances = np.arange(0, distances[-1], segment_length)
    
    new_x = fx(new_distances)
    new_y = fy(new_distances)
    
    return np.vstack([new_x, new_y]).T.tolist()


def recalculate_angles(path_points):
    """为平滑后的路径点重新计算朝向角度"""
    # (此函数与上一版相同)
    points_with_angle = []
    for i, point in enumerate(path_points):
        if i == len(path_points) - 1:
            angle = points_with_angle[-1][2] if i > 0 else 0
        else:
            dx = path_points[i + 1][0] - point[0]
            dy = path_points[i + 1][1] - point[1]
            angle = math.atan2(dy, dx)
        
        points_with_angle.append((point[0], point[1], angle))
    return points_with_angle
"""
@file: path_smoother.py
@description:
该文件是一个独立的路径后处理工具集，提供了一个函数流水线 (pipeline)，其核心目标是
将一个粗糙、离散的几何路径点序列，转换为一条平滑、等距且带有精确朝向信息的高质量
轨迹。这个模块对于确保高级车辆控制器（尤其是模型预测控制MPC）能够稳定、精确地
跟踪目标路径至关重要。

核心功能流水线:

1.  **`smooth_path` (路径平滑):**
    - **目的:** 消除由离散点构成的原始路径中的尖锐拐角，使其在曲率上更加连续，
      从而更符合车辆的物理运动学特性。
    - **算法:** 采用了一种基于梯度下降的优化方法。通过迭代调整每个点的位置，
      在两个目标之间取得平衡：
        a. **数据保真项 (由`alpha`控制):** 使平滑后的点不至于离原始路径点太远。
        b. **平滑项 (由`beta`控制):** 使每个点趋向于其前后两个点的中点，从而产生
           平滑效果。

2.  **`resample_path` (路径重采样):**
    - **目的:** 将平滑后点间距不均匀的路径，转换为一条所有相邻点之间距离都严格
      相等的路径。
    - **重要性:** 等距的路径点对于控制器（如MPC）在预测时域内进行稳定的状态推演
      至关重要。同时，它也能根据需要生成更高或更低密度的路径点。
    - **算法:** 利用了`scipy`科学计算库中的`三次样条插值 (cubic spline interpolation)`。
      这是一种高精度的插值技术，能够在保持路径平滑性的前提下，精确计算出任意
      距离处的坐标点。

3.  **`recalculate_angles` (朝向角重计算):**
    - **目的:** 为最终路径上的每一个点计算出精确的切线方向（即车辆的理想朝向角）。
    - **重要性:** 控制器不仅需要知道目标位置 (x, y)，还需要知道在该位置的理想朝向，
      以便计算航向误差 (`psi_e`) 并进行有效控制。
    - **算法:** 通过计算当前点到下一个点的方向向量，并利用 `math.atan2` 函数来获得
      该向量与X轴正方向的夹角（弧度）。

使用流程:
这三个函数通常被依次调用，形成一个完整的数据处理链：
`原始路径 -> smooth_path -> resample_path -> recalculate_angles -> 最终可用轨迹`
"""
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
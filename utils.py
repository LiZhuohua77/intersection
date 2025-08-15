"""
@file: utils.py
@description:
该文件是一个独立的工具函数模块，主要提供项目所需的通用几何计算功能。
目前，该模块的核心功能是实现了一个基于“分离轴定理”(Separating Axis Theorem, SAT)
的精确碰撞检测算法，用于判断两个车辆的“有向边界框”(Oriented Bounding Box, OBB)
是否发生重叠。

核心功能与算法:

1.  **`check_obb_collision` (碰撞检测主函数):**
    - **目的:** 这是该模块对外暴露的主要接口，用于精确地判断任意两个处于不同位置和
      朝向的车辆是否发生了物理上的碰撞。
    - **算法:** 该函数完整地实现了**分离轴定理 (SAT)**。SAT是一个用于判断两个凸多边形
      是否碰撞的著名算法。其基本思想是：如果能找到一条轴线（即“分离轴”），使得
      两个多边形在该轴上的投影（可以看作是“影子”）互不重叠，那么这两个多边形就
      一定没有发生碰撞。反之，如果对于所有可能的分离轴，两个多边形的投影都存在
      重叠，那么它们就必然发生了碰撞。

2.  **算法步骤:**
    - **a. 获取分离轴 (`_get_axes`):** 对于两个矩形（车辆的OBB），总共存在4个需要
      被检测的分离轴（即两个矩形各自的两条边的法向量）。
    - **b. 投影 (`_project_on_axis`):** 对于每一个分离轴，函数会将两个车辆的四个
      角点都投影到这条轴上，形成两条一维线段。
    - **c. 重叠检查:** 检查这两条一维线段是否重叠。只要发现有任何一条分离轴上的
      投影是不重叠的，算法就可以立刻提前终止并断定“无碰撞”。
    - **d. 最终判断:** 只有当所有4个分离轴上的投影都发生了重叠，算法才会最终断定
      “有碰撞”。

3.  **辅助函数:**
    - `_get_obb_corners`: 根据车辆的状态（位置、朝向）和尺寸，计算其OBB在世界
      坐标系下的四个角点坐标。
    - `_get_axes`: 根据OBB的角点计算出其对应的分离轴。
    - `_project_on_axis`: 执行将一个形状投影到一条轴上的核心数学运算。

在项目中的作用:
- 该模块提供的 `check_obb_collision` 函数是仿真环境中安全判断的基石。它被
  `RLVehicle` 类在每个 `step` 中调用，以确定是否发生了碰撞。碰撞是RL环境中一个
  关键的终止条件，并且是奖励函数中一个非常重要的、带有巨大负反馈的事件。
"""
def _get_obb_corners(vehicle):
    """获取车辆的有向边界框（OBB）的四个角点坐标。"""
    x, y, psi = vehicle.state['x'], vehicle.state['y'], vehicle.state['psi']
    w, l = vehicle.width, vehicle.length
    
    # 计算旋转矩阵
    cos_psi = np.cos(psi)
    sin_psi = np.sin(psi)
    
    # 车辆局部坐标系下的四个角点
    local_corners = [
        np.array([l/2, w/2]),   # 右前
        np.array([-l/2, w/2]),  # 右后
        np.array([-l/2, -w/2]), # 左后
        np.array([l/2, -w/2])   # 左前
    ]
    
    # 将角点旋转并平移到世界坐标系
    world_corners = []
    for p in local_corners:
        rotated_x = p[0] * cos_psi - p[1] * sin_psi
        rotated_y = p[0] * sin_psi + p[1] * cos_psi
        world_corners.append(np.array([x + rotated_x, y + rotated_y]))
        
    return world_corners

def _get_axes(corners):
    """根据OBB的角点获取其分离轴（边的法向量）。"""
    axes = []
    for i in range(len(corners)):
        p1 = corners[i]
        p2 = corners[(i + 1) % len(corners)] # 获取下一个点，形成边
        
        edge = p2 - p1
        # 获取边的法向量（即分离轴）
        normal = np.array([-edge[1], edge[0]])
        
        # 归一化
        norm_val = np.linalg.norm(normal)
        if norm_val > 0:
            axes.append(normal / norm_val)
            
    # 对于矩形，我们只需要两个唯一的轴
    return [axes[0], axes[1]]

def _project_on_axis(corners, axis):
    """将一个形状的所有角点投影到一个轴上，并返回最小和最大投影值。"""
    min_proj = np.dot(corners[0], axis)
    max_proj = min_proj
    for i in range(1, len(corners)):
        projection = np.dot(corners[i], axis)
        if projection < min_proj:
            min_proj = projection
        if projection > max_proj:
            max_proj = projection
    return min_proj, max_proj

def check_obb_collision(vehicle_a, vehicle_b):
    """
    使用分离轴定理检查两个车辆的OBB是否碰撞。
    
    Returns:
        bool: True 如果碰撞，否则 False。
    """
    # 1. 获取两个车辆的角点和分离轴
    corners_a = _get_obb_corners(vehicle_a)
    corners_b = _get_obb_corners(vehicle_b)
    axes_a = _get_axes(corners_a)
    axes_b = _get_axes(corners_b)
    
    all_axes = axes_a + axes_b
    
    # 2. 遍历所有分离轴
    for axis in all_axes:
        # 3. 将两个形状投影到当前轴上
        min_a, max_a = _project_on_axis(corners_a, axis)
        min_b, max_b = _project_on_axis(corners_b, axis)
        
        # 4. 检查投影是否重叠。如果不重叠，则存在一条分离轴，两物体必然不碰撞。
        if max_a < min_b or max_b < min_a:
            return False # 找到分离轴，没有碰撞
            
    # 5. 如果所有轴上的投影都重叠，则两物体必然碰撞
    return True
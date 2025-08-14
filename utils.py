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
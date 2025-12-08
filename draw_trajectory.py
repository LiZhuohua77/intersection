import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from mpl_toolkits.mplot3d import Axes3D  # 导入 3D 工具

# 1. 先把 RL_AGENT 单独加进去
file_list = ["trajectory_vehicle_RL_AGENT.csv"]

# 2. 再把所有 unfinished 的文件都找出来
file_list += sorted(glob.glob("trajectory_unfinished_*.csv"))

# 创建 3D 图像
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

for path in file_list:
    # 读取每个轨迹文件
    df = pd.read_csv(path)

    # 假设现在列是：t, vehicle_id, x, y, vx, vy, psi, speed, steering
    # ——关键一步：按时间 t 排序，保证轨迹是按时间画出来的
    df = df.sort_values("t")

    t = df["t"]
    x = df["x"]
    y = df["y"]

    # 用文件名当作图例
    label = os.path.splitext(os.path.basename(path))[0]

    # 画 3D 线：这里把 t 当成 z 轴（竖直方向）
    ax.plot(x, y, t, label=label)

# 设置坐标轴标签
ax.set_xlabel("X 位置 (m)")
ax.set_ylabel("Y 位置 (m)")
ax.set_zlabel("时间 t")

ax.set_title("多辆车在交叉口的 3D 运行轨迹（X-Y-时间）")
ax.legend()
plt.show()

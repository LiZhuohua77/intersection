# 基于预测性场景树的自动驾驶交叉口决策强化学习平台

本项目是一个基于Python的仿真与研究平台，旨在开发和评估用于无信号交叉口场景的自动驾驶决策算法。项目的核心是实现了一个集成了**预测性场景树**的**约束强化学习（CRL）**智能体，该智能体能够通过在线预测其他车辆的多种可能未来意图，做出更安全、更高效的驾驶决策。

## ✨ 核心特性 (Features)

* **动态交通环境**: 背景车辆（HV）由经典的**智能驾驶员模型（IDM）**驱动，具有随机的驾驶“个性”（保守/普通/激进）和交互“意图”（让行/通行），行为高度动态且真实。
* **预测性场景树**: 智能体（AV）的观测空间被一个创新的预测模块增强，该模块实时生成关键对手车辆在不同意图下的未来轨迹，形成场景树。
* **先进网络架构**: 采用基于**GRU（门控循环单元）**的编码器来处理和压缩场景树中的时序轨迹数据，再与常规状态融合，送入Actor-Critic网络进行决策。
* **模块化与可扩展**: 遵循Gymnasium标准接口，代码结构清晰，分为环境、车辆、规划器、模型、预测器等多个解耦的模块。
* **Stable Baselines3 集成**: 训练和评估流程利用了`Stable Baselines3`框架，支持高效的并行训练和标准化的模型管理。

## 🛠️ 技术栈 (Tech Stack)

* **核心算法**: `stable-baselines3`, `torch`
* **仿真与环境**: `gymnasium`, `pygame`
* **科学计算与数据处理**: `numpy`, `pandas`, `matplotlib`
* **控制器优化**: `casadi`

## 🌊 核心数据流 (Core Data Flow)

以下描述了在使用 `Stable Baselines3` 训练时，**一次训练步 (`env.step`)** 中信息如何在系统各模块间流动：

1.  **环境启动 (`TrafficEnv`)**:
    * `TrafficEnv` 调用 `_get_observation()` 方法，开始为AV构建观测向量。

2.  **基础感知 (`RLVehicle` -> `TrafficEnv`)**:
    * `TrafficEnv` 调用 `rl_agent.get_ego_observation()` 获取AV自身的局部状态（速度、路径误差等）。
    * `TrafficEnv` 调用 `_get_surrounding_vehicles_observation()` 感知周围半径内的所有HV，获取它们的**当前**相对状态。

3.  **关键目标预测 (`prediction.py` -> `TrafficEnv`)**:
    * `TrafficEnv` 从周边车辆中识别出最关键的冲突HV。
    * `TrafficEnv` 调用 `self.predictor.predict()` **两次**：
        * 一次假设HV意图为 `YIELD`，生成“让行轨迹”。
        * 一次假设HV意图为 `GO`，生成“通行轨迹”。
    * `TrajectoryPredictor` 内部通过“幽灵车”和IDM模型进行快速前瞻模拟，返回两条**未来**轨迹。

4.  **观测向量组装 (`TrafficEnv`)**:
    * `TrafficEnv` 将 [AV自身状态]、[周边HV当前状态]、[让行轨迹] 和 [通行轨迹] **拼接**成一个巨大的、扁平化的观测向量。

5.  **特征提取与编码 (`agent.py` -> SB3)**:
    * 该观测向量被传递给SB3模型。
    * 模型内部的 `HybridFeaturesExtractor` 首先被调用。它将向量**拆分**，并使用两个**GRU编码器**将两条轨迹分别压缩成低维的特征向量（Embeddings）。
    * 最后，它将 [AV状态]、[HV状态] 和两个轨迹的 [Embeddings] **重新拼接**成一个信息密集的、最终的特征向量。

6.  **决策 (`stable_baselines3.PPO`)**:
    * 这个最终特征向量被送入PPO策略网络的**MLP主干**部分。
    * 网络输出一个动作（归一化的加速度和转向角）。

7.  **动作执行与世界演化**:
    * `TrafficEnv` 接收到动作，并调用 `rl_agent.apply_action()` 将其应用到AV的物理模型上。
    * `TrafficEnv` 调用 `traffic_manager.update_background_traffic()`，所有HV根据其各自的`LongitudinalPlanner`（内部使用IDM）更新自己的状态。

8.  **反馈与学习**:
    * `TrafficEnv` 计算奖励和终止信号。
    * SB3的PPO算法将 `(观测, 动作, 奖励, 下一观测)` 存入其内部的`RolloutBuffer`，并在收集到足够数据后进行策略网络更新。

## 🚀 使用方法 (Usage)

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 训练模型
使用 `train.py` 脚本。可以通过命令行参数调整算法和超参数。

# 使用4个并行环境训练PPO模型
python train.py --n-envs 4

# 查看所有可用参数
python train.py --help

### 3. 评估模型
使用 evaluate.py 脚本来可视化已训练好的模型。

# 加载并评估一个已保存的模型
python evaluate.py --model-path "models/ppo_xxxxxxxx/final_model.zip" --num-episodes 10
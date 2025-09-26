# 强化学习驱动的十字路口仿真平台（PPO / SAGI-PPO）

## 1. 项目简介

这是一个基于 Python + Pygame 的二维无信号十字路口交通仿真平台，用于开发与评估强化学习驾驶策略。当前训练/评估工作流以 PPO 与 SAGI-PPO 为主，环境遵循 Gymnasium 接口，可无缝对接主流 RL 框架。平台具备可视化引擎、可复现的场景工厂，以及更贴合真实驾驶的 NPC 分层“规划-控制”栈（纵向 IDM 决策 + 纵向 PID + 横向 MPC）。

核心特性：
- 完整 RL 工作流：`train.py`（训练）与 `evaluate.py`（可视化评估）；`main.py` 用于无策略/随机动作调试与观察
- 标准化接口：Gymnasium `Env` 封装（`TrafficEnv`）
- 场景工厂：一键搭建特定冲突情景，便于课程学习与压力测试
- 物理一致性与控制：自行车模型动力学；横向 MPC（CasADi）；纵向 PID/IDM
- 场景感知观测：内置预测器，编码冲突车辆“让行/通行”两种意图的未来轨迹到观测向量

## 2. 代码总览

- 工作流入口
  - `train.py`：训练脚本，支持 `--algo {ppo|sagi_ppo}`、断点续训、TensorBoard
  - `evaluate.py`：加载模型并在 Pygame 可视化环境评估，保存每回合 CSV 轨迹与速度曲线
  - `main.py`：调试用入口（随机/占位动作），查看道路、车辆与速度曲线
- 环境与仿真
  - `traffic_env.py`：Gymnasium 环境；构建观测；接入预测器；桥接 `TrafficManager`
  - `traffic.py`：场景工厂与背景车辆生命周期管理
  - `road.py`：道路/路径生成、冲突区、势场等
  - `vehicle.py`：`Vehicle`（NPC）与 `RLVehicle`（智能体），含动力学、奖励、碰撞/偏离检查
  - `longitudinal_planner.py`：NPC 纵向高层决策（含意图/个性、让行规则）
  - `longitudinal_control.py`：纵向 PID 控制
  - `lateral_control.py`：横向 MPC 控制（CasADi）
  - `prediction.py`：场景感知预测器（两意图分支）
  - `game_engine.py`：Pygame 可视化/交互引擎
- 算法
  - `ppo.py`：PPO 基线实现（Actor-Critic + GAE + 剪切比率）
  - `sagi_ppo.py`：SAGI-PPO（奖励/成本双 Critic，基于几何关系的三情景更新策略）
  - `ddpg.py`：DDPG 旧实现（保留供参考，当前工作流不依赖）
- 配置
  - `config.py`：全局参数（仿真步长、车辆参数、MPC 参数、观测维度、预测地平线等）

## 3. 安装与环境

必需依赖（按代码实际引用）：
- pygame（或 pygame-ce）、numpy、scipy、matplotlib、pandas
- gymnasium、torch、tensorboard
- casadi（横向 MPC 必需）

Windows PowerShell 快速安装（推荐使用虚拟环境）：

```powershell
# 1) 创建并激活虚拟环境
python -m venv .venv
./.venv/Scripts/Activate.ps1

# 2) 安装基础依赖（优先尝试 requirements.txt）
pip install -r requirements.txt

# 3) 若缺少依赖，补充安装
pip install gymnasium torch pandas tensorboard
pip install pygame-ce  # 或: pip install pygame
pip install casadi     # 横向MPC所需
```

提示：CasADi 的二进制包在部分 Python 版本/平台上可能不可用，可参考其官方安装指南。

## 4. 运行与使用

### 4.1 训练

```powershell
# PPO 基线（示例）
python train.py --algo ppo --total-episodes 2000 --buffer-size 2048 --update-epochs 4 --seed 42

# SAGI-PPO（示例）
python train.py --algo sagi_ppo --total-episodes 2000 --buffer-size 2048 --update-epochs 2 --cost-limit 30 --seed 42

# 断点续训（示例路径，指向模型目录）
python train.py --algo sagi_ppo --resume --model-path "models/sagi_ppo_YYYYMMDD-HHMMSS"

# 监控训练
tensorboard --logdir .\runs
```

产物与日志：
- 模型：`models/<algo_timestamp>/`（最终）与 `models/<algo_timestamp>/checkpoints/`（间隔保存）
- 训练日志：`runs/<algo_timestamp>/`（TensorBoard）与 `logs/<algo_timestamp>/training_stats.csv`

### 4.2 评估

```powershell
# 评估 PPO
python evaluate.py --algo ppo --model-dir "models/ppo_YYYYMMDD-HHMMSS" --num-episodes 3 --seed 8491

# 评估 SAGI-PPO
python evaluate.py --algo sagi_ppo --model-dir "models/sagi_ppo_YYYYMMDD-HHMMSS" --num-episodes 3 --seed 233
```

评估时将开启 Pygame 可视化窗口，默认循环场景为 `head_on_conflict`，自动保存：
- 每回合详细日志 CSV：`evaluation_logs/<model_name>/episode_*_*.csv`
- RL/NPC 车辆速度曲线（Matplotlib 窗口）

### 4.3 无策略调试/演示

```powershell
python main.py --scenario head_on_conflict
```

用于快速观察道路、交通与 UI 交互；在纯背景车流场景下会绘制 NPC 速度曲线。

### 4.4 交互快捷键（evaluate/main）

- Space/P：暂停/继续
- B：切换自行车模型可视化
- D：切换调试信息
- T：切换路径显示
- R：重置仿真
- 1/2/3/4：切换交通模式（轻度/正常/高峰/夜间）
- H：帮助
- ESC/Q：退出
- 鼠标滚轮：缩放；鼠标拖拽：平移视角

## 5. 场景与重置选项

通过 `env.reset(options={"scenario": name, "algo": "ppo|sagi_ppo"})` 切换场景。

内置场景（见 `traffic.py:TrafficManager.setup_scenario`）：
- random：常规训练用（包含 RL 智能体，运行中按概率生成 NPC）
- agent_only：仅 RL 车辆路径跟踪（基线调试）
- protected_left_turn：Agent 左转让行直行
- unprotected_left_turn：Agent 左转与 NPC 左转的博弈
- head_on_conflict：迎面冲突
- east_west_traffic：东西向 NPC 流（无 RL Agent）
- north_south_traffic：南北向 NPC 流（无 RL Agent）

无 RL 场景下，环境返回占位观测与 0 奖励，只用于交通流统计与对照数据生成。

## 6. 动作与观测空间

动作空间（`traffic_env.py`）：`Box(low=-1, high=1, shape=(2,))`
- a[0]：纵向加速度（归一化）。内部映射到实际加速度范围，参考 `config.MAX_ACCELERATION`。
- a[1]：预留横向通道（当前横向由 MPC 控制，网络输出不直接生效）。

观测空间维度自动计算（`config.py`）：
- `AV_OBS_DIM = 6`：自车局部状态（含路径误差等）
- `HV_OBS_DIM = 4 * NUM_OBSERVED_VEHICLES`：最相关冲突 NPC 的相对状态
- `PREDICTION_HORIZON` 与 `FEATURES_PER_STEP`：两条意图分支（YIELD/GO）的轨迹预测，共计 `2 * PREDICTION_HORIZON * FEATURES_PER_STEP`
- `TOTAL_OBS_DIM = AV_OBS_DIM + HV_OBS_DIM + 2 * PREDICTION_HORIZON * FEATURES_PER_STEP`

如调整 `NUM_OBSERVED_VEHICLES`、`PREDICTION_HORIZON` 等，请同步关注模型输入层维度与已训模型的兼容性。

## 7. 关键配置（摘自 `config.py`）

- 仿真：`SIMULATION_DT = 0.05`
- 车辆：质量/几何/侧偏刚度等；`MAX_ACCELERATION = 3.0 m/s²`；`MAX_STEERING_ANGLE = 30°`
- 控制：`MPC_HORIZON = 20`，`MPC_CONTROL_HORIZON = 5`，`PID_KP/KI/KD = 12000/800/100`
- 观测：`OBSERVATION_RADIUS = 80`，`NUM_OBSERVED_VEHICLES = 1`
- 预测：`PREDICTION_HORIZON = 40`，`FEATURES_PER_STEP = 3`

## 8. 算法说明（简要）

- PPO（`ppo.py`）
  - Actor-Critic 前馈结构，GAE 优势，剪切比率，熵正则
  - `RolloutBuffer` 支持 GAE 与归一化 return/advantage
- SAGI-PPO（`sagi_ppo.py`）
  - Actor + 双 Critic（奖励/成本），支持成本预算 `cost_limit`
  - 通过成本盈余 c 与优势相关性 p，动态选择三情景更新（自由探索/约束权衡/紧急恢复）
  - 记录 `sagi/*` 指标到 TensorBoard

DDPG（`ddpg.py`）作为参考保留，当前工作流默认不使用。

## 9. 输出与日志

- 训练：`runs/<run_name>/`（TensorBoard），`logs/<run_name>/training_stats.csv`
- 模型：`models/<run_name>/`（最终），`models/<run_name>/checkpoints/`（周期性保存）
- 评估：`evaluation_logs/<model_name>/episode_*_*.csv`；速度曲线 Matplotlib 窗口

## 10. 常见问题（FAQ）

- Pygame 窗口不开启/训练卡住？训练是无头；请使用 `evaluate.py` 或 `main.py` 可视化。
- 缺失依赖报错（例如 gymnasium/casadi/torch）？按第 3 节补齐安装。
- 观测维度不匹配？检查 `config.py` 的维度定义与 `TrafficEnv._get_observation` 拼接逻辑是否一致。
- MPC 求解失败？CasADi 在极低速度可能不稳定，代码带有回退（沿用上一控制）；确保 `casadi` 安装正确。
- 性能不足/帧率低？降低窗口尺寸、隐藏热力图、减少 NPC 数量或调低渲染帧率。

## 11. 许可与致谢

本项目用于学术研究与教学示例，涉及到的第三方库（Gymnasium、PyTorch、Pygame、CasADi 等）请遵循各自的许可证条款。


@file: config.py
@description: 
本文件包含整个交通仿真与强化学习项目的全局配置参数和常量。
该文件不包含函数定义，而是通过常量组织的方式提供以下配置类别：

1. 仿真参数 (Simulation Parameters):
   控制仿真器的基本设置，如时间步长(SIMULATION_DT)和渲染窗口尺寸。

2. 车辆物理参数 (Vehicle Physical Parameters):
   定义车辆动力学模型属性，包括质量、尺寸、转动惯量、轮胎特性等。
   还包括空气动力学参数和能量转换效率参数，用于功率消耗计算。

3. PID 控制器参数 (PID Controller Parameters):
   用于纵向(速度)控制的PID控制器参数，包括比例、积分、微分增益。

4. MPC 控制器参数 (MPC Controller Parameters):
   用于横向(路径跟踪)的模型预测控制器参数，包括预测时域和权重矩阵。

5. Gipps 模型参数 (Gipps' Model Parameters):
   定义背景车辆行为的参数，基于Gipps跟驰模型实现真实的交通流仿真。

6. Agent 参数 (Agent Parameters):
   定义强化学习智能体的行为约束和感知能力，如最大加速度和观测半径。

通过修改此文件中的值，可以调整仿真环境的各个方面而无需改动核心代码逻辑。

@file: game_engine.py
@description:
该文件提供了交通仿真项目的可视化引擎，基于Pygame构建，负责处理所有用户交互和图形渲染功能。
它采用模块化设计，将相机系统、输入处理和渲染逻辑分离成独立的类，提高代码的可维护性和扩展性。

主要类和功能:
1. Camera - 相机系统，处理视图变换、缩放和移动
   - world_to_screen/screen_to_world: 坐标系转换函数
   - zoom_in/zoom_out: 视图缩放功能
   - start_drag/update_drag/stop_drag: 视图拖动功能
   
2. InputHandler - 输入处理器，负责处理所有用户输入
   - handle_event: 分发和处理各类事件
   - 提供键盘控制功能: 暂停/继续、切换可视化选项、场景重置、帮助显示等
   - 提供鼠标控制功能: 拖拽视图、缩放视图
   
3. Renderer - 渲染器，负责所有图形绘制
   - render_frame: 渲染完整场景，包括世界对象和UI元素
   - 支持势场热力图的生成和显示
   - 提供多种UI元素: 暂停指示器、控制提示、相机信息、帮助面板等
   
4. GameEngine - 游戏引擎主类，协调其他组件工作
   - 初始化Pygame环境和主要组件
   - 提供主循环控制: 事件处理、渲染、帧率控制
   - 为外部代码提供简洁的接口

交互功能:
- 相机控制: 鼠标拖拽移动视图，鼠标滚轮缩放
- 模拟控制: 暂停/继续、重置仿真、场景生成
- 可视化选项: 切换车辆模型、调试信息、路径显示、势场热力图等

@file: lateral_control.py
@description:
该文件实现了基于模型预测控制(MPC)的车辆横向控制器，用于精确的路径跟踪。
该控制器根据当前车辆状态和参考轨迹，通过在线优化计算最佳转向角指令。

主要函数:
1. MPCController.__init__(): 初始化MPC控制器，设置参数和优化器
2. MPCController._build_state_space_model(): 根据当前纵向速度构建线性化的状态空间模型
3. MPCController._setup_optimizer(): 使用CasADi构建优化问题及约束
4. MPCController.solve(): 主接口函数，求解MPC优化问题并返回最优转向角

核心原理:
- 使用"自行车模型"作为车辆动力学模型
- 状态向量: [横向误差, 航向误差, 横向速度, 横摆角速度]
- 控制输入: 前轮转向角
- 优化目标: 最小化路径跟踪误差、控制输入及其变化率
- 采用滚动时域优化策略，每次只执行优化序列的第一步

@file: longitudinal_control.py
@description:
该文件实现了一个标准的 PID (比例-积分-微分) 控制器，专门用于车辆的纵向速度控制。
它负责根据上层规划模块提供的目标速度，计算出驱动车辆加速或减速所需的具体控制力
（即油门或刹车力）。

主要函数:
1. PIDController.__init__(): 初始化PID控制器，设置各项参数和初始状态
2. PIDController.step(): 核心控制函数，根据目标速度和当前速度计算控制输出

核心组件与原理:

1.  **PIDController (PID控制器类):**
    该控制器通过不断减小"目标速度"与"当前速度"之间的误差来实现精确的速度追踪。
    它由三个部分协同工作：

    - **比例 (Proportional, P):**
      `Kp * error` - 核心作用是纠正**当前**的误差。误差越大，施加的控制力就越强。
      它是控制器响应的主要部分。

    - **积分 (Integral, I):**
      `Ki * integral_error` - 核心作用是消除系统的**稳态误差**。通过累积**过去**的
      所有误差，它可以补偿那些仅靠P控制器无法完全克服的持续性小误差（例如，持续的
      空气阻力或坡度阻力），确保车辆最终能精确达到目标速度。

    - **微分 (Derivative, D):**
      `Kd * derivative_error` - 核心作用是预测**未来**的误差趋势，并提供阻尼以
      **抑制过冲**。通过观察误差的变化速率，它可以在车辆速度快速接近目标值时
      提前"踩刹车"，防止速度超出目标太多，使整个调节过程更加平稳、稳定。

2.  **关键实现细节:**
    - **积分抗饱和 (Integral Anti-Windup):** 在实现中，通过 `np.clip` 对积分项
      `integral_error` 的累积值进行了限制。这是一个至关重要的工程实践，可以防止
      在长时间存在较大误差时积分项过度累积（即"饱和"），从而避免当误差反向时
      系统因巨大的积分惯性而产生剧烈的超调。

接口与交互:
- **`step` 方法:** 这是控制器唯一的执行接口，输入目标速度和当前速度，输出纵向控制力（正值代表油门，负值代表刹车）。
- **系统层级:** 该PID控制器位于车辆控制架构的最底层，将上层规划器的速度指令转化为可以直接作用于车辆的物理力。

@file: longitudinal_planner.py
@description:
该文件定义了背景车辆（NPC, Non-Player Character）的纵向（前进速度）规划与决策逻辑。
它负责处理两种核心的驾驶行为：常规的车辆跟驰和复杂的交叉口让行决策，并为
车辆生成相应的速度规划曲线(Speed Profile)以供执行。

主要函数:
1. GippsModel.calculate_target_speed(): 计算基于Gipps模型的目标跟车速度
2. LongitudinalPlanner._generate_profile_to_target(): 生成从当前速度平滑过渡到目标速度的曲线
3. LongitudinalPlanner._generate_cruise_profile(): 生成以期望速度巡航的速度曲线
4. LongitudinalPlanner._generate_yielding_profile(): 生成减速-等待-加速的让行速度曲线
5. LongitudinalPlanner.get_potential_speed_profiles(): 为RL Agent提供不同交互策略的速度规划
6. LongitudinalPlanner._check_intersection_yielding(): 检查是否需要在交叉口让行
7. LongitudinalPlanner.generate_initial_profile(): 为新生成的车辆提供初始速度规划
8. LongitudinalPlanner.determine_action(): 核心决策函数，分析情况并返回控制指令

核心组件:

1.  **GippsModel (Gipps跟驰模型):**
    - **目的:** 实现一个经典的、基于规则的微观交通流模型，用于模拟车辆在单一车道上
      跟随前车的行为。
    - **原理:** 该模型通过分别计算"自由加速下的期望速度"和"为避免与前车碰撞所需
      的安全刹车速度"，并取两者中的较小值作为当前时刻的目标速度。这使得车辆能在
      **追求个人期望速度**和**保证行车安全**这两个目标之间做出动态平衡。

2.  **LongitudinalPlanner (纵向规划器):**
    - **目的:** 作为更高层次的决策模块，它整合了底层的跟驰模型，并加入了更复杂的、
      基于路权和冲突的交叉口交互逻辑。
    - **核心决策逻辑:** 通过determine_action方法实现，基于当前场景状态和预定义规则做出行为决策。
    - **交互建模:** 支持保守(cooperative)和激进(aggressive)两种不同的交互行为模式。

@file: main.py
@description:
该文件是整个强化学习交通仿真项目的主程序入口。
它负责初始化仿真环境和可视化引擎，并驱动整个仿真过程。

主要函数:
1. parse_args(): 解析命令行参数，确定要加载的交通场景
2. main(): 主函数，创建环境和游戏引擎，并执行主仿真循环

核心职责:

1.  **初始化 (Initialization):**
    - 创建并实例化两个最顶层的核心对象：
        - `TrafficEnv`: 遵循Gymnasium接口的仿真环境，是整个仿真的"后端"或"模型"。
        - `GameEngine`: 基于Pygame的可视化与交互引擎，是仿真的"前端"或"视图-控制器"。

2.  **主循环 (Main Loop):**
    - 运行一个经典的"事件处理 -> 状态更新 -> 渲染"的游戏循环。
    - 事件处理: 响应用户输入，如暂停、退出、切换视角等。
    - 状态更新: 在非暂停状态下，驱动仿真世界向前演化一步。
    - 渲染: 将当前状态绘制到屏幕上。

3.  **场景管理 (Scenario Management):**
    - 支持多种预定义交通场景的加载和切换
    - 分为含有RL智能体的场景和纯背景车辆的观察场景两大类

4.  **数据收集与可视化 (Data Collection & Visualization):**
    - 收集RL智能体和背景车辆的速度历史数据
    - 在回合结束时使用matplotlib绘制速度曲线进行分析

@file: path_smoother.py
@description: 
路径平滑处理工具集，提供完整的路径处理流水线。该模块将粗糙的离散路径点转换为平滑、
等距且具有精确朝向信息的高质量轨迹，为高级车辆控制器（如MPC）提供可靠的参考轨迹。

功能函数:

1. smooth_path(path, alpha=0.9, beta=0.5, iterations=100):
   - 使用基于梯度下降的优化方法平滑路径，消除尖锐拐角
   - 平衡原始路径保真度(alpha参数)与路径平滑度(beta参数)
   - 返回平滑后的路径点列表

2. resample_path(path, segment_length=1.0):
   - 将路径重采样为等距路径点序列
   - 使用三次样条插值确保路径平滑性
   - 返回等距采样的路径点列表

3. recalculate_angles(path_points):
   - 计算路径上每个点的切线方向（车辆理想朝向角）
   - 通过相邻点计算航向角，提供控制器所需的姿态信息
   - 返回带有位置和朝向角的点列表 [(x, y, angle), ...]

典型使用流程:
原始路径 -> smooth_path -> resample_path -> recalculate_angles -> 最终可用轨迹

@file: road.py
@description: 
此文件实现了一个四向交叉路口环境的静态几何结构、路径生成与冲突检测系统。

主要功能:
1. 交叉口几何定义 - 程序化生成标准四向交叉路口布局
2. 路径生成与处理 - 生成、平滑和缓存12种可能的行驶路线
3. 冲突检测 - 通过预定义的冲突矩阵判断路径冲突关系
4. 可视化渲染 - 提供一系列方法绘制道路元素到Pygame界面
5. 势场计算 - 为任意点计算势能值，用于车辆路径规划

主要类:
- Road: 作为仿真的"地图"，负责管理交叉路口的几何表示和渲染

主要函数:
- draw_*系列函数: 处理道路元素的可视化渲染
- generate_centerline_points: 生成原始路径点序列
- get_route_points: 获取平滑处理后的路径数据
- do_paths_conflict: 判断两条路径是否存在冲突
- get_potential_at_point: 计算点的势场值，用于导航




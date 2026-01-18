import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib as mpl
import matplotlib.ticker as ticker
from tensorboard.backend.event_processing import event_accumulator

# ================== 配置区 ==================
EXPERIMENTS = ["expt1", "expt2", "expt3", "expt4"]

# 建议：为了图例好看，可以在这里定义一个由 key 到 "论文显示名称" 的映射
ALGO_PRETTY_NAMES = {
    "ppo_gru": "PPO (GRU)",
    "ppo_mlp": "PPO (MLP)",
    "ppo_lagrangian_gru": "PPO-Lag (GRU)",
    "ppo_lagrangian_mlp": "PPO-Lag (MLP)",
    "sagi_ppo_gru": "SAGI-PPO (GRU)",
    "sagi_ppo_mlp": "SAGI-PPO (MLP)",
}

ALGOS = list(ALGO_PRETTY_NAMES.keys())

PREFERRED_TAGS = ["rollout/ep_rew_mean"]

# 平滑窗口
SMOOTH_WINDOW = 20  # 稍微调大一点，让主线更平滑，背景保留原始噪点
# ==========================================


def find_best_tag(ea, preferred_tags):
    scalar_tags = ea.Tags().get("scalars", [])
    for t in preferred_tags:
        if t in scalar_tags:
            return t
    raise ValueError(f"未找到 Reward Tag. 可用: {scalar_tags}")


def load_scalar_from_tb(logdir, tag=None):
    if not os.path.isdir(logdir):
        raise FileNotFoundError(f"日志目录不存在: {logdir}")
    ea = event_accumulator.EventAccumulator(
        logdir, size_guidance={event_accumulator.SCALARS: 0}
    )
    ea.Reload()
    if tag is None:
        used_tag = find_best_tag(ea, PREFERRED_TAGS)
    else:
        used_tag = tag
    scalar_events = ea.Scalars(used_tag)
    steps = np.array([e.step for e in scalar_events], dtype=np.int64)
    values = np.array([e.value for e in scalar_events], dtype=np.float64)
    return steps, values, used_tag


def smooth_curve(values, window):
    if window is None or window <= 1:
        return values
    n = len(values)
    if n == 0:
        return values
    if n < window:
        # 如果数据点比窗口还少，就直接返回原始数据
        return values

    kernel = np.ones(window, dtype=np.float64) / window

    # 边缘用 edge padding，避免用 0 填充导致开头被拉低
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(values, (pad_left, pad_right), mode="edge")

    smoothed = np.convolve(padded, kernel, mode="valid")
    # 这里 smoothed 的长度恰好是 n
    return smoothed



def save_to_csv(steps, values, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "reward"])
        for s, v in zip(steps, values):
            writer.writerow([int(s), float(v)])


# ================== 核心美化部分 ==================

def setup_publication_style():
    """设置符合顶刊标准的 Matplotlib 样式"""
    # 字体设置：Times New Roman 是学术界标准
    config = {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 10,           # 全局字号
        "axes.labelsize": 12,      # 轴标签字号
        "axes.titlesize": 12,      # 标题字号
        "xtick.labelsize": 10,     # x轴刻度字号
        "ytick.labelsize": 10,     # y轴刻度字号
        "legend.fontsize": 9,      # 图例字号
        
        "axes.linewidth": 1.0,     # 边框粗细
        "grid.linewidth": 0.5,     # 网格粗细
        "lines.linewidth": 1.5,    # 线条粗细
        
        "xtick.direction": "in",   # 刻度朝内 (很多期刊喜欢这样)
        "ytick.direction": "in",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        
        "axes.grid": True,         # 开启网格
        "grid.alpha": 0.3,         # 网格淡一些
        "grid.linestyle": "--",    # 网格虚线
        
        "figure.dpi": 300,         # 高清分辨率
    }
    mpl.rcParams.update(config)


def get_style_for_algo(algo_name):
    """
    语义化样式分配：
    让颜色代表算法类别，线型代表网络结构。
    这样图表逻辑性更强，不仅仅是乱序的颜色。
    """
    # 1. 定义颜色 (使用 Tableau 10 或 SciencePlots 配色)
    colors = {
        "ppo": "#299d8f",          
        "ppo_lagrangian": "#e9c46a", 
        "sagi_ppo": "#d87659",      
    }
    
    # 2. 定义线型
    # GRU: 实线 (Solid), MLP: 虚线 (Dashed)
    linestyles = {
        "gru": "--",
        "mlp": "-"
    }

    # 解析当前 algo_name
    # 假设命名规则是 'base_algo_structure' 或类似
    if "sagi" in algo_name:
        base_color = colors["sagi_ppo"]
    elif "lagrangian" in algo_name:
        base_color = colors["ppo_lagrangian"]
    else:
        base_color = colors["ppo"] # default ppo

    if "mlp" in algo_name:
        ls = linestyles["mlp"]
    else:
        ls = linestyles["gru"]

    return base_color, ls


def plot_experiment(expt_name, algo_curves, out_dir):
    if not algo_curves:
        return

    setup_publication_style()
    
    # 创建画布，采用黄金比例或适合双栏论文的尺寸 (宽 3.5 inch 左右是单栏，7 inch 是通栏)
    # 这里设定为稍宽一点以便放下图例
    fig, ax = plt.subplots(figsize=(7, 4.5))

    # 遍历绘制
    # 为了图例顺序好看，我们可以按 ALGOS 列表的顺序来画
    for algo_key in ALGOS:
        if algo_key not in algo_curves:
            continue
        
        steps, rewards = algo_curves[algo_key]
        if len(steps) == 0:
            continue

        # 获取语义化样式
        color, ls = get_style_for_algo(algo_key)
        label = ALGO_PRETTY_NAMES.get(algo_key, algo_key)

        # 1. 绘制原始数据背景 (Shadow) - 增加真实感
        # alpha 设置得很低，作为背景噪音显示
        ax.plot(steps, rewards, color=color, alpha=0.15, linewidth=0.5, zorder=1)

        # 2. 绘制平滑后的主线 (Main Curve)
        rewards_sm = smooth_curve(rewards, SMOOTH_WINDOW)
        # 对齐 steps 长度
        if len(rewards_sm) < len(rewards):
            steps_sm = steps[SMOOTH_WINDOW - 1:]
        else:
            steps_sm = steps
            
        ax.plot(steps_sm, rewards_sm, 
                color=color, linestyle=ls, linewidth=2.0, 
                label=label, zorder=10)

    # ================== 坐标轴美化 ==================
    ax.set_xlabel("Training Steps", fontweight='normal')
    ax.set_ylabel("Average Episode Reward", fontweight='normal')
    # 标题可选，论文中通常把标题写在 caption 里，图上不写。
    # 如果你需要写，可以用 ax.set_title(expt_name)

    # X轴使用科学计数法 (例如 2.5 x 10^6)
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-1, 1)) 
    formatter.set_scientific(True)
    ax.xaxis.set_major_formatter(formatter)
    
    # 去除上方和右侧的边框 (Spines)，显得更现代
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ================== 图例优化 ==================
    # 将图例放在图的上方外部，横向排列，避免遮挡曲线
    ax.legend(
        loc='upper center', 
        bbox_to_anchor=(0.5, 1.15), # 放到图表上方
        ncol=3,                     # 分3列显示
        frameon=False,              # 去掉图例边框
        handletextpad=0.5,
        columnspacing=1.5
    )

    plt.tight_layout()

    # 保存
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, f"{expt_name}_pretty.png")
    pdf_path = os.path.join(out_dir, f"{expt_name}_pretty.pdf")

    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight') # PDF 格式最适合 LaTeX
    print(f"[INFO] 已保存精美图像: {pdf_path}")
    plt.close()

# ==============================================

def main():
    # 假设脚本在 intersection/
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")
    export_root = os.path.join(base_dir, "exported")

    for expt in EXPERIMENTS:
        expt_dir = os.path.join(models_dir, expt, "data")
        if not os.path.isdir(expt_dir):
            print(f"[WARN] {expt} 不存在, 跳过.")
            continue

        print(f"\n[INFO] 处理实验: {expt}")
        algo_curves = {}

        for algo in ALGOS:
            logdir = os.path.join(expt_dir, algo)
            if not os.path.isdir(logdir):
                continue

            try:
                steps, rewards, _ = load_scalar_from_tb(logdir)

                # --------- 只对实验三做截断 -----------
                if expt == "expt3":
                    mask = steps <= 3.5e7
                    steps = steps[mask]
                    rewards = rewards[mask]
                # ------------------------------------

                # 保存 CSV（截断后再保存）
                csv_dir = os.path.join(export_root, expt, "csv")
                save_to_csv(steps, rewards, os.path.join(csv_dir, f"{algo}.csv"))

                algo_curves[algo] = (steps, rewards)
            except Exception as e:
                print(f"[ERROR] {algo} 加载失败: {e}")


        # 画图
        fig_dir = os.path.join(export_root, expt, "figs")
        plot_experiment(expt, algo_curves, fig_dir)

if __name__ == "__main__":
    main()
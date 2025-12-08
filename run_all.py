import os
import subprocess
import sys

# 项目根目录
BASE_DIR = r"D:/Code/intersection"

# 用当前 Python 解释器来跑，避免环境不一致
PYTHON = sys.executable

# 你要评估的 6 种算法（模型文件名还是 algo_final_model.zip）
ALGOS = [
    "sagi_ppo_mlp",
    "sagi_ppo_gru",
    "ppo_gru",
    "ppo_mlp",
    "ppo_lagrangian_gru",
    "ppo_lagrangian_mlp",
]

NUM_EPISODES = 50

# expt 与场景的对应关系：
#   expt1 -> agent_only_simple
#   expt2 -> crossing_conflict
#   expt3 -> random_traffic
EXPT_SCENARIO_LIST = [
    ("expt1", "agent_only_simple"),
    ("expt2", "crossing_conflict"),
    ("expt3", "random_traffic"),
]


def main():
    # 切到工程目录，这样 evaluate_headless.py 里的相对引用没问题
    os.chdir(BASE_DIR)

    for expt_name, scenario in EXPT_SCENARIO_LIST:
        base_model_dir = os.path.join(BASE_DIR, "models", expt_name)

        print("\n" + "#" * 80)
        print(f"开始评估实验目录: {expt_name}")
        print(f"模型目录: {base_model_dir}")
        print(f"对应场景: {scenario}")
        print("#" * 80 + "\n")

        for algo in ALGOS:
            model_filename = f"{algo}_final_model.zip"
            model_path = os.path.join(base_model_dir, model_filename)

            print("\n" + "=" * 80)
            print(f"[{expt_name}] 开始评估算法: {algo}")
            print(f"模型路径: {model_path}")
            print(f"回合数:   {NUM_EPISODES}")
            print(f"场景:     {scenario}")
            print("=" * 80 + "\n")

            # 先简单检查一下模型文件是否存在，防止路径写错
            if not os.path.isfile(model_path):
                print(f"[WARNING] 找不到模型文件: {model_path}，跳过该模型。")
                continue

            cmd = [
                PYTHON,
                "evaluate_headless.py",
                "--algo", algo,
                "--model_path", model_path,
                "--num-episodes", str(NUM_EPISODES),
                "--scenario", scenario,
            ]

            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"[WARNING] 在 {expt_name} 中评估 {algo} 时出错，跳过该模型。错误信息：{e}")


if __name__ == "__main__":
    main()

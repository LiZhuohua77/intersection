import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------
# 1. 数据录入
# --------------------------

records = [
    # Scenario 1
    {'scenario': 'S1', 'algo': 'PPO-Lag-GRU',
     'reward_mean': 1393.76, 'reward_std': 741.38,
     'cost_mean': 585.55, 'cost_std': 497.84,
     'time': 25.81, 'jerk': 4.42, 'scte': 0.28,
     'energy': 0.0893, 'success': 100.0, 'collision': 0.0},

    {'scenario': 'S1', 'algo': 'PPO-Lag-MLP',
     'reward_mean': 2035.84, 'reward_std': 451.74,
     'cost_mean': 257.85, 'cost_std': 299.95,
     'time': 26.10, 'jerk': 0.89, 'scte': -0.11,
     'energy': 0.0992, 'success': 100.0, 'collision': 0.0},

    {'scenario': 'S1', 'algo': 'PPO-GRU',
     'reward_mean': 1905.23, 'reward_std': 428.08,
     'cost_mean': 331.24, 'cost_std': 296.77,
     'time': 25.11, 'jerk': 0.04, 'scte': 0.04,
     'energy': 0.0663, 'success': 100.0, 'collision': 0.0},

    {'scenario': 'S1', 'algo': 'PPO-MLP',
     'reward_mean': 2198.12, 'reward_std': 333.38,
     'cost_mean': 216.11, 'cost_std': 229.86,
     'time': 26.40, 'jerk': 0.14, 'scte': -0.03,
     'energy': 0.0704, 'success': 100.0, 'collision': 0.0},

    {'scenario': 'S1', 'algo': 'SAGI-GRU',
     'reward_mean': 2229.59, 'reward_std': 23.10,
     'cost_mean': 690.60, 'cost_std': 538.08,
     'time': 25.53, 'jerk': 0.28, 'scte': 0.28,
     'energy': 0.0892, 'success': 100.0, 'collision': 0.0},

    {'scenario': 'S1', 'algo': 'SAGI-MLP',
     'reward_mean': 2215.39, 'reward_std': 62.87,
     'cost_mean': 286.43, 'cost_std': 335.94,
     'time': 25.62, 'jerk': 12.62, 'scte': 0.16,
     'energy': 0.1743, 'success': 100.0, 'collision': 0.0},

    # Scenario 2
    {'scenario': 'S2', 'algo': 'PPO-Lag-GRU',
     'reward_mean': 2327.44, 'reward_std': 293.08,
     'cost_mean': 0.00, 'cost_std': 0.00,
     'time': 24.02, 'jerk': 0.03, 'scte': -0.17,
     'energy': 0.0779, 'success': 94.0, 'collision': 6.0},

    {'scenario': 'S2', 'algo': 'PPO-Lag-MLP',
     'reward_mean': 2429.66, 'reward_std': 18.55,
     'cost_mean': 0.00, 'cost_std': 0.00,
     'time': 25.17, 'jerk': 0.05, 'scte': 0.00,
     'energy': 0.0829, 'success': 100.0, 'collision': 0.0},

    {'scenario': 'S2', 'algo': 'PPO-GRU',
     'reward_mean': 92.59, 'reward_std': 1805.51,
     'cost_mean': 1039.79, 'cost_std': 1210.78,
     'time': 18.16, 'jerk': 0.08, 'scte': 1.63,
     'energy': 0.0623, 'success': 69.0, 'collision': 7.0},

    {'scenario': 'S2', 'algo': 'PPO-MLP',
     'reward_mean': 2452.22, 'reward_std': 3.05,
     'cost_mean': 0.00, 'cost_std': 0.00,
     'time': 25.14, 'jerk': 0.19, 'scte': -0.04,
     'energy': 0.0738, 'success': 100.0, 'collision': 0.0},

    {'scenario': 'S2', 'algo': 'SAGI-GRU',
     'reward_mean': 2358.34, 'reward_std': 313.67,
     'cost_mean': 0.00, 'cost_std': 0.00,
     'time': 24.59, 'jerk': 0.03, 'scte': -0.01,
     'energy': 0.0682, 'success': 94.0, 'collision': 6.0},

    {'scenario': 'S2', 'algo': 'SAGI-MLP',
     'reward_mean': 2408.49, 'reward_std': 4.58,
     'cost_mean': 0.00, 'cost_std': 0.00,
     'time': 24.85, 'jerk': 0.01, 'scte': -0.10,
     'energy': 0.0840, 'success': 100.0, 'collision': 0.0},

    # Scenario 3
    {'scenario': 'S3', 'algo': 'PPO-Lag-GRU',
     'reward_mean': 2014.05, 'reward_std': 433.75,
     'cost_mean': 248.61, 'cost_std': 270.99,
     'time': 24.96, 'jerk': 0.05, 'scte': -0.04,
     'energy': 0.0812, 'success': 97.0, 'collision': 3.0},

    {'scenario': 'S3', 'algo': 'PPO-Lag-MLP',
     'reward_mean': 2346.21, 'reward_std': 280.09,
     'cost_mean': 104.15, 'cost_std': 145.46,
     'time': 26.53, 'jerk': 0.03, 'scte': -0.05,
     'energy': 0.0747, 'success': 97.0, 'collision': 3.0},

    {'scenario': 'S3', 'algo': 'PPO-GRU',
     'reward_mean': 1786.90, 'reward_std': 599.98,
     'cost_mean': 384.03, 'cost_std': 373.49,
     'time': 25.19, 'jerk': 0.04, 'scte': -0.16,
     'energy': 0.0844, 'success': 95.0, 'collision': 5.0},

    {'scenario': 'S3', 'algo': 'PPO-MLP',
     'reward_mean': 2394.85, 'reward_std': 252.63,
     'cost_mean': 90.15, 'cost_std': 142.39,
     'time': 26.78, 'jerk': 0.21, 'scte': 0.00,
     'energy': 0.0683, 'success': 99.0, 'collision': 1.0},

    {'scenario': 'S3', 'algo': 'SAGI-GRU',
     'reward_mean': 2441.26, 'reward_std': 253.84,
     'cost_mean': 223.68, 'cost_std': 262.57,
     'time': 25.79, 'jerk': 0.04, 'scte': -0.08,
     'energy': 0.0709, 'success': 96.0, 'collision': 4.0},

    {'scenario': 'S3', 'algo': 'SAGI-MLP',
     'reward_mean': 2474.67, 'reward_std': 205.40,
     'cost_mean': 140.25, 'cost_std': 177.37,
     'time': 25.90, 'jerk': 0.07, 'scte': -0.02,
     'energy': 0.0742, 'success': 97.5, 'collision': 2.5},
]

df = pd.DataFrame(records)

# 保证输出目录存在
os.makedirs("figs", exist_ok=True)

algos_order = ['PPO-Lag-GRU', 'PPO-Lag-MLP', 'PPO-GRU', 'PPO-MLP', 'SAGI-GRU', 'SAGI-MLP']
scen_order = ['S1', 'S2', 'S3']
scen_labels = {'S1': 'Scenario 1', 'S2': 'Scenario 2', 'S3': 'Scenario 3'}

# --------------------------
# 2. 碰撞率对比图（Fig. collision_rates）
# --------------------------

plt.figure(figsize=(8, 4))
x = np.arange(len(algos_order))
width = 0.22

for i, scen in enumerate(scen_order):
    sub = df[df['scenario'] == scen].set_index('algo')
    coll = [sub.loc[a, 'collision'] for a in algos_order]
    plt.bar(x + i * width, coll, width, label=scen_labels[scen])

plt.xticks(x + width, algos_order, rotation=45, ha='right')
plt.ylabel('Collision rate [%]')
plt.ylim(0, 10)
plt.legend()
plt.tight_layout()
plt.savefig("figs/collision_rates.pdf")
plt.close()

# --------------------------
# 3. Reward vs. collision（Fig. reward_vs_collision）
# --------------------------

plt.figure(figsize=(6, 5))

markers = {'S1': 'o', 'S2': 's', 'S3': 'D'}
colors = {'S1': 'tab:blue', 'S2': 'tab:orange', 'S3': 'tab:green'}

for scen in scen_order:
    sub = df[df['scenario'] == scen]
    plt.scatter(sub['collision'], sub['reward_mean'],
                marker=markers[scen], c=colors[scen],
                label=scen_labels[scen])
    # 可选：标注算法名
    for _, row in sub.iterrows():
        plt.text(row['collision'] + 0.05, row['reward_mean'],
                 row['algo'], fontsize=8)

plt.xlabel('Collision rate [%]')
plt.ylabel('Average return')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("figs/reward_vs_collision.pdf")
plt.close()

# --------------------------
# 4. Scenario 3 的 jerk vs energy（Fig. jerk_vs_energy_scen3）
# --------------------------

plt.figure(figsize=(6, 5))
sub3 = df[df['scenario'] == 'S3']

plt.scatter(sub3['energy'], sub3['jerk'])

for _, row in sub3.iterrows():
    plt.text(row['energy'] + 0.0005, row['jerk'],
             row['algo'], fontsize=8)

plt.xlabel('Net electrical energy [kWh]')
plt.ylabel('Average jerk [m/s$^3$]')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("figs/jerk_vs_energy_scen3.pdf")
plt.close()

print("Saved figures to figs/:")
print("  - collision_rates.pdf")
print("  - reward_vs_collision.pdf")
print("  - jerk_vs_energy_scen3.pdf")

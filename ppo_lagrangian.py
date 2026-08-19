"""PPO-Lagrangian baseline sharing SAGI-PPO's constrained optimizer."""

from typing import Optional, Type, Union

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv

# 关键：从您现有的 SAGI-PPO 文件中导入基类
# 我们假设您的 SAGIPPO 类在 'sagi_ppo.py' 文件中
from sagi_ppo import SAGIPPO

class PPOLagrangian(SAGIPPO):
    """
    PPO-Lagrangian 算法 (作为 SAGI-PPO 的基线)。

    该类继承自 SAGIPPO, 从而确保了：
    1. 共享相同的 ActorCriticCostPolicy (策略网络结构)
    2. 共享相同的 SAGIRolloutBuffer (经验回放缓冲区)
    3. 共享相同的 collect_rollouts 方法
    4. 共享所有相同的超参数 (cost_limit, lambda_lr, cost_vf_coef 等)

    这使得本算法与 SAGI-PPO 的对比 (Ablation Study) 绝对公平。

    唯一的区别是 *重写* (override) 了 `train()` 方法:
    - 移除了 SAGI-PPO 的 (c, p) KKT 诊断 (Case A 和 Case C)。
    - 移除了昂贵的梯度内积 `p` 的计算。
    - *始终* 执行 Case B (PPO-Lagrangian) 的更新逻辑。
    """

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        # --- 复制 SAGIPPO 的所有参数以确保签名一致 ---
        initial_cost_limit: float = 500.0,
        final_cost_limit: float = 30.0,
        decay_start_step: Optional[int] = None,
        cost_warmup_fraction: float = 0.10,
        cost_anneal_fraction: float = 0.40,
        lambda_lr: float = 0.035, 
        cost_vf_coef: float = 0.5,
        **kwargs,
    ):
        # 初始化父类 (SAGIPPO)
        # 这将自动设置好 cost_vf, lambda_k (self.lambda_), cost_limit, 
        # SAGIRolloutBuffer, ActorCriticCostPolicy 等
        super().__init__(
            policy=policy,
            env=env,
            initial_cost_limit=initial_cost_limit,
            final_cost_limit=final_cost_limit,
            decay_start_step=decay_start_step,
            cost_warmup_fraction=cost_warmup_fraction,
            cost_anneal_fraction=cost_anneal_fraction,
            lambda_lr=lambda_lr,
            cost_vf_coef=cost_vf_coef,
            **kwargs,
        )

    def train(self) -> None:
        """
        重写的训练方法 (PPO-Lagrangian 逻辑)。
        
        该方法移除了 Case A (奖励) 和 Case C (安全) 的诊断,
        强制执行标准的 PPO-Lagrangian (Case B) 更新。
        """
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        # --- 1. [PPO-L] 使用与 SAGI-PPO 完全相同的三阶段成本日程 ---
        current_step = self.num_timesteps
        total_steps = self._total_timesteps
        current_cost_limit, schedule_phase = self.get_cost_limit(current_step, total_steps)
        
        self.cost_limit = current_cost_limit
        # (日志记录前缀改为 'train/' 以便与 SAGI-PPO 的 'sagi/' 区分)
        self.logger.record("train/current_cost_limit", self.cost_limit)
        self.logger.record("train/cost_schedule_phase", schedule_phase)

        clip_range = self.clip_range(self._current_progress_remaining)

        # --- 2. [PPO-L] 计算成本盈余 'c' (与 SAGI-PPO 相同) ---
        j_c_k = self.rollout_buffer.get_mean_episode_costs()
        c = j_c_k - self.cost_limit
        self.logger.record("train/empirical_discounted_cost", j_c_k)

        # --- 3. [PPO-L] 更新 Lambda (原对偶梯度上升) ---
        # (这在 SAGI-PPO 中是在 A-B-C 逻辑之后做的)
        self.lambda_ = max(0, self.lambda_ + self.lambda_lr * c)
        
        self.logger.record("train/cost_surplus_c", c)
        self.logger.record("train/lambda", self.lambda_)
        
        # --- 4. [PPO-L] 准备数据缓冲区 (与 SAGI-PPO 相同) ---
        # Reward and cost arrays are flattened together and sampled with one permutation.
        self.rollout_buffer.prepare_for_sampling()
        original_advantages = self.rollout_buffer.advantages.copy() # (A_R)
        cost_advantages_flat = self.rollout_buffer.cost_advantages # (A_C)

        # --- 5. [PPO-L] 计算拉格朗日优势 (强制 Case B) ---
        # (移除了 SAGI-PPO 的 'p' 计算和 'if/elif/else' 逻辑)
        # (这就是 PPO-Lagrangian 和 SAGI-PPO 的唯一区别)
        self.rollout_buffer.advantages = (original_advantages - self.lambda_ * cost_advantages_flat) / (1 + self.lambda_)

        # --- 6. [PPO-L] Run the shared PPO optimizer for all configured epochs. ---
        try:
            self._optimize_policy(clip_range)
        finally:
            # Restore reward advantages for logging and any later buffer users.
            self.rollout_buffer.advantages = original_advantages

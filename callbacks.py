# callbacks.py
from stable_baselines3.common.callbacks import BaseCallback

class CurriculumCallback(BaseCallback):
    """
    一个自定义的回调函数，用于根据训练进度更新环境参数（例如HV的初始速度）。
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        # 您可以在这里添加更多控制参数，例如 ramp-up 的总步数等
        # 但我们现在直接使用总步数 self.model._total_timesteps

    def _on_step(self) -> bool:
        """
        这个函数会在每个训练步骤后被调用。
        """
        # 为了效率，我们不需要每一步都更新，可以在每个 rollout 结束后更新一次
        # self.n_calls 是回调被调用的总次数
        # self.model.n_steps 是每个 rollout 的步数
        if self.n_calls % self.model.n_steps == 0:
            current_step = self.num_timesteps      # 当前已进行的训练步数
            total_steps = self.model._total_timesteps # 训练的总步数
            
            # 使用 env_method 将信息传递给 VecEnv 中的所有子环境
            # 我们将调用子环境中名为 'update_curriculum' 的方法
            self.training_env.env_method('update_curriculum', current_step, total_steps)
            
            if self.verbose > 0:
                # 打印日志，确认回调在工作
                print(f"Callback: Updated curriculum parameters at step {current_step}")
                
        # 返回 True 表示继续训练
        return True
"""
模块: callbacks.py

概述:
此文件包含用于 Stable Baselines3 训练过程的自定义回调函数。
主要实现了一个课程学习（Curriculum Learning）的回调，它可以在训练期间动态地调整环境的参数，
以实现从易到难的渐进式学习。

类:
- CurriculumCallback: 一个根据训练进度更新环境参数的自定义回调。
"""
from stable_baselines3.common.callbacks import BaseCallback

class CurriculumCallback(BaseCallback):
    """一个自定义的回调函数，用于实现课程学习。

    此回调会在训练过程中，在每个 rollout 结束时，根据当前的训练进度
    （已完成的步数与总步数的比例）调用环境的特定方法（'update_curriculum'），
    从而允许环境动态地调整其难度或其他参数。
    """
    def __init__(self, verbose: int = 0):
        """初始化 CurriculumCallback。

        Args:
            verbose (int): 日志详细程度级别。0 表示无输出，1 表示输出信息。
        """
        super().__init__(verbose)
        # 您可以在这里添加更多控制参数，例如 ramp-up 的总步数等
        # 但我们现在直接使用总步数 self.model._total_timesteps

    def _on_step(self) -> bool:
        """在每个训练步骤后被调用，用于检查并更新课程。

        此方法是回调的核心。为了提高效率，它并非在每一步都执行更新，
        而是在每个 rollout 收集周期结束时触发。触发时，它会获取当前的
        总训练步数和预设的总训练步数，并将这两个值传递给所有并行环境
        实例的 'update_curriculum' 方法。

        Returns:
            bool: 返回 True 表示继续训练，返回 False 则会提前终止训练。
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
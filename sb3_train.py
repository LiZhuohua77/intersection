from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from agent import HybridFeaturesExtractor # 导入我们新的特征提取器
from config import * # 导入维度定义
from traffic_env import TrafficEnv

env = TrafficEnv()

# 定义策略网络的参数
policy_kwargs = dict(
    features_extractor_class=HybridFeaturesExtractor,
    features_extractor_kwargs=dict(
        av_obs_dim=AV_OBS_DIM,
        hv_obs_dim=HV_OBS_DIM,
        traj_len=PREDICTION_HORIZON,
        traj_feat_dim=FEATURES_PER_STEP,
        rnn_hidden_dim=64 # 或者从args传入
    ),
    net_arch=[dict(pi=[256, 256], vf=[256, 256])] # 定义MLP部分的结构
)

# 创建PPO模型，并传入自定义的策略参数
model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)

# 开始训练
model.learn(total_timesteps=1000000)
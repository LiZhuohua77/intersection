"""
@file: agent.py
@description:
定义了用于处理复杂、包含序列化轨迹数据的观测空间的Actor-Critic网络架构。

该模块的核心是一个混合网络 (HybridActorCritic)，它包含：
1. 两个独立的、基于GRU的“轨迹编码器”，用于分别处理“让行”和“通行”两种假设下的未来轨迹序列。
2. 一个主干网络，将编码后的轨迹特征与车辆的常规状态信息融合。
3. 最终的Actor和Critic头部，用于输出策略分布和价值估计。

这个架构取代了使用简单MLP处理高维观测的传统方法，能够更有效地从时序数据中提取信息。
"""
import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym

# 权重初始化函数
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
        nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.GRU):
        for name, param in m.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)

class SimpleFeaturesExtractor(BaseFeaturesExtractor):
    """
    一个简化的特征提取器，故意忽略所有轨迹序列数据。
    它只提取AV和HV的当前状态，用于创建一个"看不懂"轨迹的基线模型。
    """
    def __init__(self, observation_space: gym.Space, av_obs_dim: int, hv_obs_dim: int):
        # 提取出的特征维度就是 AV 和 HV 状态维度之和
        features_dim = av_obs_dim + hv_obs_dim
        super().__init__(observation_space, features_dim)
        
        self.av_obs_dim = av_obs_dim
        self.hv_obs_dim = hv_obs_dim
        self.end_idx = av_obs_dim + hv_obs_dim

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # 只截取观测向量的开头部分（AV状态 + HV状态）
        return observations[:, :self.end_idx]
    
class HybridFeaturesExtractor(BaseFeaturesExtractor):
    """
    一个为SB3设计的、能够处理我们混合观测空间的自定义特征提取器。
    """
    def __init__(self, observation_space: gym.Space, av_obs_dim, hv_obs_dim, 
                 traj_len, traj_feat_dim, rnn_hidden_dim=64):
        
        # 计算特征提取后的最终维度
        features_dim = av_obs_dim + hv_obs_dim + (2 * rnn_hidden_dim)
        super().__init__(observation_space, features_dim)

        self.av_obs_dim = av_obs_dim
        self.hv_obs_dim = hv_obs_dim
        self.traj_len = traj_len
        self.traj_feat_dim = traj_feat_dim
        
        # 定义轨迹编码器
        self.yield_traj_encoder = nn.GRU(input_size=traj_feat_dim, hidden_size=rnn_hidden_dim, batch_first=True)
        self.go_traj_encoder = nn.GRU(input_size=traj_feat_dim, hidden_size=rnn_hidden_dim, batch_first=True)

        # 初始化权重
        self.apply(init_weights) # 假设init_weights函数也在这个文件里

    def _split_observation(self, observation):
        """辅助函数，将扁平的观测向量拆分为有意义的几部分。"""
        current_idx = 0
        
        av_obs = observation[:, current_idx : current_idx + self.av_obs_dim]
        current_idx += self.av_obs_dim
        
        hv_obs = observation[:, current_idx : current_idx + self.hv_obs_dim]
        current_idx += self.hv_obs_dim
        
        traj_flat_len = self.traj_len * self.traj_feat_dim
        
        yield_traj_flat = observation[:, current_idx : current_idx + traj_flat_len]
        current_idx += traj_flat_len
        
        go_traj_flat = observation[:, current_idx : current_idx + traj_flat_len]
        
        # 将扁平的轨迹数据重塑为 (batch_size, sequence_length, feature_dim)
        yield_traj = yield_traj_flat.view(-1, self.traj_len, self.traj_feat_dim)
        go_traj = go_traj_flat.view(-1, self.traj_len, self.traj_feat_dim)
        
        return av_obs, hv_obs, yield_traj, go_traj

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # 1. 拆分观测
        av_obs, hv_obs, yield_traj, go_traj = self._split_observation(observations)
        
        # 2. 编码轨迹
        _, yield_embedding = self.yield_traj_encoder(yield_traj)
        _, go_embedding = self.go_traj_encoder(go_traj)
        
        yield_embedding = yield_embedding.squeeze(0)
        go_embedding = go_embedding.squeeze(0)
        
        # 3. 拼接成最终的特征向量
        combined_features = torch.cat([av_obs, hv_obs, yield_embedding, go_embedding], dim=1)
        return combined_features
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

class HybridActorCritic(nn.Module):
    """
    [新] 一个能够处理混合数据（常规状态 + 轨迹序列）的Actor-Critic网络。
    """
    def __init__(self, av_obs_dim, hv_obs_dim, traj_len, traj_feat_dim, action_dim, hidden_dim=256, rnn_hidden_dim=64):
        """
        初始化混合网络架构。

        参数:
            av_obs_dim (int): AV自身状态的维度。
            hv_obs_dim (int): HV当前相对状态的维度。
            traj_len (int): 每条预测轨迹的长度 (即 PREDICTION_HORIZON)。
            traj_feat_dim (int): 每条轨迹在每个时间步的特征维度 (即 FEATURES_PER_STEP)。
            action_dim (int): 动作空间的维度。
            hidden_dim (int): MLP主干网络的隐藏层维度。
            rnn_hidden_dim (int): GRU轨迹编码器的隐藏层维度。
        """
        super().__init__()

        self.av_obs_dim = av_obs_dim
        self.hv_obs_dim = hv_obs_dim
        self.traj_len = traj_len
        self.traj_feat_dim = traj_feat_dim
        
        # --- 1. 轨迹编码器 (Trajectory Encoders) ---
        # 我们使用两个独立的GRU网络来分别编码“让行”和“通行”轨迹
        self.yield_traj_encoder = nn.GRU(input_size=traj_feat_dim, hidden_size=rnn_hidden_dim, batch_first=True)
        self.go_traj_encoder = nn.GRU(input_size=traj_feat_dim, hidden_size=rnn_hidden_dim, batch_first=True)

        # --- 2. 主干网络 (MLP Backbone) ---
        # 主干网络的输入维度 = AV状态 + HV状态 + 两个编码后的轨迹特征
        combined_feature_dim = av_obs_dim + hv_obs_dim + (2 * rnn_hidden_dim)
        
        self.backbone = nn.Sequential(
            nn.Linear(combined_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # --- 3. 策略头 (Actor Head) 和 价值头 (Critic Head) ---
        # Actor和Critic共享主干网络的输出特征
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        self.critic_head = nn.Linear(hidden_dim, 1)

        # --- 4. 动作分布的标准差 ---
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))
        
        # --- 5. 初始化所有权重 ---
        self.apply(init_weights)

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

    def forward(self, observation):
        """
        定义网络的前向传播逻辑。
        """
        # 1. 将输入的扁平观测向量拆分
        av_obs, hv_obs, yield_traj, go_traj = self._split_observation(observation)
        
        # 2. 通过RNN编码器处理轨迹数据
        # GRU的输出是 (output, hidden_state)，我们只需要最后的隐藏状态
        _, yield_embedding = self.yield_traj_encoder(yield_traj)
        _, go_embedding = self.go_traj_encoder(go_traj)
        
        # 调整embedding的形状以进行拼接: (1, batch_size, rnn_hidden_dim) -> (batch_size, rnn_hidden_dim)
        yield_embedding = yield_embedding.squeeze(0)
        go_embedding = go_embedding.squeeze(0)
        
        # 3. 拼接所有特征
        combined_features = torch.cat([av_obs, hv_obs, yield_embedding, go_embedding], dim=1)
        
        # 4. 通过共享的主干网络
        backbone_features = self.backbone(combined_features)
        
        # 5. 计算策略(Actor)和价值(Critic)
        mean = self.actor_head(backbone_features)
        value = self.critic_head(backbone_features)
        
        # 6. 创建动作分布
        std = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        
        return dist, value
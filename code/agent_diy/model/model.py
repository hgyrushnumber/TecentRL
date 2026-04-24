#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Neural network model for Gorge Chase SAC.
峡谷追猎 SAC 神经网络模型。

修改重点：
1. Actor 和 Critic 不再共用同一个 backbone，避免策略更新污染 Q 网络表征。
2. 提供 actor() / critic() / forward() 三类接口。
3. 支持 target_model = deepcopy(model)，用于稳定计算 target Q。
4. 保留 CNN 地图编码 + 标量特征编码结构。
"""

import torch
import torch.nn as nn

from agent_diy.conf.conf import Config


def make_fc_layer(in_features, out_features, gain=1.0):
    """Create a linear layer with orthogonal initialization."""
    fc = nn.Linear(in_features, out_features)
    nn.init.orthogonal_(fc.weight.data, gain=gain)
    nn.init.zeros_(fc.bias.data)
    return fc


class FeatureEncoder(nn.Module):
    """
    Feature encoder:
    - map_flat -> CNN -> map_embed
    - scalar features -> MLP -> scalar_embed
    - concat(map_embed, scalar_embed) -> fused feature
    """

    def __init__(self):
        super().__init__()

        hidden_dim = Config.HIDDEN_DIM
        map_window = Config.LOCAL_MAP_WINDOW
        map_channels = Config.MAP_CHANNELS

        scalar_dim = (
            Config.FEATURES[0]
            + Config.FEATURES[1]
            + Config.FEATURES[2]
            + Config.FEATURES[3]
            + Config.FEATURES[5]
            + Config.FEATURES[6]
            + Config.FEATURES[7]
        )

        self.map_window = map_window
        self.map_channels = map_channels

        self.hero_end = Config.FEATURES[0]
        self.m1_end = self.hero_end + Config.FEATURES[1]
        self.m2_end = self.m1_end + Config.FEATURES[2]
        self.rel_end = self.m2_end + Config.FEATURES[3]
        self.map_end = self.rel_end + Config.FEATURES[4]
        self.legal_end = self.map_end + Config.FEATURES[5]
        self.progress_end = self.legal_end + Config.FEATURES[6]
        self.treasure_dir_end = self.progress_end + Config.FEATURES[7]

        self.map_encoder = nn.Sequential(
            nn.Conv2d(map_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        conv_h = (map_window + 3) // 4
        conv_w = (map_window + 3) // 4
        conv_out = 32 * conv_h * conv_w

        self.map_proj = nn.Sequential(
            make_fc_layer(conv_out, hidden_dim),
            nn.ReLU(),
        )

        self.scalar_encoder = nn.Sequential(
            make_fc_layer(scalar_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, obs):
        hero_feat = obs[:, :self.hero_end]
        m1_feat = obs[:, self.hero_end:self.m1_end]
        m2_feat = obs[:, self.m1_end:self.m2_end]
        rel_monster_feat = obs[:, self.m2_end:self.rel_end]
        map_flat = obs[:, self.rel_end:self.map_end]
        legal_feat = obs[:, self.map_end:self.legal_end]
        progress_feat = obs[:, self.legal_end:self.progress_end]
        treasure_dir_feat = obs[:, self.progress_end:self.treasure_dir_end]

        scalar_feat = torch.cat(
            [
                hero_feat,
                m1_feat,
                m2_feat,
                rel_monster_feat,
                legal_feat,
                progress_feat,
                treasure_dir_feat,
            ],
            dim=1,
        )

        map_tensor = map_flat.view(
            -1,
            self.map_channels,
            self.map_window,
            self.map_window,
        )

        map_embed = self.map_proj(self.map_encoder(map_tensor))
        scalar_embed = self.scalar_encoder(scalar_feat)

        return torch.cat([map_embed, scalar_embed], dim=1)


class Model(nn.Module):
    """
    Discrete SAC model:
    - Actor: obs -> logits(action_num)
    - Critic: obs -> q1(action_num), q2(action_num)

    注意：
    Actor 和 Critic 使用独立 encoder/backbone。
    这样 actor 更新不会直接改变 critic 的特征空间。
    """

    def __init__(self, device=None):
        super().__init__()
        self.model_name = "gorge_chase_sac"

        hidden_dim = Config.HIDDEN_DIM
        mid_dim = Config.MID_DIM
        action_num = Config.ACTION_NUM

        fused_dim = hidden_dim * 2

        # Actor network
        self.actor_encoder = FeatureEncoder()
        self.actor_trunk = nn.Sequential(
            make_fc_layer(fused_dim, hidden_dim),
            nn.ReLU(),
            make_fc_layer(hidden_dim, mid_dim),
            nn.ReLU(),
        )
        # actor 输出层 gain 小一点，避免初始 logits 过大导致策略过早塌缩
        self.actor_head = make_fc_layer(mid_dim, action_num, gain=0.01)

        # Critic network
        self.critic_encoder = FeatureEncoder()
        self.critic_trunk = nn.Sequential(
            make_fc_layer(fused_dim, hidden_dim),
            nn.ReLU(),
            make_fc_layer(hidden_dim, mid_dim),
            nn.ReLU(),
        )
        self.q1_head = make_fc_layer(mid_dim, action_num)
        self.q2_head = make_fc_layer(mid_dim, action_num)

    def actor(self, obs):
        """
        Return action logits.
        """
        feat = self.actor_encoder(obs)
        hidden = self.actor_trunk(feat)
        logits = self.actor_head(hidden)
        return logits

    def critic(self, obs):
        """
        Return twin Q values.
        """
        feat = self.critic_encoder(obs)
        hidden = self.critic_trunk(feat)
        q1 = self.q1_head(hidden)
        q2 = self.q2_head(hidden)
        return q1, q2

    def forward(self, obs, inference=False):
        """
        Compatible with old code:
        return logits, q1, q2
        """
        logits = self.actor(obs)
        q1, q2 = self.critic(obs)
        return logits, q1, q2

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()
#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Neural network model for Gorge Chase SAC.
峡谷追猎 SAC 神经网络模型。
"""

import torch
import torch.nn as nn

from agent_diy.conf.conf import Config


def make_fc_layer(in_features, out_features):
    """Create a linear layer with orthogonal initialization."""
    fc = nn.Linear(in_features, out_features)
    nn.init.orthogonal_(fc.weight.data)
    nn.init.zeros_(fc.bias.data)
    return fc


class Model(nn.Module):
    """Shared backbone + actor head + twin Q heads."""

    def __init__(self, device=None):
        super().__init__()
        self.model_name = "gorge_chase_sac"
        self.device = device

        hidden_dim = Config.HIDDEN_DIM
        mid_dim = Config.MID_DIM
        action_num = Config.ACTION_NUM
        map_window = Config.LOCAL_MAP_WINDOW
        map_channels = Config.MAP_CHANNELS
        scalar_dim = (
            Config.FEATURES[0]
            + Config.FEATURES[1]
            + Config.FEATURES[2]
            + Config.FEATURES[3]
            + Config.FEATURES[5]
            + Config.FEATURES[6]
        )
        map_flat_dim = Config.FEATURES[4]

        self.map_window = map_window
        self.map_channels = map_channels
        self.map_flat_dim = map_flat_dim
        self.scalar_dim = scalar_dim

        # Spatial encoder (CNN) / 空间特征卷积编码器
        self.map_encoder = nn.Sequential(
            nn.Conv2d(map_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        conv_out = 32 * ((map_window + 3) // 4) * ((map_window + 3) // 4)
        self.map_proj = nn.Sequential(make_fc_layer(conv_out, hidden_dim), nn.ReLU())

        # Scalar encoder / 标量特征编码器
        self.scalar_encoder = nn.Sequential(make_fc_layer(scalar_dim, hidden_dim), nn.ReLU())

        # Shared backbone / 融合骨干
        self.backbone = nn.Sequential(make_fc_layer(hidden_dim * 2, hidden_dim), nn.ReLU(), make_fc_layer(hidden_dim, mid_dim), nn.ReLU())

        # Actor head / 策略头
        self.actor_head = make_fc_layer(mid_dim, action_num)

        # Twin critics / 双 Q 网络
        self.q1_head = make_fc_layer(mid_dim, action_num)
        self.q2_head = make_fc_layer(mid_dim, action_num)

    def forward(self, obs, inference=False):
        hero_end = Config.FEATURES[0]
        m1_end = hero_end + Config.FEATURES[1]
        m2_end = m1_end + Config.FEATURES[2]
        rel_end = m2_end + Config.FEATURES[3]
        map_end = rel_end + Config.FEATURES[4]
        legal_end = map_end + Config.FEATURES[5]

        hero_feat = obs[:, :hero_end]
        m1_feat = obs[:, hero_end:m1_end]
        m2_feat = obs[:, m1_end:m2_end]
        rel_monster_feat = obs[:, m2_end:rel_end]
        map_flat = obs[:, rel_end:map_end]
        legal_feat = obs[:, map_end:legal_end]
        progress_feat = obs[:, legal_end:]

        scalar_feat = torch.cat([hero_feat, m1_feat, m2_feat, rel_monster_feat, legal_feat, progress_feat], dim=1)
        map_tensor = map_flat.view(-1, self.map_channels, self.map_window, self.map_window)

        map_embed = self.map_proj(self.map_encoder(map_tensor))
        scalar_embed = self.scalar_encoder(scalar_feat)
        hidden = self.backbone(torch.cat([map_embed, scalar_embed], dim=1))

        logits = self.actor_head(hidden)
        q1 = self.q1_head(hidden)
        q2 = self.q2_head(hidden)
        return logits, q1, q2

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()

#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Neural network model for Gorge Chase SAC.
峡谷追猎 SAC 神经网络模型。

当前结构：
1. Actor 和 Critic 使用独立 encoder/backbone。
2. Local CNN 处理 21x21 局部障碍图。
3. Global CNN 处理 128x128 全局实体热力图。
4. Scalar MLP 处理数值特征。
5. 支持 actor() / critic() / forward() 三类接口。
6. 支持 target_model = deepcopy(model)，用于稳定计算 target Q。
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
    Feature encoder.

    Feature layout:
    [
        hero_feat,          # 4
        monster_feat,       # 12
        local_map_feat,     # MAP_FEATURE_DIM: 1 * 21 * 21
        global_map_feat,    # GLOBAL_MAP_FEATURE_DIM: 4 * 128 * 128
        legal_action,       # 16
        status_feat,        # 6
        anti_stuck_feat,    # 2, [dx10_norm, dz10_norm]
        treasure_dir_feat,  # 3
        buff_dir_feat,      # 3
    ]

    Branches:
    - local obstacle map -> local CNN
    - global entity heatmap -> global CNN
    - scalar features -> MLP
    """

    def __init__(self):
        super().__init__()

        hidden_dim = Config.HIDDEN_DIM

        self.local_map_window = Config.LOCAL_MAP_WINDOW
        self.local_map_channels = Config.MAP_CHANNELS

        self.global_map_size = Config.GLOBAL_MAP_SIZE
        self.global_map_channels = Config.GLOBAL_MAP_CHANNELS

        # ------------------------------------------------------------------
        # Feature split offsets
        # ------------------------------------------------------------------
        self.hero_end = Config.FEATURES[0]
        self.monster_end = self.hero_end + Config.FEATURES[1]
        self.local_map_end = self.monster_end + Config.FEATURES[2]
        self.global_map_end = self.local_map_end + Config.FEATURES[3]
        self.legal_end = self.global_map_end + Config.FEATURES[4]
        self.status_end = self.legal_end + Config.FEATURES[5]
        self.anti_stuck_end = self.status_end + Config.FEATURES[6]
        self.treasure_dir_end = self.anti_stuck_end + Config.FEATURES[7]
        self.buff_dir_end = self.treasure_dir_end + Config.FEATURES[8]

        if self.buff_dir_end != Config.DIM_OF_OBSERVATION:
            raise ValueError(
                f"FEATURES split mismatch: split end {self.buff_dir_end}, "
                f"expected {Config.DIM_OF_OBSERVATION}"
            )

        scalar_dim = (
            Config.DIM_OF_OBSERVATION
            - Config.MAP_FEATURE_DIM
            - Config.GLOBAL_MAP_FEATURE_DIM
        )

        # ------------------------------------------------------------------
        # Local obstacle CNN: [B, 1, 21, 21]
        # ------------------------------------------------------------------
        self.local_map_encoder = nn.Sequential(
            nn.Conv2d(self.local_map_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        local_conv_h = (self.local_map_window + 3) // 4
        local_conv_w = (self.local_map_window + 3) // 4
        local_conv_out = 32 * local_conv_h * local_conv_w

        self.local_map_proj = nn.Sequential(
            make_fc_layer(local_conv_out, hidden_dim),
            nn.ReLU(),
        )

        # ------------------------------------------------------------------
        # Global entity CNN: [B, 4, 128, 128]
        # 快速下采样：128 -> 64 -> 32 -> 16 -> 8
        # ------------------------------------------------------------------
        self.global_map_encoder = nn.Sequential(
            nn.Conv2d(
                self.global_map_channels,
                16,
                kernel_size=5,
                stride=2,
                padding=2,
            ),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Flatten(),
        )

        global_conv_size = self.global_map_size // 16
        global_conv_out = 64 * global_conv_size * global_conv_size

        self.global_map_proj = nn.Sequential(
            make_fc_layer(global_conv_out, hidden_dim),
            nn.ReLU(),
        )

        # ------------------------------------------------------------------
        # Scalar MLP
        # scalar_dim 自动包含：
        # hero + monster + legal + status + anti_stuck + treasure_dir + buff_dir
        # ------------------------------------------------------------------
        self.scalar_encoder = nn.Sequential(
            make_fc_layer(scalar_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, obs):
        if obs.shape[1] != Config.DIM_OF_OBSERVATION:
            raise ValueError(
                f"obs dim mismatch: got {obs.shape[1]}, "
                f"expected {Config.DIM_OF_OBSERVATION}"
            )

        hero_feat = obs[:, :self.hero_end]
        monster_feat = obs[:, self.hero_end:self.monster_end]
        local_map_flat = obs[:, self.monster_end:self.local_map_end]
        global_map_flat = obs[:, self.local_map_end:self.global_map_end]
        legal_feat = obs[:, self.global_map_end:self.legal_end]
        status_feat = obs[:, self.legal_end:self.status_end]
        anti_stuck_feat = obs[:, self.status_end:self.anti_stuck_end]
        treasure_dir_feat = obs[:, self.anti_stuck_end:self.treasure_dir_end]
        buff_dir_feat = obs[:, self.treasure_dir_end:self.buff_dir_end]

        scalar_feat = torch.cat(
            [
                hero_feat,
                monster_feat,
                legal_feat,
                status_feat,
                anti_stuck_feat,
                treasure_dir_feat,
                buff_dir_feat,
            ],
            dim=1,
        )

        local_map_tensor = local_map_flat.view(
            -1,
            self.local_map_channels,
            self.local_map_window,
            self.local_map_window,
        )

        global_map_tensor = global_map_flat.view(
            -1,
            self.global_map_channels,
            self.global_map_size,
            self.global_map_size,
        )

        local_map_embed = self.local_map_proj(
            self.local_map_encoder(local_map_tensor)
        )

        global_map_embed = self.global_map_proj(
            self.global_map_encoder(global_map_tensor)
        )

        scalar_embed = self.scalar_encoder(scalar_feat)

        return torch.cat(
            [
                local_map_embed,
                global_map_embed,
                scalar_embed,
            ],
            dim=1,
        )


class Model(nn.Module):
    """
    Discrete SAC model:
    - Actor: obs -> logits(action_num)
    - Critic: obs -> q1(action_num), q2(action_num)

    Actor 和 Critic 使用独立 encoder/backbone。
    """

    def __init__(self, device=None):
        super().__init__()
        self.model_name = "gorge_chase_sac"

        hidden_dim = Config.HIDDEN_DIM
        mid_dim = Config.MID_DIM
        action_num = Config.ACTION_NUM

        # local_map_embed + global_map_embed + scalar_embed
        fused_dim = hidden_dim * 3

        # ------------------------------------------------------------------
        # Actor network
        # ------------------------------------------------------------------
        self.actor_encoder = FeatureEncoder()
        self.actor_trunk = nn.Sequential(
            make_fc_layer(fused_dim, hidden_dim),
            nn.ReLU(),
            make_fc_layer(hidden_dim, mid_dim),
            nn.ReLU(),
        )

        self.actor_head = make_fc_layer(
            mid_dim,
            action_num,
            gain=float(getattr(Config, "ACTOR_HEAD_INIT_GAIN", 0.01)),
        )

        # ------------------------------------------------------------------
        # Critic network
        # ------------------------------------------------------------------
        self.critic_encoder = FeatureEncoder()
        self.critic_trunk = nn.Sequential(
            make_fc_layer(fused_dim, hidden_dim),
            nn.ReLU(),
            make_fc_layer(hidden_dim, mid_dim),
            nn.ReLU(),
        )

        self.q1_head = make_fc_layer(
            mid_dim,
            action_num,
            gain=float(getattr(Config, "CRITIC_HEAD_INIT_GAIN", 1.0)),
        )
        self.q2_head = make_fc_layer(
            mid_dim,
            action_num,
            gain=float(getattr(Config, "CRITIC_HEAD_INIT_GAIN", 1.0)),
        )

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
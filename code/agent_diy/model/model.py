#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

SAC Networks for Gorge Chase (Discrete Action Space).
峡谷追猎 SAC 网络（离散动作空间，多通道空间分支版）。

网络结构：
  Actor   : 标量分支 + 6x21x21 CNN空间分支 -> 动作概率分布 π(a|s)
  Critic  : 标量分支 + 6x21x21 CNN空间分支 -> 双Q网络 Q1/Q2

输入维度：
  标量特征:
    hero(6)
    monster1(11)
    monster2(11)
    treasure(6)
    buff(5)
    legal_action(16)
    planning(10)
    => 65
  空间特征:
    6 x 21 x 21 = 2646
  总维度:
    2711
"""

import torch
import torch.nn as nn


from agent_diy.conf.conf import Config

# Spatial shape
SPATIAL_CHANNELS = 6
LOCAL_MAP_SIZE = 21
SPATIAL_DIM = SPATIAL_CHANNELS * LOCAL_MAP_SIZE * LOCAL_MAP_SIZE  # 2646

# Scalar dims
SCALAR_FRONT_DIM = 39   # hero(6)+monster1(11)+monster2(11)+treasure(6)+buff(5)
SCALAR_BACK_DIM = 26    # legal_action(16)+planning(10)
SCALAR_TOTAL_DIM = SCALAR_FRONT_DIM + SCALAR_BACK_DIM  # 65


def _make_fc(in_f, out_f, gain=1.0):
    """Linear layer with orthogonal init."""
    fc = nn.Linear(in_f, out_f)
    nn.init.orthogonal_(fc.weight, gain=gain)
    nn.init.zeros_(fc.bias)
    return fc


def _make_conv(in_c, out_c, kernel_size=3, stride=1, padding=1, gain=1.0):
    """Conv2d with orthogonal init."""
    conv = nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding)
    nn.init.orthogonal_(conv.weight, gain=gain)
    nn.init.zeros_(conv.bias)
    return conv


class ScalarEncoder(nn.Module):
    """
    标量特征编码器
    输入: [B, 65]
    输出: [B, scalar_out_dim]
    """
    def __init__(self, input_dim=SCALAR_TOTAL_DIM, hidden_dim=256, out_dim=128, dropout_p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            _make_fc(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout_p),

            _make_fc(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout_p),

            _make_fc(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ELU(),
        )

    def forward(self, x):
        return self.net(x)


class SpatialEncoder(nn.Module):
    """
    多通道空间编码器
    输入: [B, 6, 21, 21]
    输出: [B, spatial_out_dim]
    """
    def __init__(self, in_channels=SPATIAL_CHANNELS, spatial_out_dim=128, dropout_p=0.1):
        super().__init__()

        self.conv = nn.Sequential(
            _make_conv(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(4, 16),
            nn.ELU(),

            _make_conv(16, 32, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 32),
            nn.ELU(),

            nn.MaxPool2d(kernel_size=2, stride=2),  # 21 -> 10

            _make_conv(32, 64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 64),
            nn.ELU(),

            _make_conv(64, 64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 64),
            nn.ELU(),

            nn.MaxPool2d(kernel_size=2, stride=2),  # 10 -> 5
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            _make_fc(64 * 5 * 5, 256),
            nn.LayerNorm(256),
            nn.ELU(),
            nn.Dropout(dropout_p),

            _make_fc(256, spatial_out_dim),
            nn.LayerNorm(spatial_out_dim),
            nn.ELU(),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


class FusionEncoder(nn.Module):
    """
    将 obs 拆成:
      - 标量特征 [B, 65]
      - 空间特征 [B, 6, 21, 21]
    再融合成统一表示 [B, fused_dim]
    """
    def __init__(self, scalar_hidden=256, scalar_out=128, spatial_out=128, fused_dim=256, dropout_p=0.1):
        super().__init__()
        self.scalar_encoder = ScalarEncoder(
            input_dim=SCALAR_TOTAL_DIM,
            hidden_dim=scalar_hidden,
            out_dim=scalar_out,
            dropout_p=dropout_p,
        )
        self.spatial_encoder = SpatialEncoder(
            in_channels=SPATIAL_CHANNELS,
            spatial_out_dim=spatial_out,
            dropout_p=dropout_p,
        )

        self.fusion = nn.Sequential(
            _make_fc(scalar_out + spatial_out, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.ELU(),
            nn.Dropout(dropout_p),

            _make_fc(fused_dim, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.ELU(),
        )

    @staticmethod
    def split_obs(obs):
        """
        obs 结构:
          front_scalar(39)
          spatial_flat(2646)
          back_scalar(26)
        总计:
          39 + 2646 + 26 = 2711
        """
        front = obs[:, :SCALAR_FRONT_DIM]
        spatial_flat = obs[:, SCALAR_FRONT_DIM: SCALAR_FRONT_DIM + SPATIAL_DIM]
        back = obs[:, SCALAR_FRONT_DIM + SPATIAL_DIM:]

        scalar_feat = torch.cat([front, back], dim=-1)          # [B, 65]
        spatial_feat = spatial_flat.view(-1, SPATIAL_CHANNELS, LOCAL_MAP_SIZE, LOCAL_MAP_SIZE)

        return scalar_feat, spatial_feat

    def forward(self, obs):
        scalar_feat, spatial_feat = self.split_obs(obs)
        scalar_emb = self.scalar_encoder(scalar_feat)
        spatial_emb = self.spatial_encoder(spatial_feat)

        fused = torch.cat([scalar_emb, spatial_emb], dim=-1)
        fused = self.fusion(fused)
        return fused


class Actor(nn.Module):
    """SAC Actor: outputs action probability distribution π(a|s)."""

    def __init__(self, input_dim, hidden_dim, mid_dim, action_num):
        super().__init__()

        self.encoder = FusionEncoder(
            scalar_hidden=256,
            scalar_out=128,
            spatial_out=128,
            fused_dim=256,
            dropout_p=0.1,
        )

        self.head = nn.Sequential(
            _make_fc(256, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),

            _make_fc(hidden_dim, action_num, gain=0.01),
        )

    def forward(self, obs, legal_action=None):
        """
        Return masked action probability distribution.
        obs: [B, 2711]
        legal_action: [B, action_num]
        """
        z = self.encoder(obs)
        logits = self.head(z)

        if legal_action is not None:
            logits = logits + (1.0 - legal_action) * (-1e9)

        probs = torch.softmax(logits, dim=-1)
        return probs

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()


class Critic(nn.Module):
    """SAC Critic: dual Q-networks Q1, Q2."""

    def __init__(self, input_dim, hidden_dim, mid_dim, action_num):
        super().__init__()

        self.encoder1 = FusionEncoder(
            scalar_hidden=256,
            scalar_out=128,
            spatial_out=128,
            fused_dim=256,
            dropout_p=0.1,
        )
        self.encoder2 = FusionEncoder(
            scalar_hidden=256,
            scalar_out=128,
            spatial_out=128,
            fused_dim=256,
            dropout_p=0.1,
        )

        self.head1 = nn.Sequential(
            _make_fc(256, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),

            _make_fc(hidden_dim, action_num),
        )

        self.head2 = nn.Sequential(
            _make_fc(256, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),

            _make_fc(hidden_dim, action_num),
        )

    def forward(self, obs, legal_action=None):
        """
        Return Q1(s,·) and Q2(s,·) for all actions.
        obs: [B, 2711]
        legal_action: [B, action_num]
        """
        z1 = self.encoder1(obs)
        q1 = self.head1(z1)

        z2 = self.encoder2(obs)
        q2 = self.head2(z2)

        if legal_action is not None:
            mask = (1.0 - legal_action) * (-1e9)
            q1 = q1 + mask
            q2 = q2 + mask

        return q1, q2

    def q1(self, obs, legal_action=None):
        z1 = self.encoder1(obs)
        q1 = self.head1(z1)
        if legal_action is not None:
            q1 = q1 + (1.0 - legal_action) * (-1e9)
        return q1

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()


# ── 兼容旧接口：Model = Actor ────────────────────────────────────────
Model = Actor
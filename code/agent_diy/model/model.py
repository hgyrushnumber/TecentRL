#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

SAC Networks for Gorge Chase (Discrete Action Space).
峡谷追猎 SAC 网络（离散动作空间）。

网络结构：
  Actor   : obs → 共享骨干 → 动作概率分布 π(a|s)
  Critic  : obs → 双Q网络 Q1(s), Q2(s) → 每个动作的Q值向量
  (双Q网络用于减少过估计偏差，取 min(Q1, Q2) 计算目标值)
"""

import torch
import torch.nn as nn
import numpy as np

from agent_diy.conf.conf import Config


def _make_fc(in_f, out_f, gain=1.0):
    """Linear layer with orthogonal init. / 正交初始化线性层。"""
    fc = nn.Linear(in_f, out_f)
    nn.init.orthogonal_(fc.weight, gain=gain)
    nn.init.zeros_(fc.bias)
    return fc


def _build_mlp(input_dim, hidden_dim, mid_dim):
    """Shared MLP backbone with residual connection.

    共享MLP骨干网络（含残差连接，激活函数改为ELU，梯度更平滑）。
    """
    backbone = nn.Sequential(
        _make_fc(input_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),  # 替换LayerNorm为BatchNorm1d
        nn.ELU(),
        nn.Dropout(0.2),  # 添加Dropout防止过拟合
        _make_fc(hidden_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ELU(),
        nn.Dropout(0.2),
        _make_fc(hidden_dim, mid_dim),
        nn.BatchNorm1d(mid_dim),
        nn.ELU(),
        nn.Dropout(0.2),
    )
    skip = _make_fc(input_dim, mid_dim)
    return backbone, skip


class Actor(nn.Module):
    """SAC Actor: outputs action probability distribution π(a|s).

    SAC Actor 网络：输出合法动作上的概率分布。
    """

    def __init__(self, input_dim, hidden_dim, mid_dim, action_num):
        super().__init__()
        self.backbone, self.skip = _build_mlp(input_dim, hidden_dim, mid_dim)
        self.head = nn.Sequential(
            _make_fc(mid_dim, mid_dim),
            nn.LayerNorm(mid_dim),
            nn.ELU(),
            _make_fc(mid_dim, action_num, gain=0.01),  # 小增益稳定初始输出
        )

    def forward(self, obs, legal_action=None):
        """Return action logits and masked probability distribution."""
        h = self.backbone(obs) + self.skip(obs)
        logits = self.head(h)
        if legal_action is not None:
            # 非法动作掩码：-1e9 使其概率趋近0
            logits = logits + (1.0 - legal_action) * (-1e9)
        probs = torch.softmax(logits, dim=-1)
        return probs

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()


class Critic(nn.Module):
    """SAC Critic: dual Q-networks Q1, Q2 outputting Q(s, a) for all actions.

    双Q网络：同时输出所有动作的Q值，取 min 减少过估计。
    传入 legal_action 掩码，对非法动作 Q 值置为极小值，消除过估计偏差。
    """

    def __init__(self, input_dim, hidden_dim, mid_dim, action_num):
        super().__init__()
        # Q1 网络
        self.backbone1, self.skip1 = _build_mlp(input_dim, hidden_dim, mid_dim)
        self.head1 = nn.Sequential(
            _make_fc(mid_dim, mid_dim),
            nn.LayerNorm(mid_dim),
            nn.ELU(),
            _make_fc(mid_dim, action_num),
        )
        # Q2 网络
        self.backbone2, self.skip2 = _build_mlp(input_dim, hidden_dim, mid_dim)
        self.head2 = nn.Sequential(
            _make_fc(mid_dim, mid_dim),
            nn.LayerNorm(mid_dim),
            nn.ELU(),
            _make_fc(mid_dim, action_num),
        )

    def forward(self, obs, legal_action=None):
        """Return Q1(s,·) and Q2(s,·) for all actions.

        若传入 legal_action，将非法动作 Q 值替换为 -1e9，消除过估计偏差。
        """
        h1 = self.backbone1(obs) + self.skip1(obs)
        q1 = self.head1(h1)
        h2 = self.backbone2(obs) + self.skip2(obs)
        q2 = self.head2(h2)
        if legal_action is not None:
            mask = (1.0 - legal_action) * (-1e9)
            q1 = q1 + mask
            q2 = q2 + mask
        return q1, q2

    def q1(self, obs, legal_action=None):
        """Return only Q1 (used in actor update)."""
        h1 = self.backbone1(obs) + self.skip1(obs)
        q1 = self.head1(h1)
        if legal_action is not None:
            q1 = q1 + (1.0 - legal_action) * (-1e9)
        return q1

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()


# ── 兼容旧接口：Model = Actor（workflow 中仍用 agent.model 做推理）────────
Model = Actor
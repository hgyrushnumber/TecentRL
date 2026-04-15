#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Enhanced neural network model for Gorge Chase DIY Agent.
峡谷追猎 DIY Agent 增强神经网络模型。

网络改进：
  1. 更宽的隐层（256→128 vs PPO 的 128→64）
  2. LayerNorm 归一化稳定训练
  3. 残差连接防止梯度消失
  4. Actor/Critic 使用独立的第二层（减少表示冲突）
  5. 正交初始化
"""

import torch
import torch.nn as nn
import numpy as np

from agent_diy.conf.conf import Config


def make_fc(in_dim, out_dim, gain=1.0):
    """Linear layer with orthogonal init. / 正交初始化线性层"""
    fc = nn.Linear(in_dim, out_dim)
    nn.init.orthogonal_(fc.weight, gain=gain)
    nn.init.zeros_(fc.bias)
    return fc


class ResidualBlock(nn.Module):
    """Two-layer residual block with LayerNorm.

    带 LayerNorm 的双层残差块。
    """

    def __init__(self, dim):
        super().__init__()
        self.fc1 = make_fc(dim, dim)
        self.fc2 = make_fc(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.act = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.act(self.norm1(self.fc1(x)))
        out = self.norm2(self.fc2(out))
        return self.act(out + residual)


class Model(nn.Module):
    """Enhanced Actor-Critic model with residual backbone.

    增强版 Actor-Critic 模型，带残差骨干网络。

    架构：
      Input (60D)
        → FC(256) + ReLU + LayerNorm
        → ResidualBlock(256)
        → FC(128) + ReLU + LayerNorm
        ↙               ↘
      Actor branch      Critic branch
      FC(128) → ReLU    FC(128) → ReLU
      FC(8)             FC(1)
    """

    def __init__(self, device=None):
        super().__init__()
        self.model_name = "gorge_chase_diy_enhanced"
        self.device = device

        input_dim = Config.DIM_OF_OBSERVATION   # 60
        hidden_dim = 256
        mid_dim = 128
        action_num = Config.ACTION_NUM           # 8
        value_num = Config.VALUE_NUM             # 1

        # Shared backbone / 共享骨干
        self.stem = nn.Sequential(
            make_fc(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

        self.res_block = ResidualBlock(hidden_dim)

        self.trunk = nn.Sequential(
            make_fc(hidden_dim, mid_dim),
            nn.ReLU(),
            nn.LayerNorm(mid_dim),
        )

        # Actor head (独立分支) / 策略头
        self.actor_branch = nn.Sequential(
            make_fc(mid_dim, mid_dim),
            nn.ReLU(),
        )
        self.actor_out = make_fc(mid_dim, action_num, gain=0.01)  # 小增益使初始策略接近均匀

        # Critic head (独立分支) / 价值头
        self.critic_branch = nn.Sequential(
            make_fc(mid_dim, mid_dim),
            nn.ReLU(),
        )
        self.critic_out = make_fc(mid_dim, value_num, gain=1.0)

    def forward(self, obs, inference=False):
        h = self.stem(obs)
        h = self.res_block(h)
        h = self.trunk(h)

        logits = self.actor_out(self.actor_branch(h))
        value = self.critic_out(self.critic_branch(h))
        return logits, value

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()
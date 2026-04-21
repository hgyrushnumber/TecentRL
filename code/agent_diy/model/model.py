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

        input_dim = Config.DIM_OF_OBSERVATION
        hidden_dim = Config.HIDDEN_DIM
        mid_dim = Config.MID_DIM
        action_num = Config.ACTION_NUM

        # Shared backbone / 共享骨干
        self.backbone = nn.Sequential(
            make_fc_layer(input_dim, hidden_dim),
            nn.ReLU(),
            make_fc_layer(hidden_dim, mid_dim),
            nn.ReLU(),
        )

        # Actor head / 策略头
        self.actor_head = make_fc_layer(mid_dim, action_num)

        # Twin critics / 双 Q 网络
        self.q1_head = make_fc_layer(mid_dim, action_num)
        self.q2_head = make_fc_layer(mid_dim, action_num)

    def forward(self, obs, inference=False):
        hidden = self.backbone(obs)
        logits = self.actor_head(hidden)
        q1 = self.q1_head(hidden)
        q2 = self.q2_head(hidden)
        return logits, q1, q2

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()

#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Neural network model for Gorge Chase PPO.
峡谷追猎 PPO 神经网络模型。
"""

import torch
import torch.nn as nn
import numpy as np

from agent_diy.conf.conf import Config


def make_fc_layer(in_features, out_features):
    """Create a linear layer with orthogonal initialization.

    创建正交初始化的线性层。
    """
    fc = nn.Linear(in_features, out_features)
    nn.init.orthogonal_(fc.weight.data)
    nn.init.zeros_(fc.bias.data)
    return fc


class Model(nn.Module):
    """Enhanced MLP backbone with residual connections + Actor/Critic dual heads.

    增强MLP骨干网络（含残差连接）+ Actor/Critic 双头。
    """

    def __init__(self, device=None):
        super().__init__()
        self.model_name = "gorge_chase_enhanced"
        self.device = device

        input_dim = Config.DIM_OF_OBSERVATION  # 现在为50维
        hidden_dim = 256  # 增加隐藏层大小以适应更多特征
        mid_dim = 128
        action_num = Config.ACTION_NUM
        value_num = Config.VALUE_NUM

        # Enhanced shared backbone with residual connections / 增强共享骨干网络（含残差连接）
        self.backbone = nn.Sequential(
            make_fc_layer(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # 添加层归一化
            nn.ReLU(),
            make_fc_layer(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            make_fc_layer(hidden_dim, mid_dim),
            nn.LayerNorm(mid_dim),
            nn.ReLU(),
        )
        
        # Skip connection / 跳跃连接
        self.skip_connection = make_fc_layer(input_dim, mid_dim)

        # Enhanced Actor head / 增强策略头
        self.actor_hidden = make_fc_layer(mid_dim, mid_dim)
        self.actor_norm = nn.LayerNorm(mid_dim)
        self.actor_head = make_fc_layer(mid_dim, action_num)

        # Enhanced Critic head / 增强价值头
        self.critic_hidden = make_fc_layer(mid_dim, mid_dim)
        self.critic_norm = nn.LayerNorm(mid_dim)
        self.critic_head = make_fc_layer(mid_dim, value_num)

    def forward(self, obs, inference=False):
        # Backbone with residual connection / 骨干网络（含残差连接）
        backbone_out = self.backbone(obs)
        skip_out = self.skip_connection(obs)
        hidden = backbone_out + skip_out  # Residual connection
        
        # Actor head with enhanced structure / 增强策略头
        actor_hidden = self.actor_hidden(hidden)
        actor_hidden = self.actor_norm(actor_hidden)
        actor_hidden = nn.functional.relu(actor_hidden)
        logits = self.actor_head(actor_hidden)
        
        # Critic head with enhanced structure / 增强价值头
        critic_hidden = self.critic_hidden(hidden)
        critic_hidden = self.critic_norm(critic_hidden)
        critic_hidden = nn.functional.relu(critic_hidden)
        value = self.critic_head(critic_hidden)
        
        return logits, value

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()

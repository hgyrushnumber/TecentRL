#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Data definitions, GAE computation for Gorge Chase DIY Agent.
峡谷追猎 DIY Agent 数据类定义与 GAE 计算。
"""

import numpy as np
from common_python.utils.common_func import create_cls
from agent_diy.conf.conf import Config


# ObsData: 60D feature vector + 8D legal action mask
ObsData = create_cls("ObsData", feature=None, legal_action=None)

# ActData: stochastic action, greedy action, action probs, value estimate
ActData = create_cls("ActData", action=None, d_action=None, prob=None, value=None)

# SampleData: single-frame training sample
# 必须使用整数定义维度（不能用None），框架层会自动生成FIELD_DIMS并处理序列化
SampleData = create_cls(
    "SampleData",
    obs=Config.DIM_OF_OBSERVATION,       # 60D 观测
    legal_action=Config.ACTION_NUM,      # 8D 合法动作掩码
    act=1,                               # 1D 动作
    reward=Config.VALUE_NUM,             # 1D 即时奖励
    reward_sum=Config.VALUE_NUM,         # 1D 回报（GAE目标）
    done=1,                              # 1D 终止标志
    value=Config.VALUE_NUM,             # 1D 价值估计
    next_value=Config.VALUE_NUM,        # 1D 下一步价值
    advantage=Config.VALUE_NUM,         # 1D GAE优势
    prob=Config.ACTION_NUM,             # 8D 旧策略概率
)


def sample_process(list_sample_data):
    """Fill next_value and compute GAE advantage.

    填充 next_value 并使用 GAE 计算优势函数。
    """
    # 填充 next_value
    for i in range(len(list_sample_data) - 1):
        list_sample_data[i].next_value = list_sample_data[i + 1].value

    _calc_gae(list_sample_data)
    return list_sample_data


def _calc_gae(list_sample_data):
    """Compute GAE (Generalized Advantage Estimation).

    计算广义优势估计（GAE）。
    δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
    A_t = δ_t + (γλ) * δ_{t+1} + (γλ)^2 * δ_{t+2} + ...
    """
    gae = 0.0
    gamma = Config.GAMMA
    lamda = Config.LAMDA
    for sample in reversed(list_sample_data):
        delta = -sample.value + sample.reward + gamma * sample.next_value
        gae = gae * gamma * lamda + delta
        sample.advantage = gae
        # reward_sum 作为价值目标（GAE + baseline）
        sample.reward_sum = gae + sample.value


def reward_shaping(frame_no, score, terminated, truncated, remain_info, _remain_info, obs, _obs):
    """Placeholder kept for workflow compatibility.

    保留接口兼容性，实际奖励在 preprocessor 中计算。
    """
    pass
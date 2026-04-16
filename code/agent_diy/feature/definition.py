#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Data definitions and ReplayBuffer for Gorge Chase SAC.
峡谷追猎 SAC 数据类定义与经验回放池。
"""

import numpy as np
from collections import deque
from common_python.utils.common_func import create_cls
from agent_diy.conf.conf import Config


# ObsData: 特征向量与合法动作掩码（接口不变）
ObsData = create_cls("ObsData", feature=None, legal_action=None)

# ActData: 动作、贪心动作、概率（SAC不需要value字段，保留兼容）
ActData = create_cls("ActData", action=None, d_action=None, prob=None, value=None)

# SampleData: SAC单步转换 (s, a, r, s', done, legal_action, next_legal_action)
# 保持字段名与原PPO一致，便于workflow复用
SampleData = create_cls(
    "SampleData",
    obs=Config.DIM_OF_OBSERVATION,          # 当前状态
    legal_action=Config.ACTION_NUM,          # 当前合法动作掩码
    act=1,                                   # 执行的动作
    reward=Config.VALUE_NUM,                 # 即时奖励
    next_obs=Config.DIM_OF_OBSERVATION,      # 下一状态
    next_legal_action=Config.ACTION_NUM,     # 下一状态合法动作掩码
    done=1,                                  # 终止标志
    # 以下字段SAC不使用，置0保留兼容性
    reward_sum=Config.VALUE_NUM,
    value=Config.VALUE_NUM,
    next_value=Config.VALUE_NUM,
    advantage=Config.VALUE_NUM,
    prob=Config.ACTION_NUM,
)


class ReplayBuffer:
    """Circular replay buffer for SAC off-policy training.

    SAC 离策略训练经验回放池（环形缓冲区）。
    """

    def __init__(self, capacity=None):
        self.capacity = capacity or Config.REPLAY_BUFFER_SIZE
        self._buf = deque(maxlen=self.capacity)

    def push(self, transition):
        """Add a SampleData transition to buffer. / 添加一条转换样本。"""
        self._buf.append(transition)

    def push_episode(self, episode_list):
        """Add all transitions from one episode. / 添加一局的所有转换。"""
        for t in episode_list:
            self._buf.append(t)

    def sample(self, batch_size=None):
        """Randomly sample a batch. / 随机采样一个批次。"""
        batch_size = batch_size or Config.BATCH_SIZE
        indices = np.random.randint(0, len(self._buf), size=batch_size)
        batch = [self._buf[i] for i in indices]
        return batch

    def __len__(self):
        return len(self._buf)

    @property
    def ready(self):
        """True when buffer has enough samples to start training."""
        return len(self._buf) >= Config.LEARNING_STARTS


def sample_process(list_sample_data):
    """Compatibility shim: SAC does not need GAE.

    兼容接口：SAC 不需要 GAE，直接返回原列表。
    """
    return list_sample_data
#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Data definitions and ReplayBuffer for Gorge Chase SAC.
峡谷追猎 SAC 数据类定义与经验回放池。

分布式架构说明：
  - SampleData 由 Actor 进程构造，经框架序列化发送给 Learner
  - ReplayBuffer 由 Algorithm（Learner侧）内置维护
  - workflow 只负责收集数据并 yield，不做任何训练
"""

import numpy as np
from collections import deque
from common_python.utils.common_func import create_cls
from agent_diy.conf.conf import Config


# ObsData / ActData 保持不变
ObsData = create_cls("ObsData", feature=None, legal_action=None)
ActData = create_cls("ActData", action=None, d_action=None, prob=None, value=None)

# SampleData：在标准字段基础上增加 SAC 所需的 next_obs / next_legal_action
# 框架通过字段维度做序列化，int 表示向量维度，None 表示标量
SampleData = create_cls(
    "SampleData",
    obs=Config.DIM_OF_OBSERVATION,        # 当前状态
    legal_action=Config.ACTION_NUM,        # 当前合法动作掩码
    act=1,                                 # 执行的动作
    reward=Config.VALUE_NUM,               # 即时奖励
    done=1,                                # 终止标志
    next_obs=Config.DIM_OF_OBSERVATION,    # 下一状态（SAC 所需）
    next_legal_action=Config.ACTION_NUM,   # 下一状态合法掩码（SAC 所需）
    # 以下字段框架/PPO会用到，SAC置0保持兼容
    reward_sum=Config.VALUE_NUM,
    value=Config.VALUE_NUM,
    next_value=Config.VALUE_NUM,
    advantage=Config.VALUE_NUM,
    prob=Config.ACTION_NUM,
)


class ReplayBuffer:
    """Circular replay buffer for SAC Learner side.

    SAC 经验回放池，由 Algorithm（Learner侧）内置维护。
    Actor 每局产生的 SampleData 列表通过 learn() 批量推入。
    """

    def __init__(self, capacity):
        self._buf = deque(maxlen=capacity)

    def push_batch(self, transitions):
        """Push a list of SampleData from one episode."""
        for t in transitions:
            self._buf.append(t)

    def sample(self, batch_size):
        """Randomly sample a batch."""
        idx = np.random.randint(0, len(self._buf), size=batch_size)
        return [self._buf[i] for i in idx]

    def __len__(self):
        return len(self._buf)


def sample_process(list_sample_data):
    """SAC 不需要 GAE，直接返回原列表（兼容框架调用）。"""
    return list_sample_data
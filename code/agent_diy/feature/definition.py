#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Data definitions for Gorge Chase SAC.
峡谷追猎 SAC 数据定义。
"""

from common_python.utils.common_func import create_cls
from agent_ppo.conf.conf import Config


# ObsData: feature vector + legal action mask
ObsData = create_cls("ObsData", feature=None, legal_action=None)

# ActData: only action outputs needed by workflow
ActData = create_cls("ActData", action=None, d_action=None)

# SampleData: pure SAC transition
SampleData = create_cls(
    "SampleData",
    obs=Config.DIM_OF_OBSERVATION,
    legal_action=Config.ACTION_NUM,
    act=1,
    reward=Config.VALUE_NUM,
    done=1,
    next_obs=Config.DIM_OF_OBSERVATION,
    next_legal_action=Config.ACTION_NUM,
)


def sample_process(list_sample_data):
    """
    SAC 下这里不再做 GAE，也不再补 PPO 字段。
    直接原样返回。
    """
    return list_sample_data
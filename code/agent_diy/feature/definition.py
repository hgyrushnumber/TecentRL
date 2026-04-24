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
from agent_diy.conf.conf import Config


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
    reward=1,
    done=1,
    next_obs=Config.DIM_OF_OBSERVATION,
    next_legal_action=Config.ACTION_NUM,
)


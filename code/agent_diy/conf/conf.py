#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Configuration for Gorge Chase PPO.
峡谷追猎 PPO 配置。
"""


class Config:

    # Feature dimensions / 特征维度（共59维 - 增加了技能动作）
    FEATURES = [
        4,    # 英雄自身特征
        9,    # 怪物1特征（增加了4个高级特征）
        9,    # 怪物2特征（增加了4个高级特征）
        4,    # 宝箱特征（新增）
        3,    # Buff特征（新增）
        16,   # 局部地图特征
        10,   # 合法动作掩码（增加了2个技能动作）
        4,    # 进度特征（增加了2个高级特征）
    ]
    FEATURE_SPLIT_SHAPE = FEATURES
    FEATURE_LEN = sum(FEATURE_SPLIT_SHAPE)
    DIM_OF_OBSERVATION = FEATURE_LEN

    # Action space / 动作空间：8个移动方向 + 2个技能
    # [0-7]: 移动方向（上、下、左、右、左上、右上、左下、右下）
    # [8]: 使用闪现技能
    # [9]: 使用天赋技能
    ACTION_NUM = 10

    # Value head / 价值头：单头生存奖励
    VALUE_NUM = 1

    # PPO hyperparameters / PPO 超参数
    GAMMA = 0.99
    LAMDA = 0.95
    INIT_LEARNING_RATE_START = 0.0003
    BETA_START = 0.001
    CLIP_PARAM = 0.2
    VF_COEF = 1.0
    GRAD_CLIP_RANGE = 0.5

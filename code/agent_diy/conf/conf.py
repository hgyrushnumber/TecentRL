#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Configuration for Gorge Chase SAC-compatible training.
峡谷追猎 SAC 兼容配置。
"""


class Config:
    # Spatial encoder settings / 空间编码设置
    LOCAL_MAP_WINDOW = 21
    MAP_CHANNELS = 4  # hero, monster, treasure, obstacle
    MAP_FEATURE_DIM = LOCAL_MAP_WINDOW * LOCAL_MAP_WINDOW * MAP_CHANNELS

    # Feature dimensions / 特征维度（共1805维）
    FEATURES = [
        4,
        5,
        5,
        6,  # out-of-vision monster relative info (2 monsters x [dx, dz, dist])
        MAP_FEATURE_DIM,
        16,
        2,
        3,  # nearest treasure direction + distance (dx, dz, dist_norm)
    ]
    FEATURE_SPLIT_SHAPE = FEATURES
    FEATURE_LEN = sum(FEATURE_SPLIT_SHAPE)
    DIM_OF_OBSERVATION = FEATURE_LEN

    # Action space / 动作空间：16个动作（8移动+8闪现）
    ACTION_NUM = 16

    # 保留为 1，兼容 workflow 与旧字段
    VALUE_NUM = 1

    # Network
    HIDDEN_DIM = 128
    MID_DIM = 64

    # SAC hyperparameters
    GAMMA = 0.99
    TAU = 0.005
    ALPHA = 0.2
    AUTO_ALPHA = True
    TARGET_ENTROPY = 2.0
    ALPHA_LR = 1e-4
    ALPHA_MIN = 1e-3
    ALPHA_MAX = 10.0
    Q_MIX_COEF = 0.1

    # Optimizer
    INIT_LEARNING_RATE_START = 2e-4
    GRAD_CLIP_RANGE = 5.0

    # ---- Legacy PPO fields kept only for compatibility ----
    LAMDA = 0.95
    BETA_START = 0.001
    CLIP_PARAM = 0.2
    VF_COEF = 1.0

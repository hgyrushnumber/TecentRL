#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Configuration for Gorge Chase SAC.
峡谷追猎 SAC（离散动作）配置。
"""


class Config:

    # ── Feature dimensions / 特征维度 ────────────────────────────────
    # hero(6)
    # monster1(11)
    # monster2(11)
    # treasure(6)
    # buff(5)
    # spatial(6 * 21 * 21 = 2646)
    # legal_action(16)
    # planning(10)
    FEATURES = [
        6,
        11,
        11,
        6,
        5,
        2646,
        16,
        10,
    ]
    FEATURE_SPLIT_SHAPE = FEATURES
    FEATURE_LEN = sum(FEATURE_SPLIT_SHAPE)   # 2711
    DIM_OF_OBSERVATION = FEATURE_LEN

    # Action space
    ACTION_NUM = 16
    VALUE_NUM = 1

    # ── SAC 超参数 ─────────────────────────────────────────────────
    GAMMA = 0.99
    INIT_LEARNING_RATE_START = 3e-5
    INIT_LEARNING_RATE_END = 1e-5
    LR_DECAY_STEPS = 2_000_000
    GRAD_CLIP_RANGE = 1.0

    # 自动熵调整 (auto-alpha tuning)
    AUTO_ALPHA = True
    ALPHA_LR = 3e-4
    TARGET_ENTROPY_RATIO = 0.9
    # 实际 target_entropy 在 algorithm.py 中写成：
    # -TARGET_ENTROPY_RATIO * log(|A|)

    # Soft target update
    TAU = 0.005

    # Replay buffer
    REPLAY_BUFFER_SIZE = 200_000
    BATCH_SIZE = 256
    LEARNING_STARTS = 15_000
    UPDATES_PER_LEARN = 4

    # PER
    PER_ALPHA = 0.4
    PER_BETA_START = 0.4
    PER_BETA_FRAMES = 150_000

    # AMP
    USE_AMP = True

    # α 约束范围
    ALPHA_MIN = 0.05
    ALPHA_MAX = 1.0

    # ── 兼容性保留 ────────────────────────────────────────────────
    LAMDA = 0.95
    BETA_START = 0.01
    CLIP_PARAM = 0.2
    VF_COEF = 0.5
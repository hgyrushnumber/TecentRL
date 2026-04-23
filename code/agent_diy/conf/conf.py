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
    MAP_CHANNELS = 1  # obstacle only
    MAP_FEATURE_DIM = LOCAL_MAP_WINDOW * LOCAL_MAP_WINDOW * MAP_CHANNELS

    # Feature dimensions / 特征维度（共482维）
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
    ACTOR_UPDATE_INTERVAL = 2
    TARGET_Q_CLIP = 8.0
    CRITIC_USE_HUBER = True

    # Reward knobs (can be tuned directly in conf)
    REWARD_SURVIVE = 0.01
    REWARD_STEP_SCORE = 0.05
    REWARD_DIST_SHAPING = 0.08
    REWARD_PROGRESSIVE_STEP = 0.02
    REWARD_STAGE_PROGRESS = 0.04
    MILESTONE_STEP = 20
    REWARD_TREASURE_SCORE = 0.6
    REWARD_TREASURE_APPROACH = 0.12
    PENALTY_TREASURE_AWAY = 0.03
    TREASURE_SAFE_DISTANCE_TH = 0.35
    REWARD_TREASURE_SAFE_BONUS_CAP = 0.04
    REWARD_TREASURE_SAFE_BONUS_COEF = 0.16

    REWARD_BUFF = 0.2
    PENALTY_DANGER = 0.08
    PENALTY_SECOND_MONSTER = 0.04
    PENALTY_CORNER = 0.04
    PENALTY_ENCIRCLE = 0.03
    PENALTY_INVALID_MOVE = 0.03
    PENALTY_REPEAT_VISIT = 0.02
    PENALTY_REPEAT_UNIQUE = 0.02
    REWARD_CORRIDOR = 0.05
    REWARD_NEAR_SPEEDUP = 0.03
    REWARD_LATE_SURVIVAL = 0.05

    POST_FLASH_WINDOW = 8
    REWARD_POST_FLASH_MOVE = 0.02
    PENALTY_POST_FLASH_IDLE = 0.04
    PENALTY_POST_FLASH_SAFE_IDLE = 0.03
    THRESH_POST_FLASH_MOVE = 0.35
    THRESH_POST_FLASH_IDLE = 0.2
    THRESH_POST_FLASH_SAFE_IDLE = 0.25
    REWARD_POST_FLASH_TREASURE_CAP = 0.03
    REWARD_POST_FLASH_TREASURE_COEF = 0.12
    FLASH_ESCAPE_REWARD = 0.25
    FLASH_ABUSE_PENALTY = 0.08

    # ---- Legacy PPO fields kept only for compatibility ----
    LAMDA = 0.95
    BETA_START = 0.001
    CLIP_PARAM = 0.2
    VF_COEF = 1.0

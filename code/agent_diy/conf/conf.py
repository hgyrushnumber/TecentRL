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
    # ----------------------------------------------------------------------
    # Local map encoder / 局部地图编码
    # ----------------------------------------------------------------------
    LOCAL_MAP_WINDOW = 21

    # 局部 CNN 只处理障碍物：
    # map_info: 0 = obstacle, 1 = walkable
    # 不再把 treasure / monster / buff 投影到局部 CNN，避免和全局实体图冗余。
    MAP_CHANNELS = 1
    MAP_FEATURE_DIM = LOCAL_MAP_WINDOW * LOCAL_MAP_WINDOW * MAP_CHANNELS

    MAP_VALUE_OBSTACLE = 0
    MAP_VALUE_WALKABLE = 1

    # ----------------------------------------------------------------------
    # Global entity map encoder / 全局实体热力图
    # ----------------------------------------------------------------------
    # 全局 CNN 处理 hero / treasure / monster / buff 的全局位置关系。
    # 使用 128x128 是因为环境坐标本身就是 128 尺度。
    GLOBAL_MAP_SIZE = 128
    GLOBAL_MAP_CHANNELS = 4
    GLOBAL_MAP_FEATURE_DIM = GLOBAL_MAP_SIZE * GLOBAL_MAP_SIZE * GLOBAL_MAP_CHANNELS

    # global_entity_map channels:
    # channel 0: hero
    # channel 1: treasure
    # channel 2: monster
    # channel 3: buff
    GLOBAL_CHANNEL_HERO = 0
    GLOBAL_CHANNEL_TREASURE = 1
    GLOBAL_CHANNEL_MONSTER = 2
    GLOBAL_CHANNEL_BUFF = 3

    # Gaussian heatmap sigma / 高斯扩散参数
    HERO_SIGMA = 1.5
    TREASURE_SIGMA = 2.0
    BUFF_SIGMA = 2.0
    MONSTER_SIGMA = 3.0
    MONSTER_SIGMA_SPEED_COEF = 1.0

    # ----------------------------------------------------------------------
    # Object types / 物件类型
    # ----------------------------------------------------------------------
    ORGAN_TYPE_TREASURE = 1
    ORGAN_TYPE_BUFF = 2

    # 是否允许使用 extra_info
    USE_EXTRA_INFO_FEATURE = True
    USE_EXTRA_INFO_REWARD = True

    # ----------------------------------------------------------------------
    # Feature dimensions / 特征维度
    # ----------------------------------------------------------------------
    # hero self: 4
    # monster features: 2 * 6 = 12
    # local obstacle map: 21 * 21 * 1 = 441
    # global entity map: 128 * 128 * 4 = 65536
    # legal action mask: 16
    # progress/status: 6
    # nearest treasure direction: 3
    # nearest buff direction: 3
    FEATURES = [
        4,                       # hero self: x, z, flash_cd, buff_remaining
        12,                      # monster features: 2 monsters * 6
        MAP_FEATURE_DIM,         # local obstacle map
        GLOBAL_MAP_FEATURE_DIM,  # global entity heatmap
        16,                      # legal action mask
        6,                       # progress / treasure / buff status
        3,                       # nearest treasure direction + distance
        3,                       # nearest buff direction + distance
    ]

    FEATURE_SPLIT_SHAPE = FEATURES
    FEATURE_LEN = sum(FEATURE_SPLIT_SHAPE)
    DIM_OF_OBSERVATION = FEATURE_LEN

    # ----------------------------------------------------------------------
    # Action space / 动作空间
    # ----------------------------------------------------------------------
    ACTION_NUM = 16

    # 约定：
    # 0~7  普通移动
    # 8~15 闪现动作
    # 如果环境动作编号不是这样，这里必须同步修改。
    FLASH_ACTION_START = 8

    # ----------------------------------------------------------------------
    # Network / 网络结构
    # ----------------------------------------------------------------------
    HIDDEN_DIM = 256
    MID_DIM = 128

    ACTOR_HEAD_INIT_GAIN = 0.01
    CRITIC_HEAD_INIT_GAIN = 1.0

    # ----------------------------------------------------------------------
    # SAC hyperparameters / SAC核心参数
    # ----------------------------------------------------------------------
    GAMMA = 0.98
    TAU = 0.005

    ALPHA = 0.2
    AUTO_ALPHA = True
    TARGET_ENTROPY = 1.0

    ALPHA_LR = 1e-5
    ALPHA_MIN = 0.01
    ALPHA_MAX = 2.0

    # ----------------------------------------------------------------------
    # Optimizer / 优化器参数
    # ----------------------------------------------------------------------
    INIT_LEARNING_RATE_START = 1e-4
    ACTOR_LR = 1e-4
    CRITIC_LR = 5e-5

    GRAD_CLIP_RANGE = 2.0
    ACTOR_UPDATE_INTERVAL = 1

    USE_TARGET_Q_CLIP = False
    TARGET_Q_CLIP = 30.0
    CRITIC_USE_HUBER = True

    # ----------------------------------------------------------------------
    # Reward scale / 奖励缩放
    # ----------------------------------------------------------------------
    REWARD_SCALE = 0.005

    # 生存奖励：先提升步数
    REWARD_SURVIVAL = 0.9

    # 移动奖励：防止看不到怪兽时原地不动
    REWARD_MOVE = 0.3

    # 怪兽距离塑形
    REWARD_MONSTER_DISTANCE = 35.0
    PENALTY_DANGER = 30.0
    DANGER_DISTANCE_TH = 0.32
    MONSTER_SPEED_DANGER_COEF = 1.0

    # 宝箱奖励：当前阶段先保守，避免为了宝箱过早死亡
    REWARD_TREASURE_SCORE = 35.0
    REWARD_TREASURE_APPROACH = 12.0
    PENALTY_TREASURE_AWAY = 4.0
    PENALTY_TREASURE_GREED_DANGER = 8.0
    TREASURE_SAFE_DISTANCE_TH = 0.38

    # Buff 奖励
    REWARD_BUFF_COLLECT = 20.0
    REWARD_BUFF_APPROACH = 10.0
    PENALTY_BUFF_AWAY = 2.0
    PENALTY_BUFF_GREED_DANGER = 2.0
    BUFF_SAFE_DISTANCE_TH = 0.35

    # 闪现奖励 / 惩罚
    REWARD_FLASH_ESCAPE = 45.0
    PENALTY_FLASH_TOWARD_MONSTER = 45.0
    PENALTY_FLASH_WASTE = 8.0

    FLASH_DANGER_DIST_TH = 0.35
    FLASH_ESCAPE_PROGRESS_TH = 0.03

    # 行为惩罚
    PENALTY_INVALID_MOVE = 5.0
    PENALTY_REPEAT_VISIT = 1.5

    # 终局奖励 / 惩罚
    TERMINAL_CAUGHT_PENALTY = -150.0
    TERMINAL_SUCCESS_REWARD = 100.0
    TERMINAL_TIMEOUT_PENALTY = -5.0

    # ----------------------------------------------------------------------
    # Normalization constants / 归一化常量
    # ----------------------------------------------------------------------
    MAP_SIZE = 128.0
    MAX_MONSTER_SPEED = 5.0
    MAX_DIST_BUCKET = 5.0
    MAX_FLASH_CD = 100.0
    MAX_BUFF_DURATION = 50.0
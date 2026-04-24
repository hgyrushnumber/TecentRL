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
    # Spatial encoder settings / 空间编码设置
    # ----------------------------------------------------------------------
    LOCAL_MAP_WINDOW = 21

    # CNN 局部地图通道：
    # channel 0: obstacle 障碍物
    # channel 1: treasure 宝箱
    # channel 2: monster 怪兽
    # channel 3: buff 增益道具
    MAP_CHANNELS = 4  # obstacle only
    MAP_FEATURE_DIM = LOCAL_MAP_WINDOW * LOCAL_MAP_WINDOW * MAP_CHANNELS

    # 这里的 0/2/3/4 只是推荐占位，如果环境编码不同，必须同步修改。
    MAP_VALUE_OBSTACLE = 0
    MAP_VALUE_TREASURE = 2
    MAP_VALUE_MONSTER = 3
    MAP_VALUE_BUFF = 4
    # ----------------------------------------------------------------------
    # Feature dimensions / 特征维度（共482维）
    # ----------------------------------------------------------------------
    FEATURES = [
        4,               # hero self feature
        5,               # monster 1 feature
        5,               # monster 2 feature
        6,               # out-of-vision monster relative info: 2 monsters x [dx, dz, dist]
        MAP_FEATURE_DIM, # local obstacle map
        16,              # legal action mask
        2,               # progress feature
        3,               # nearest treasure direction + distance: [dx, dz, dist_norm]
    ]

    FEATURE_SPLIT_SHAPE = FEATURES
    FEATURE_LEN = sum(FEATURE_SPLIT_SHAPE)
    DIM_OF_OBSERVATION = FEATURE_LEN

    # ----------------------------------------------------------------------
    # Action space / 动作空间：16个动作（8移动 + 8闪现）
    # ----------------------------------------------------------------------
    ACTION_NUM = 16

    # ----------------------------------------------------------------------
    # Network / 网络结构
    # ----------------------------------------------------------------------
    HIDDEN_DIM = 256
    MID_DIM = 128

    # ----------------------------------------------------------------------
    # SAC hyperparameters / SAC核心参数
    # ----------------------------------------------------------------------
    # 0.99 在当前 reward 尺度下容易让 Q target 涨到 100+
    # 0.98 可以降低长期回报尺度，让 Critic 更稳定
    GAMMA = 0.98

    # target network soft update
    # 0.01 偏快，target 追在线网络太紧；0.005 更稳
    TAU = 0.005

    # entropy temperature
    ALPHA = 0.2
    AUTO_ALPHA = True

    # 当前合法动作数约 8.5，最大熵 log(8.5)≈2.14
    # 之前 entropy 接近 2.1，说明策略太随机
    # 目标熵先设 1.0，让策略逐渐从随机转向有偏好
    TARGET_ENTROPY = 1.0

    ALPHA_LR = 1e-5
    ALPHA_MIN = 0.01
    ALPHA_MAX = 2.0

    # ----------------------------------------------------------------------
    # Optimizer / 优化器参数
    # ----------------------------------------------------------------------
    INIT_LEARNING_RATE_START = 1e-4

    # 推荐在新版 algorithm.py 中分别读取 ACTOR_LR / CRITIC_LR
    # Actor 当前梯度偏小，保留 1e-4
    ACTOR_LR = 1e-4

    # Critic 梯度仍偏大，降低到 5e-5
    CRITIC_LR = 5e-5

    # 当前 critic_grad_norm_post 长期贴 5，说明 5 太宽且长期触发
    # 改成 2，并配合降低 critic lr
    GRAD_CLIP_RANGE = 2.0

    # Actor 当前策略过随机，Actor 梯度偏小，可以每步更新
    ACTOR_UPDATE_INTERVAL = 1

    # target Q clipping
    # reward 缩放后通常不需要强制裁剪 target Q
    USE_TARGET_Q_CLIP = False
    TARGET_Q_CLIP = 30.0

    # Critic loss
    CRITIC_USE_HUBER = True

    # ----------------------------------------------------------------------
    # Reward scale / 奖励缩放
    # ----------------------------------------------------------------------
    # 主方案：等比缩小 reward
    REWARD_SCALE = 0.005


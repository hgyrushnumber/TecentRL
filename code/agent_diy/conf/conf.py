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

    # Feature dimensions / 特征维度（共61维）
    FEATURES = [
        6,    # 英雄自身特征
        9,    # 怪物1特征
        9,    # 怪物2特征
        4,    # 宝箱特征
        3,    # Buff特征
        16,   # 局部地图特征
        10,   # 合法动作掩码
        4,    # 进度特征
    ]
    FEATURE_SPLIT_SHAPE = FEATURES
    FEATURE_LEN = sum(FEATURE_SPLIT_SHAPE)   # 61
    DIM_OF_OBSERVATION = FEATURE_LEN

    # Action space / 动作空间：8移动 + 2技能
    ACTION_NUM = 10
    VALUE_NUM = 1

    # ── SAC 超参数 ─────────────────────────────────────────────────────
    GAMMA = 0.99                    # 折扣因子
    INIT_LEARNING_RATE_START = 3e-4 # Actor / Critic 学习率
    GRAD_CLIP_RANGE = 1.0           # 梯度裁剪

    # 自动熵调整 (auto-alpha tuning)
    # target_entropy = -log(1/|A|) * 0.98（接近均匀分布熵的98%）
    AUTO_ALPHA = True
    ALPHA_LR = 3e-4
    TARGET_ENTROPY_RATIO = 0.6      # 目标熵 = ratio * log(ACTION_NUM) ≈ 1.38，平衡探索与利用

    # Soft target update / 软更新系数
    TAU = 0.005

    # Replay buffer / 经验回放池
    REPLAY_BUFFER_SIZE = 20_000     # 缓冲区容量（约20局数据，自然淘汰旧策略，防止旧数据污染）
    BATCH_SIZE = 256                # 每次训练采样批大小
    LEARNING_STARTS = 1000          # 开始训练前先收集的样本数

    # ── 兼容性保留（部分接口仍会读取）────────────────────────────────
    LAMDA = 0.95
    BETA_START = 0.01
    CLIP_PARAM = 0.2
    VF_COEF = 0.5
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

    # Feature dimensions / 特征维度（共65维）
    FEATURES = [
        6,    # 英雄自身特征
        10,   # 怪物1特征（含方向角）
        10,   # 怪物2特征（含方向角）
        6,    # 宝箱特征（含方向向量）
        3,    # Buff特征
        16,   # 局部地图特征
        10,   # 合法动作掩码
        4,    # 进度特征
    ]
    FEATURE_SPLIT_SHAPE = FEATURES
    FEATURE_LEN = sum(FEATURE_SPLIT_SHAPE)   # 65
    DIM_OF_OBSERVATION = FEATURE_LEN

    # Action space / 动作空间：8移动 + 2技能
    ACTION_NUM = 10
    VALUE_NUM = 1

    # ── SAC 超参数 ─────────────────────────────────────────────────────
    GAMMA = 0.99                    # 折扣因子
    INIT_LEARNING_RATE_START = 3e-4 # Actor / Critic 学习率
    INIT_LEARNING_RATE_END = 3e-5   # 学习率衰减终点（cosine decay）
    LR_DECAY_STEPS = 100_000        # 学习率衰减步数
    GRAD_CLIP_RANGE = 0.5           # 梯度裁剪（收紧，减少梯度爆炸）

    # 自动熵调整 (auto-alpha tuning)
    # target_entropy = ratio * log(|A|)，平衡探索与利用
    AUTO_ALPHA = True
    ALPHA_LR = 3e-4
    TARGET_ENTROPY_RATIO = 0.5      # 降低目标熵，减少过度探索，加快收敛

    # Soft target update / 软更新系数
    TAU = 0.005                     # 回调至0.005，目标网络更新更稳定

    # Replay buffer / 经验回放池
    REPLAY_BUFFER_SIZE = 20_000     # 缓冲区容量
    BATCH_SIZE = 512                # 每次训练采样批大小
    LEARNING_STARTS = 8000          # 提高预热阈值，Q值充分稳定后再训练

    # ── 兼容性保留（部分接口仍会读取）────────────────────────────────
    LAMDA = 0.95
    BETA_START = 0.01
    CLIP_PARAM = 0.2
    VF_COEF = 0.5
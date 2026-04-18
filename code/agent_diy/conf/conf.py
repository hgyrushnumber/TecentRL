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

    # Feature dimensions / 特征维度（共102维）
    FEATURES = [
        6,    # 英雄自身特征
        11,   # 怪物1特征（含完整方向角 sin+cos）
        11,   # 怪物2特征（含完整方向角 sin+cos）
        6,    # 宝箱特征（含方向向量）
        5,    # Buff特征（含方向向量）
        49,   # 局部地图特征（7×7窗口）
        10,   # 合法动作掩码
        4,    # 进度特征
    ]
    FEATURE_SPLIT_SHAPE = FEATURES
    FEATURE_LEN = sum(FEATURE_SPLIT_SHAPE)   # 102
    DIM_OF_OBSERVATION = FEATURE_LEN

    # Action space / 动作空间：8移动 + 2技能
    ACTION_NUM = 10
    VALUE_NUM = 1

    # ── SAC 超参数 ─────────────────────────────────────────────────────
    GAMMA = 0.99                    # 折扣因子
    INIT_LEARNING_RATE_START = 5e-4 # Actor / Critic 学习率（提升）
    INIT_LEARNING_RATE_END = 1e-5   # 学习率衰减终点（降低）
    LR_DECAY_STEPS = 200_000        # 学习率衰减步数（延长至200k）
    GRAD_CLIP_RANGE = 1.0           # 梯度裁剪（放宽）

    # 自动熵调整 (auto-alpha tuning)
    # target_entropy = ratio * log(|A|)，平衡探索与利用
    AUTO_ALPHA = True
    ALPHA_LR = 3e-4
    TARGET_ENTROPY_RATIO = 0.5      # 降低到0.5，减少初期无效探索

    # Soft target update / 软更新系数
    TAU = 0.005                     # 回调至0.005，目标网络更新更稳定

    # Replay buffer / 经验回放池
    REPLAY_BUFFER_SIZE = 50_000     # 扩大缓冲区至50k，提升样本多样性
    BATCH_SIZE = 512                # 降低批大小，提升更新频率
    LEARNING_STARTS = 10_000        # 提升预热阈值至10k，确保样本多样性

    # ── 优先经验回放（PER）────────────────────────────────────────────
    PER_ALPHA = 0.6                 # 优先级指数（0=均匀采样，1=完全优先级）
    PER_BETA_START = 0.4            # IS权重初始值（逐渐增至1消除偏差）
    PER_BETA_FRAMES = 100_000       # β增长至1的帧数

    # ── 混合精度训练（AMP）────────────────────────────────────────────
    USE_AMP = True                  # 启用混合精度训练（GPU加速）

    # ── 兼容性保留（部分接口仍会读取）────────────────────────────────
    LAMDA = 0.95
    BETA_START = 0.01
    CLIP_PARAM = 0.2
    VF_COEF = 0.5
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

    # Feature dimensions / 特征维度（共114维）
    FEATURES = [
        6,    # 英雄自身特征
        11,   # 怪物1特征（含完整方向角 sin+cos）
        11,   # 怪物2特征（含完整方向角 sin+cos）
        6,    # 宝箱特征（含方向向量）
        5,    # Buff特征（含方向向量）
        49,   # 局部地图特征（7×7窗口）
        16,   # 合法动作掩码（16维：8移动+8闪现）
        10,   # 时序/规划特征（含ETA、逃逸性等）
    ]
    FEATURE_SPLIT_SHAPE = FEATURES
    FEATURE_LEN = sum(FEATURE_SPLIT_SHAPE)   # 114
    DIM_OF_OBSERVATION = FEATURE_LEN

    # Action space / 动作空间：8移动 + 8方向闪现
    ACTION_NUM = 16
    VALUE_NUM = 1

    # ── SAC 超参数 ─────────────────────────────────────────────────────
    GAMMA = 0.99                    # 折扣因子
    INIT_LEARNING_RATE_START = 1e-4 # Actor / Critic 学习率（降低，延缓收敛）
    INIT_LEARNING_RATE_END = 1e-5   # 学习率衰减终点
    LR_DECAY_STEPS = 2_000_000      # 延长学习率衰减步数，防止Actor锁死
    GRAD_CLIP_RANGE = 1.0           # 梯度裁剪

    # 自动熵调整 (auto-alpha tuning)
    # target_entropy = ratio * log(|A|)，平衡探索与利用
    AUTO_ALPHA = True
    ALPHA_LR = 3e-3    # 提高学习率，加速α调节响应
    TARGET_ENTROPY_RATIO = 0.9     # 进一步提高目标熵，抑制策略过早塌缩

    # Soft target update / 软更新系数
    TAU = 0.005                     # 目标网络软更新系数

    # Replay buffer / 经验回放池
    REPLAY_BUFFER_SIZE = 200_000    # 经验回放池容量（提升，增强样本多样性）
    BATCH_SIZE = 512                # 批大小（降低，缓解过平滑/早收敛）
    LEARNING_STARTS = 10_000        # 增大预热阈值至10k，收集更多多样化样本
    UPDATES_PER_LEARN = 8           # 固定每轮更新次数，控制UTD比

    # ── 优先经验回放（PER）────────────────────────────────────────────
    PER_ALPHA = 0.3                 # 优先级指数（降低，增加采样多样性）
    PER_BETA_START = 0.4            # IS权重初始值（逐渐增至1消除偏差）
    PER_BETA_FRAMES = 100_000       # β增长至1的帧数

    # ── 混合精度训练（AMP）────────────────────────────────────────────
    USE_AMP = True                  # 启用混合精度训练（GPU加速）

    # α约束范围（自动熵调节）- 防止过早衰减
    ALPHA_MIN = 0.2    # 提高下限，防止α过早衰减
    ALPHA_MAX = 3.0

    # 训练稳定性（奖励/目标Q裁剪）
    REWARD_CLIP = 2.0
    TARGET_Q_CLIP = 30.0
    # 分数对齐奖励：每步奖励 = score_delta / SCORE_REWARD_SCALE + SHAPING_REWARD_WEIGHT * shaping
    # 主目标为 total_score，建议以分数增量作为主信号，shaping 仅作辅助。
    SCORE_REWARD_SCALE = 50.0
    SHAPING_REWARD_WEIGHT = 0.2

    # ── 兼容性保留（部分接口仍会读取）────────────────────────────────
    LAMDA = 0.95
    BETA_START = 0.01
    CLIP_PARAM = 0.2
    VF_COEF = 0.5

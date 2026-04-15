#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Configuration for Gorge Chase DIY Agent (Enhanced PPO).
峡谷追猎 DIY Agent 配置（增强版 PPO）。

改进要点：
  1. 更丰富的特征维度（60D vs PPO的40D），加入方向向量、危险感知等
  2. 更大的网络隐层
  3. 更精细的超参数调整（更高的熵系数鼓励探索、更小的clip）
  4. 引入 LR 调度支持
"""


class Config:

    # -----------------------------------------------------------------------
    # Feature dimensions / 特征维度
    # 英雄自身: 6D (pos_x, pos_z, flash_cd, buff_remain, hp_ratio, has_buff)
    # 怪物1:   8D (in_view, pos_x, pos_z, speed, dist, dir_x, dir_z, danger)
    # 怪物2:   8D (同上)
    # 局部地图: 16D (4×4 障碍物掩码)
    # 合法动作: 8D
    # 进度:    4D (step_norm, survival_ratio, time_pressure, alive_bonus)
    # 方向偏好: 10D (最近出逃方向 one-hot + 2D 方向分量)
    # -----------------------------------------------------------------------
    FEATURES = [6, 8, 8, 16, 8, 4, 10]
    FEATURE_SPLIT_SHAPE = FEATURES
    FEATURE_LEN = sum(FEATURE_SPLIT_SHAPE)
    DIM_OF_OBSERVATION = FEATURE_LEN  # 60

    # Action space / 动作空间：8个方向
    ACTION_NUM = 8

    # Value head / 价值头
    VALUE_NUM = 1

    # -----------------------------------------------------------------------
    # PPO hyperparameters / PPO 超参数
    # -----------------------------------------------------------------------
    GAMMA = 0.99          # 折扣因子
    LAMDA = 0.95          # GAE lambda

    INIT_LEARNING_RATE_START = 1e-4   # 初始学习率（略小于PPO，避免过拟合早期噪声）
    LR_DECAY_STEPS = 50000            # 学习率线性衰减步数
    LR_END = 1e-5                     # 最终学习率

    BETA_START = 0.01     # 熵系数初始值（比PPO大，鼓励更多探索）
    BETA_END = 0.001      # 熵系数最终值
    BETA_DECAY_STEPS = 80000

    CLIP_PARAM = 0.15     # PPO clip 范围（略小于0.2，更稳定）
    VF_COEF = 0.5         # 价值损失权重
    GRAD_CLIP_RANGE = 0.5 # 梯度裁剪

    # Normalization / 优势函数归一化
    NORMALIZE_ADVANTAGE = True

    # Mini-batch epochs / 每批样本多次更新轮数
    PPO_EPOCHS = 1
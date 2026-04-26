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
    # 如果本地训练压力较大，可改成 32，但必须同步修改 preprocessor.to_grid 和 model.py。
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

    # Gaussian heatmap sigma / CNN 输入热力图高斯扩散参数
    # 注意：这组 sigma 用于 CNN 定位，不用于 reward 势场引导。
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
    # anti-stuck feature: 2, [dx10_norm, dz10_norm]
    # nearest treasure direction: 3
    # nearest buff direction: 3
    FEATURES = [
        4,                       # hero self: x, z, flash_cd, buff_remaining
        12,                      # monster features: 2 monsters * 6
        MAP_FEATURE_DIM,         # local obstacle map
        GLOBAL_MAP_FEATURE_DIM,  # global entity heatmap
        16,                      # legal action mask
        6,                       # progress / treasure / buff status
        2,                       # anti-stuck feature: dx10_norm, dz10_norm
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

    ALPHA_LR = 5e-6
    ALPHA_MIN = 0.01
    ALPHA_MAX = 2.0

    # ----------------------------------------------------------------------
    # Optimizer / 优化器参数
    # ----------------------------------------------------------------------
    INIT_LEARNING_RATE_START = 1e-4

    # 新增 feature 后通常需要重新训练；如果做 fine-tune，建议降低学习率。
    ACTOR_LR = 5e-5
    CRITIC_LR = 3e-5

    GRAD_CLIP_RANGE = 2.0
    ACTOR_UPDATE_INTERVAL = 1

    USE_TARGET_Q_CLIP = False
    TARGET_Q_CLIP = 30.0
    CRITIC_USE_HUBER = True

    # ----------------------------------------------------------------------
    # Reward scale / 奖励缩放
    # ----------------------------------------------------------------------
    REWARD_SCALE = 0.005

    # ----------------------------------------------------------------------
    # Survival / 生存
    # ----------------------------------------------------------------------
    REWARD_SURVIVAL = 1.0

    # ----------------------------------------------------------------------
    # Global field shaping / 全局势场塑形
    # ----------------------------------------------------------------------
    # 当前仅作为轻量方向引导，不再作为强惩罚主导。
    REWARD_GLOBAL_MONSTER_FIELD = 6.0
    PENALTY_GLOBAL_MONSTER_FIELD = 8.0
    FIELD_PROGRESS_EPS = 0.005

    REWARD_GLOBAL_TREASURE_FIELD = 3.0
    PENALTY_GLOBAL_TREASURE_GREED = 3.0

    REWARD_GLOBAL_BUFF_FIELD = 4.0
    PENALTY_GLOBAL_BUFF_GREED = 0.5

    # Reward 势场 sigma：
    # 这组 sigma 用于 reward 的方向引导，不是 CNN 输入热力图 sigma。
    REWARD_MONSTER_FIELD_SIGMA = 6.0
    REWARD_TREASURE_FIELD_SIGMA = 12.0
    REWARD_BUFF_FIELD_SIGMA = 12.0

    # ----------------------------------------------------------------------
    # Hard danger penalty / 近距离硬危险惩罚
    # ----------------------------------------------------------------------
    # 之前 danger_penalty 过大，容易压过正向学习信号，这里先降压。
    PENALTY_DANGER = 20.0
    DANGER_DISTANCE_TH = 0.30
    MONSTER_SPEED_DANGER_COEF = 1.0

    # ----------------------------------------------------------------------
    # Treasure event reward / 宝箱事件奖励
    # ----------------------------------------------------------------------
    # 真正吃到宝箱奖励。
    REWARD_TREASURE_SCORE = 28.0
    TREASURE_SAFE_DISTANCE_TH = 0.42
    TREASURE_DANGER_SCORE_SCALE = 0.35

    # 第一次看见某个宝箱位置时给予奖励。
    REWARD_FIRST_SEEN_TREASURE = 3.0

    # ----------------------------------------------------------------------
    # Buff event reward / Buff 事件奖励
    # ----------------------------------------------------------------------
    # 真正吃到 Buff 奖励。
    REWARD_BUFF_COLLECT = 28.0
    BUFF_SAFE_DISTANCE_TH = 0.38

    # ----------------------------------------------------------------------
    # Anti-stuck reward / 防磨蹭奖励
    # ----------------------------------------------------------------------
    # 如果当前坐标和 10 步前坐标的欧式距离小于阈值，认为存在卡墙/磨蹭。
    ANTI_STUCK_WINDOW = 10
    ANTI_STUCK_DISTANCE_TH = 5.0

    # 注意：这里是 raw reward，最终还会乘 REWARD_SCALE。
    # PENALTY_ANTI_STUCK = 8.0 时，scaled 后约为 -0.04。
    PENALTY_ANTI_STUCK = 8.0

    # ----------------------------------------------------------------------
    # Flash reward / 闪现奖励与惩罚
    # ----------------------------------------------------------------------
    REWARD_FLASH_ESCAPE_BASE = 10.0
    REWARD_FLASH_ESCAPE = 45.0

    # preprocessor.py 中应使用固定惩罚，不再乘 abs(monster_progress)。
    PENALTY_FLASH_TOWARD_MONSTER = 18.0
    PENALTY_FLASH_WASTE = 6.0

    FLASH_DANGER_DIST_TH = 0.40
    FLASH_ESCAPE_PROGRESS_TH = 0.01

    # ----------------------------------------------------------------------
    # Movement / 行为约束
    # ----------------------------------------------------------------------
    # 移动奖励只作为防止原地不动，preprocessor.py 中应只奖励安全移动。
    REWARD_MOVE = 0.15

    PENALTY_INVALID_MOVE = 5.0

    # 已有 repeat_penalty 和 anti-stuck 有部分重叠，先降低 repeat 权重。
    PENALTY_REPEAT_VISIT = 0.5

    # ----------------------------------------------------------------------
    # Terminal reward / 终局奖励与惩罚
    # ----------------------------------------------------------------------
    TERMINAL_CAUGHT_PENALTY = -220.0
    TERMINAL_SUCCESS_REWARD = 150.0
    TERMINAL_TIMEOUT_PENALTY = 20.0

    # ----------------------------------------------------------------------
    # Normalization constants / 归一化常量
    # ----------------------------------------------------------------------
    MAP_SIZE = 128.0
    MAX_MONSTER_SPEED = 5.0
    MAX_DIST_BUCKET = 5.0
    MAX_FLASH_CD = 100.0
    MAX_BUFF_DURATION = 50.0
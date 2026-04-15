#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Enhanced feature preprocessor and reward design for Gorge Chase DIY Agent.
峡谷追猎 DIY Agent 增强特征预处理与奖励设计。

特征改进（60D vs PPO 的 40D）：
  英雄自身: 6D  → 新增 has_buff 标志和 buff_active_bonus
  怪物特征: 8D×2 → 新增方向向量(dir_x, dir_z)和危险等级 danger
  进度特征: 4D  → 新增 time_pressure（剩余步数压力）
  方向偏好: 10D → 8方向逃跑分数 one-hot + 2D 安全方向分量

奖励改进：
  - 存活奖励（基础）
  - 与最近怪物距离增大给正奖励（与PPO相同逻辑，但更大权重）
  - 危险惩罚（怪物非常近时额外惩罚）
  - 拾取 buff 奖励
  - 时间步惩罚随时间线性增大（激励快速逃跑而非原地等待）
"""

import numpy as np

# Map size / 地图尺寸
MAP_SIZE = 128.0
# Max monster speed / 最大怪物速度
MAX_MONSTER_SPEED = 5.0
# Max flash cooldown / 最大闪现冷却
MAX_FLASH_CD = 2000.0
# Max buff duration / buff最大持续时间
MAX_BUFF_DURATION = 50.0
# Danger threshold (norm dist) / 危险距离阈值（归一化）
DANGER_THRESHOLD = 0.1
# Very close danger threshold / 极度危险距离
CRITICAL_THRESHOLD = 0.05

# 8方向 angle（弧度），对应动作 0~7
_ACTION_ANGLES = np.array([
    0.0,            # 右
    np.pi / 4,      # 右上
    np.pi / 2,      # 上
    3 * np.pi / 4,  # 左上
    np.pi,          # 左
    5 * np.pi / 4,  # 左下
    3 * np.pi / 2,  # 下
    7 * np.pi / 4,  # 右下
], dtype=np.float32)

_ACTION_DX = np.cos(_ACTION_ANGLES)
_ACTION_DZ = np.sin(_ACTION_ANGLES)


def _norm(v, v_max, v_min=0.0):
    """Normalize value to [0, 1]. / 归一化到 [0,1]"""
    v = float(np.clip(v, v_min, v_max))
    return (v - v_min) / (v_max - v_min) if (v_max - v_min) > 1e-6 else 0.0


class Preprocessor:
    def __init__(self):
        self.reset()

    def reset(self):
        self.step_no = 0
        self.max_step = 1000
        self.last_min_monster_dist_norm = 0.5
        self.last_had_buff = 0.0
        self.prev_hero_pos = None

    def feature_process(self, env_obs, last_action):
        """Process env_obs into 60D feature, legal_action mask, and reward.

        将 env_obs 转换为 60D 特征向量、合法动作掩码和即时奖励。
        """
        observation = env_obs["observation"]
        frame_state = observation["frame_state"]
        env_info = observation["env_info"]
        map_info = observation["map_info"]
        legal_act_raw = observation["legal_action"]

        self.step_no = observation["step_no"]
        self.max_step = env_info.get("max_step", 1000)

        # ------------------------------------------------------------------
        # 1. 英雄自身特征 (6D)
        # ------------------------------------------------------------------
        hero = frame_state["heroes"]
        hero_pos = hero["pos"]
        hero_x = hero_pos["x"]
        hero_z = hero_pos["z"]
        hero_x_norm = _norm(hero_x, MAP_SIZE)
        hero_z_norm = _norm(hero_z, MAP_SIZE)
        flash_cd_norm = _norm(hero.get("flash_cooldown", 0), MAX_FLASH_CD)
        buff_remain = hero.get("buff_remaining_time", 0)
        buff_remain_norm = _norm(buff_remain, MAX_BUFF_DURATION)
        has_buff = float(buff_remain > 0)
        # 英雄地图中心偏移（是否靠近边缘，靠近边缘更危险）
        center_dist = np.sqrt((hero_x_norm - 0.5) ** 2 + (hero_z_norm - 0.5) ** 2)
        edge_danger = float(np.clip(center_dist * 2.0, 0.0, 1.0))

        hero_feat = np.array(
            [hero_x_norm, hero_z_norm, flash_cd_norm, buff_remain_norm, has_buff, edge_danger],
            dtype=np.float32,
        )

        # ------------------------------------------------------------------
        # 2. 怪物特征 (8D × 2)
        # ------------------------------------------------------------------
        monsters = frame_state.get("monsters", [])
        monster_feats = []
        all_monster_dist_norms = []

        for i in range(2):
            if i < len(monsters):
                m = monsters[i]
                is_in_view = float(m.get("is_in_view", 0))
                m_pos = m["pos"]
                if is_in_view:
                    m_x = m_pos["x"]
                    m_z = m_pos["z"]
                    m_x_norm = _norm(m_x, MAP_SIZE)
                    m_z_norm = _norm(m_z, MAP_SIZE)
                    m_speed_norm = _norm(m.get("speed", 1), MAX_MONSTER_SPEED)

                    raw_dist = np.sqrt((hero_x - m_x) ** 2 + (hero_z - m_z) ** 2)
                    dist_norm = _norm(raw_dist, MAP_SIZE * 1.41)

                    # 方向向量（从怪物指向英雄，归一化）
                    if raw_dist > 1e-6:
                        dir_x = (hero_x - m_x) / raw_dist / MAP_SIZE
                        dir_z = (hero_z - m_z) / raw_dist / MAP_SIZE
                    else:
                        dir_x, dir_z = 0.0, 0.0

                    # 危险等级：距离越近 + 速度越快 = 越危险
                    danger = float(np.clip(
                        (1.0 - dist_norm) * (0.5 + 0.5 * m_speed_norm), 0.0, 1.0
                    ))
                    all_monster_dist_norms.append(dist_norm)
                else:
                    m_x_norm = m_z_norm = 0.0
                    m_speed_norm = 0.0
                    dist_norm = 1.0
                    dir_x = dir_z = 0.0
                    danger = 0.0

                monster_feats.append(np.array(
                    [is_in_view, m_x_norm, m_z_norm, m_speed_norm,
                     dist_norm, dir_x, dir_z, danger],
                    dtype=np.float32,
                ))
            else:
                monster_feats.append(np.zeros(8, dtype=np.float32))

        # ------------------------------------------------------------------
        # 3. 局部地图特征 (16D)，4×4 障碍物掩码
        # ------------------------------------------------------------------
        map_feat = np.zeros(16, dtype=np.float32)
        if map_info is not None and len(map_info) >= 13:
            center = len(map_info) // 2
            flat_idx = 0
            for row in range(center - 2, center + 2):
                for col in range(center - 2, center + 2):
                    if 0 <= row < len(map_info) and 0 <= col < len(map_info[0]):
                        map_feat[flat_idx] = float(map_info[row][col] != 0)
                    flat_idx += 1

        # ------------------------------------------------------------------
        # 4. 合法动作掩码 (8D)
        # ------------------------------------------------------------------
        legal_action = [1] * 8
        if isinstance(legal_act_raw, list) and legal_act_raw:
            if isinstance(legal_act_raw[0], bool):
                for j in range(min(8, len(legal_act_raw))):
                    legal_action[j] = int(legal_act_raw[j])
            else:
                valid_set = {int(a) for a in legal_act_raw if int(a) < 8}
                legal_action = [1 if j in valid_set else 0 for j in range(8)]

        if sum(legal_action) == 0:
            legal_action = [1] * 8

        # ------------------------------------------------------------------
        # 5. 进度特征 (4D)
        # ------------------------------------------------------------------
        step_norm = _norm(self.step_no, self.max_step)
        remaining_norm = 1.0 - step_norm
        # 时间压力：剩余步数越少，压力越大（非线性）
        time_pressure = float(np.clip(step_norm ** 2, 0.0, 1.0))
        # 存活奖励积累（鼓励更长生存）
        alive_bonus = step_norm * 0.5

        progress_feat = np.array(
            [step_norm, remaining_norm, time_pressure, alive_bonus],
            dtype=np.float32,
        )

        # ------------------------------------------------------------------
        # 6. 方向偏好特征 (10D)：每个方向的"安全分"+ 最优方向分量
        # ------------------------------------------------------------------
        # 对每个方向评估安全分（远离最近怪物的程度）
        direction_scores = np.zeros(8, dtype=np.float32)
        min_dist_norm = min(all_monster_dist_norms) if all_monster_dist_norms else 1.0

        for act_i in range(8):
            score = 0.0
            for m_feat in monster_feats:
                if m_feat[0] > 0:  # is_in_view
                    # 动作方向与逃跑方向（指向英雄）的点积
                    dot = _ACTION_DX[act_i] * m_feat[5] + _ACTION_DZ[act_i] * m_feat[6]
                    score += float(np.clip(dot, 0.0, 1.0)) * m_feat[7]  # 危险加权
            direction_scores[act_i] = score

        # Softmax normalize
        d_exp = np.exp(direction_scores - direction_scores.max())
        direction_scores_norm = d_exp / (d_exp.sum() + 1e-8)

        # 最优逃跑方向的 2D 分量
        best_act = int(np.argmax(direction_scores_norm))
        escape_dir = np.array([_ACTION_DX[best_act], _ACTION_DZ[best_act]], dtype=np.float32)

        dir_feat = np.concatenate([direction_scores_norm, escape_dir])  # 10D

        # ------------------------------------------------------------------
        # 7. 拼接所有特征 (60D)
        # ------------------------------------------------------------------
        feature = np.concatenate([
            hero_feat,           # 6D
            monster_feats[0],    # 8D
            monster_feats[1],    # 8D
            map_feat,            # 16D
            np.array(legal_action, dtype=np.float32),  # 8D
            progress_feat,       # 4D
            dir_feat,            # 10D
        ])

        # ------------------------------------------------------------------
        # 8. 奖励设计
        # ------------------------------------------------------------------
        cur_min_dist_norm = min(all_monster_dist_norms) if all_monster_dist_norms else 1.0

        # 基础存活奖励
        survive_reward = 0.02

        # 距离整形奖励（怪物远离则正奖励，靠近则惩罚，权重比PPO更大）
        dist_shaping = 0.15 * (cur_min_dist_norm - self.last_min_monster_dist_norm)

        # 危险惩罚（怪物极近时给额外惩罚）
        danger_penalty = 0.0
        if cur_min_dist_norm < DANGER_THRESHOLD:
            danger_penalty = -0.1 * (DANGER_THRESHOLD - cur_min_dist_norm) / DANGER_THRESHOLD
        if cur_min_dist_norm < CRITICAL_THRESHOLD:
            danger_penalty -= 0.2  # 极度危险额外惩罚

        # 拾取 buff 奖励（has_buff 从0变1）
        buff_pickup_reward = 0.0
        if has_buff > 0 and self.last_had_buff == 0:
            buff_pickup_reward = 0.3

        self.last_min_monster_dist_norm = cur_min_dist_norm
        self.last_had_buff = has_buff

        reward = [survive_reward + dist_shaping + danger_penalty + buff_pickup_reward]

        return feature, legal_action, reward
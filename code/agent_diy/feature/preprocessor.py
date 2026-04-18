#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Feature preprocessor and reward design for Gorge Chase PPO.
峡谷追猎 PPO 特征预处理与奖励设计。
"""

import numpy as np

# Map size / 地图尺寸（128×128）
MAP_SIZE = 128.0
# Max monster speed / 最大怪物速度
MAX_MONSTER_SPEED = 5.0
# Max flash cooldown / 最大闪现冷却步数
MAX_FLASH_CD = 2000.0
# Max buff duration / buff最大持续时间
MAX_BUFF_DURATION = 50.0
# Max relative velocity / 最大相对速度（每步像素）
MAX_REL_VELOCITY = 10.0
# Max collision risk / 最大碰撞风险（speed/dist）
MAX_COLLISION_RISK = 10.0
# Max distance on map diagonal
MAX_DIST = MAP_SIZE * 1.41


def _norm(v, v_max, v_min=0.0):
    """Normalize value to [0, 1]. / 归一化到 [0, 1]。"""
    v = float(np.clip(v, v_min, v_max))
    return (v - v_min) / (v_max - v_min) if (v_max - v_min) > 1e-6 else 0.0


class Preprocessor:
    def __init__(self):
        self.history_length = 3   # 保留3步历史（减少内存占用）
        self.reset()

    def reset(self):
        self.step_no = 0
        self.max_step = 1000
        self.last_min_monster_dist_norm = 0.5
        # 怪物历史位置（用于相对速度计算）：list of [pos_x, pos_z] or None
        self.monster_prev_pos = [None, None]
        self.last_hero_pos = None
        self.last_treasure_count = -1   # -1 表示未初始化
        self.last_buff_active = False
        self.last_treasure_dist_norm = -1.0  # -1 表示未初始化，用于 delta 奖励

    def feature_process(self, env_obs, last_action):
        """Process env_obs into feature vector, legal_action mask, and reward.

        将 env_obs 转换为特征向量、合法动作掩码和即时奖励。
        """
        observation = env_obs["observation"]
        frame_state = observation["frame_state"]
        env_info = observation["env_info"]
        map_info = observation["map_info"]
        legal_act_raw = observation["legal_action"]

        self.step_no = observation["step_no"]
        self.max_step = env_info.get("max_step", 1000)

        # ── 英雄特征 (6D) ────────────────────────────────────────────────
        hero = frame_state["heroes"]
        hero_pos = hero["pos"]
        hx = hero_pos["x"]
        hz = hero_pos["z"]
        flash_cd = hero.get("flash_cooldown", 0)
        talent_cd = hero.get("talent_cooldown", 0)

        hero_feat = np.array([
            _norm(hx, MAP_SIZE),
            _norm(hz, MAP_SIZE),
            _norm(flash_cd, MAX_FLASH_CD),
            _norm(hero.get("buff_remaining_time", 0), MAX_BUFF_DURATION),
            1.0 if flash_cd == 0 else 0.0,    # 闪现可用
            1.0 if talent_cd == 0 else 0.0,   # 天赋可用
        ], dtype=np.float32)

        buff_remain_norm = hero_feat[3]

        # ── 怪物特征 (11D × 2) ───────────────────────────────────────────
        # 11D = is_in_view / pos_x / pos_z / speed / dist / vel_x / vel_z /
        #        pred_dist / collision_risk / dir_sin / dir_cos
        # 修复：同时保留 dir_sin + dir_cos，方向角编码完整（原版仅 sin 导致90°/270°混淆）
        # 修复：怪物不在视野时保留上一帧估算位置，不直接将 dist 置为最远值1.0，
        #        避免模型误判"怪物已消失"而放松警惕
        monsters = frame_state.get("monsters", [])
        monster_feats = []
        cur_min_dist_norm = 1.0
        max_collision_risk_norm = 0.0

        for i in range(2):
            if i < len(monsters) and monsters[i].get("is_in_view", 0):
                m = monsters[i]
                mx = m["pos"]["x"]
                mz = m["pos"]["z"]
                spd = m.get("speed", 1)

                raw_dist = np.sqrt((hx - mx) ** 2 + (hz - mz) ** 2)
                dist_norm = _norm(raw_dist, MAX_DIST)
                cur_min_dist_norm = min(cur_min_dist_norm, dist_norm)

                # 相对速度（与上一帧位置差）
                if self.monster_prev_pos[i] is not None:
                    pvx, pvz = self.monster_prev_pos[i]
                    vel_x = mx - pvx
                    vel_z = mz - pvz
                else:
                    vel_x = vel_z = 0.0
                self.monster_prev_pos[i] = (mx, mz)

                # 线性预测3步后距离
                pred_x = np.clip(mx + vel_x * 3, 0, MAP_SIZE)
                pred_z = np.clip(mz + vel_z * 3, 0, MAP_SIZE)
                pred_dist_norm = _norm(
                    np.sqrt((hx - pred_x) ** 2 + (hz - pred_z) ** 2), MAX_DIST
                )

                # 碰撞风险 = speed / distance
                cr_norm = _norm(spd / max(raw_dist, 1.0), MAX_COLLISION_RISK)
                max_collision_risk_norm = max(max_collision_risk_norm, cr_norm)

                # 方向角：同时保留 sin + cos，完整编码来袭方向（修复：原版只有sin，方向歧义）
                angle = np.arctan2(mz - hz, mx - hx)  # [-π, π]
                dir_sin = (np.sin(angle) + 1.0) * 0.5  # [0, 1]
                dir_cos = (np.cos(angle) + 1.0) * 0.5  # [0, 1]

                monster_feats.append(np.array([
                    1.0,
                    _norm(mx, MAP_SIZE),
                    _norm(mz, MAP_SIZE),
                    _norm(spd, MAX_MONSTER_SPEED),
                    dist_norm,
                    _norm(vel_x, MAX_REL_VELOCITY),
                    _norm(vel_z, MAX_REL_VELOCITY),
                    pred_dist_norm,
                    cr_norm,
                    dir_sin,
                    dir_cos,   # 修复补全：方向角 cos（原版缺失导致 90°/270° 无法区分）
                ], dtype=np.float32))
            else:
                # 修复：怪物不在视野时，利用上一帧保存的位置进行惯性估算
                # 原版直接置 dist_norm=1.0 → 模型误认为怪物"安全最远"
                if self.monster_prev_pos[i] is not None:
                    est_mx, est_mz = self.monster_prev_pos[i]
                    est_raw_dist = np.sqrt((hx - est_mx) ** 2 + (hz - est_mz) ** 2)
                    est_dist_norm = _norm(est_raw_dist, MAX_DIST)
                    # 视野外用估算距离，但 is_in_view=0 告知模型此为估算值
                    cur_min_dist_norm = min(cur_min_dist_norm, est_dist_norm)
                    monster_feats.append(np.array([
                        0., 0., 0., 0.,
                        est_dist_norm,   # 估算距离，而非固定1.0
                        0., 0.,
                        est_dist_norm,   # pred_dist 同估算
                        0., 0.5, 0.5,    # dir sin/cos 置中性值
                    ], dtype=np.float32))
                else:
                    # 从未出现过，真正未知
                    monster_feats.append(np.array([0., 0., 0., 0., 1., 0., 0., 1., 0., 0.5, 0.5], dtype=np.float32))

        # ── 宝箱特征 (6D) ────────────────────────────────────────────────
        # 相比原4D，新增方向向量(dx_norm, dz_norm)，模型无需自学位置差
        treasure_feat = np.zeros(6, dtype=np.float32)
        try:
            treasures = frame_state.get("treasures", [])
            if treasures:
                min_td = float("inf")
                tx, tz = 0.0, 0.0
                for t in treasures:
                    if isinstance(t, dict) and "pos" in t:
                        tp = t["pos"]
                        td = np.sqrt((hx - tp["x"]) ** 2 + (hz - tp["z"]) ** 2)
                        if td < min_td:
                            min_td, tx, tz = td, tp["x"], tp["z"]
                if min_td < float("inf"):
                    treasure_feat[0] = _norm(tx, MAP_SIZE)
                    treasure_feat[1] = _norm(tz, MAP_SIZE)
                    treasure_feat[2] = _norm(min_td, MAX_DIST)
                    treasure_feat[3] = _norm(len(treasures), 10.0)
                    # 方向向量：归一化到 [0,1]（原始范围 [-1,1]）
                    dx = (tx - hx) / max(min_td, 1e-6)
                    dz = (tz - hz) / max(min_td, 1e-6)
                    treasure_feat[4] = (dx + 1.0) * 0.5   # [0, 1]
                    treasure_feat[5] = (dz + 1.0) * 0.5   # [0, 1]
        except Exception:
            pass

        # ── Buff特征 (5D) ────────────────────────────────────────────────
        # 新增方向向量(dx_norm, dz_norm)，与宝箱特征对齐，帮助模型感知buff方向
        buff_feat = np.zeros(5, dtype=np.float32)
        try:
            buffs = frame_state.get("buffs", [])
            if buffs:
                min_bd = float("inf")
                bx, bz = 0.0, 0.0
                for b in buffs:
                    if isinstance(b, dict) and "pos" in b:
                        bp = b["pos"]
                        bd = np.sqrt((hx - bp["x"]) ** 2 + (hz - bp["z"]) ** 2)
                        if bd < min_bd:
                            min_bd, bx, bz = bd, bp["x"], bp["z"]
                if min_bd < float("inf"):
                    buff_feat[0] = _norm(bx, MAP_SIZE)
                    buff_feat[1] = _norm(bz, MAP_SIZE)
                    buff_feat[2] = _norm(min_bd, MAX_DIST)
                    # 方向向量：归一化到 [0,1]（原始范围 [-1,1]）
                    dx = (bx - hx) / max(min_bd, 1e-6)
                    dz = (bz - hz) / max(min_bd, 1e-6)
                    buff_feat[3] = (dx + 1.0) * 0.5   # [0, 1]
                    buff_feat[4] = (dz + 1.0) * 0.5   # [0, 1]
        except Exception:
            pass

        # ── 局部地图特征 (49D) ───────────────────────────────────────────
        # 修复：窗口从 4×4=16 扩大到 7×7=49，覆盖范围更广，
        # 模型可提前感知周围障碍，防止被逼入死角
        map_feat = np.zeros(49, dtype=np.float32)
        if map_info is not None and len(map_info) >= 7:
            center = len(map_info) // 2
            flat_idx = 0
            for row in range(center - 3, center + 4):
                for col in range(center - 3, center + 4):
                    if 0 <= row < len(map_info) and 0 <= col < len(map_info[0]):
                        map_feat[flat_idx] = float(map_info[row][col] != 0)
                    flat_idx += 1

        # ── 合法动作掩码 (10D) ───────────────────────────────────────────
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
        legal_action.append(1 if flash_cd == 0 else 0)   # [8] 闪现
        legal_action.append(1 if talent_cd == 0 else 0)  # [9] 天赋

        # ── 进度特征 (4D) ────────────────────────────────────────────────
        # 4 个独立信息：归一化步数 / 距离最近怪物 / 碰撞风险 / 后半段标志
        step_norm = _norm(self.step_no, self.max_step)
        progress_feat = np.array([
            step_norm,
            cur_min_dist_norm,                        # 当前最近怪物距离
            max_collision_risk_norm,                  # 当前最大碰撞风险
            1.0 if self.step_no > self.max_step * 0.5 else 0.0,  # 后半段标志
        ], dtype=np.float32)

        # ── 拼接特征 (102D) ──────────────────────────────────────────────
        # 6 + 11 + 11 + 6 + 5 + 49 + 10 + 4 = 102
        feature = np.concatenate([
            hero_feat,           # 6
            monster_feats[0],    # 11  (修复：+dir_cos，完整方向角编码)
            monster_feats[1],    # 11  (修复：+dir_cos，完整方向角编码)
            treasure_feat,       # 6   (原4D + 方向向量2D)
            buff_feat,           # 5   (原3D + 方向向量2D)
            map_feat,            # 49  (修复：7×7窗口，原4×4=16D)
            np.array(legal_action, dtype=np.float32),  # 10
            progress_feat,       # 4
        ])  # total = 102

        # ── 奖励计算（简化版：仅保留核心信号）──────────────────────────
        # 
        # 问题诊断：原奖励设计存在致命缺陷
        # 1. 奖励项过多（11项）→ 噪声淹没信号
        # 2. 宝箱接近奖励累积200分 vs 收集仅5分 → 模型绕圈刷奖励
        # 3. 存活奖励累积40分 → 模型原地不动
        # 
        # 修复策略：简化为3个核心奖励
        # 1. 宝箱收集（稀疏、高价值） - 主要目标
        # 2. 危险惩罚（即时反馈） - 避免死亡
        # 3. 终局奖励（胜负信号） - 长期目标
        # ────────────────────────────────────────────────────────────────

        # === 核心1: 宝箱收集奖励（大幅提升） ===
        # 原值: 5.0 → 新值: 50.0（提升10倍，确保是主导信号）
        treasure_reward = 0.0
        try:
            treasures_remain = len(frame_state.get("treasures", []))
            if self.last_treasure_count == -1:
                self.last_treasure_count = treasures_remain
            elif treasures_remain < self.last_treasure_count:
                collected = self.last_treasure_count - treasures_remain
                treasure_reward = 50.0 * collected  # 提升至50分/个
                self.last_treasure_count = treasures_remain
            else:
                self.last_treasure_count = treasures_remain
        except Exception:
            pass

        # === 核心2: 危险惩罚（动态，仅在真正危险时触发） ===
        # 设计原则：距离越近，惩罚指数增长，给模型明确的学习信号
        risk_penalty = 0.0
        if cur_min_dist_norm < 0.15:  # 仅在距离<0.15（约27格）时惩罚
            # 指数惩罚：距离0.15→-0.5, 距离0.05→-5.0
            danger_level = (0.15 - cur_min_dist_norm) / 0.15
            risk_penalty = -0.5 * (danger_level ** 2) * 20  # 最高-5.0
        
        # === 核心3: 弱引导信号（帮助早期学习，但不足以主导策略） ===
        # 宝箱方向引导（仅在安全时）
        treasure_guide = 0.0
        if treasure_feat[2] > 0.0 and cur_min_dist_norm > 0.25:  # 安全时才引导
            # 向宝箱移动给小奖励，但远小于收集奖励
            cur_td = treasure_feat[2]
            if self.last_treasure_dist_norm >= 0.0:
                dist_delta = self.last_treasure_dist_norm - cur_td
                if dist_delta > 0:
                    treasure_guide = 0.1 * dist_delta  # 每步最多0.1分（收集=50分）
        self.last_treasure_dist_norm = treasure_feat[2] if treasure_feat[2] > 0.0 else -1.0

        # 更新状态
        self.last_min_monster_dist_norm = cur_min_dist_norm

        # === 奖励汇总 ===
        # 权重设计：确保收集宝箱 >> 引导信号 > 噪声
        total_reward = treasure_reward + risk_penalty + treasure_guide

        return feature, legal_action, [total_reward]
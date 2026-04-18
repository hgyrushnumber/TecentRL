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

        # ── 奖励设计（优先级分层 + Reward Scaling）──────────────────────
        #
        # 设计哲学：存活是第一要素，活着才有机会拿分；安全时才去收宝箱。
        #
        # 优先级层次（信号强度递减）：
        #   P1. 存活惩罚（危险靠近）  → 最强负信号，绝对优先
        #   P2. 存活奖励（持续活着）  → 持续正信号，鼓励不被抓
        #   P3. 宝箱收集              → 高价值稀疏奖励，安全前提下的目标
        #   P4. 宝箱方向引导          → 仅在安全时生效，辅助早期探索
        #
        # Reward Scaling 设计原则：
        #   所有奖励项经过统一缩放后送入 SAC，目标是将每步奖励控制在 [-1, 1] 附近，
        #   防止大幅奖励冲击（如宝箱+50）导致 Critic Q 值发散。
        #   缩放因子 REWARD_SCALE = 0.1，缩放后各项量级：
        #     - 危险惩罚: 最高 -0.5/步（P1，主导信号）
        #     - 存活奖励: +0.1/步（P2，一局1000步累积 +100 原始 → 缩放后 +10）
        #     - 宝箱收集: +5.0/个（P3，稀疏高价值）
        #     - 方向引导: 最高 ~+0.05/步（P4，辅助）
        # ────────────────────────────────────────────────────────────────
        REWARD_SCALE = 0.1   # 统一缩放因子，控制 Q 值量级

        # === P1: 危险惩罚（存活第一要素，越近惩罚越重）===
        # 阈值 0.30（约55格）：给模型足够的预警窗口
        # 指数惩罚确保越靠近惩罚越陡，强迫模型主动远离
        risk_penalty = 0.0
        if cur_min_dist_norm < 0.30:
            danger_level = (0.30 - cur_min_dist_norm) / 0.30  # [0, 1]
            # 平方指数：边界处约 -0.05，极近处约 -5.0（原始值）
            risk_penalty = -5.0 * (danger_level ** 2)

        # === P2: 存活奖励（活着本身就有价值）===
        # 每步给予正奖励，但仅在"相对安全"时给满额
        # 危险区（dist < 0.30）时减半，给模型「危险中存活价值更低」的信号
        if cur_min_dist_norm >= 0.30:
            survival_reward = 1.0   # 安全区：满额存活奖励
        else:
            survival_reward = 0.3   # 危险区：减半，避免危险区「赖着不走」

        # === P3: 宝箱收集奖励（安全前提下的主要得分目标）===
        treasure_reward = 0.0
        try:
            treasures_remain = len(frame_state.get("treasures", []))
            if self.last_treasure_count == -1:
                self.last_treasure_count = treasures_remain
            elif treasures_remain < self.last_treasure_count:
                collected = self.last_treasure_count - treasures_remain
                treasure_reward = 50.0 * collected   # 每个宝箱 +50（原始），缩放后 +5.0
                self.last_treasure_count = treasures_remain
            else:
                self.last_treasure_count = treasures_remain
        except Exception:
            pass

        # === P4: 宝箱方向引导（仅在安全区生效，不与P1冲突）===
        # 触发条件严格限定在安全区（dist > 0.30），彻底消除冲突区间
        treasure_guide = 0.0
        if treasure_feat[2] > 0.0 and cur_min_dist_norm > 0.30:
            cur_td = treasure_feat[2]
            if self.last_treasure_dist_norm >= 0.0:
                dist_delta = self.last_treasure_dist_norm - cur_td
                if dist_delta > 0:
                    # 原始约每步 +0.001~0.005，缩放后约 +0.0001~0.0005
                    # 乘以5.0提升引导强度，缩放后约 +0.0005~0.0025/步
                    treasure_guide = 5.0 * dist_delta
        self.last_treasure_dist_norm = treasure_feat[2] if treasure_feat[2] > 0.0 else -1.0

        # 更新状态
        self.last_min_monster_dist_norm = cur_min_dist_norm

        # === 奖励汇总 + Reward Scaling ===
        # 原始奖励各项量级：
        #   risk_penalty: [-5.0, 0]   survival: [0.3, 1.0]
        #   treasure: [0, 50]          guide: [0, ~0.025]
        # 统一乘以 REWARD_SCALE=0.1，缩放后：
        #   risk: [-0.5, 0]  survival: [0.03, 0.1]  treasure: [0, 5.0]  guide: [0, ~0.0025]
        raw_reward = risk_penalty + survival_reward + treasure_reward + treasure_guide
        total_reward = raw_reward * REWARD_SCALE

        return feature, legal_action, [total_reward]
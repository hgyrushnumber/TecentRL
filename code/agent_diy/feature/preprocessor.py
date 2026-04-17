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

        # ── 怪物特征 (9D × 2) ────────────────────────────────────────────
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
                ], dtype=np.float32))
            else:
                self.monster_prev_pos[i] = None
                monster_feats.append(np.array([0., 0., 0., 0., 1., 0., 0., 1., 0.], dtype=np.float32))

        # ── 宝箱特征 (4D) ────────────────────────────────────────────────
        treasure_feat = np.zeros(4, dtype=np.float32)
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
        except Exception:
            pass

        # ── Buff特征 (3D) ────────────────────────────────────────────────
        buff_feat = np.zeros(3, dtype=np.float32)
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
        except Exception:
            pass

        # ── 局部地图特征 (16D) ───────────────────────────────────────────
        map_feat = np.zeros(16, dtype=np.float32)
        if map_info is not None and len(map_info) >= 13:
            center = len(map_info) // 2
            flat_idx = 0
            for row in range(center - 2, center + 2):
                for col in range(center - 2, center + 2):
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

        # ── 拼接特征 (61D) ───────────────────────────────────────────────
        feature = np.concatenate([
            hero_feat,           # 6
            monster_feats[0],    # 9
            monster_feats[1],    # 9
            treasure_feat,       # 4
            buff_feat,           # 3
            map_feat,            # 16
            np.array(legal_action, dtype=np.float32),  # 10
            progress_feat,       # 4
        ])  # total = 61

        # ── 奖励计算 ─────────────────────────────────────────────────────
        progress_ratio = self.step_no / max(self.max_step, 1)

        # 1. 基础存活奖励（动态）
        survive_reward = 0.02 * (1.0 + progress_ratio)

        # 2. 远离怪物奖励（已移除：dist_delta 方差过大，导致模型学到无意义震荡走位）

        # 3. 碰撞风险惩罚（固定权重）
        risk_penalty = -0.03 * max_collision_risk_norm

        # 4. 里程碑奖励（每50步均匀间隔，线性递增 0.05~0.20，信号密集稳定）
        if self.step_no > 0 and self.step_no % 50 == 0:
            milestone_reward = 0.05 + 0.15 * (self.step_no / self.max_step)
        else:
            milestone_reward = 0.0

        # 5. 紧急避险奖励（已移除：模型会故意靠近怪物再逃跑来反复触发，导致策略不稳定）

        # 6. 宝箱收集奖励
        # 任务得分：1个宝箱=100分=约67步；1步存活=0.02奖励
        # 因此1个宝箱对应奖励 = 0.02 × 67 ≈ 1.34
        treasure_reward = 0.0
        try:
            cur_tc = int(round(treasure_feat[3] * 10))
            if self.last_treasure_count == -1:
                self.last_treasure_count = cur_tc
            elif cur_tc < self.last_treasure_count:
                treasure_reward = 1.34 * (self.last_treasure_count - cur_tc)
                self.last_treasure_count = cur_tc
        except Exception:
            pass

        # 7. 接近宝箱引导奖励
        # 放宽触发距离 0.3→0.6（覆盖全图约50%范围），安全条件适度放宽至>0.3
        # 奖励随距离线性衰减，越近越高
        treasure_proximity_reward = 0.0
        if treasure_feat[2] > 0.0 and cur_min_dist_norm > 0.3:
            # 距离越近奖励越高，0.6以外不给，0.6以内线性插值到0.08
            proximity_ratio = max(0.0, 1.0 - treasure_feat[2] / 0.6)
            treasure_proximity_reward = 0.08 * proximity_ratio

        # 8. Buff获取奖励（缩放到 0.2）
        buff_reward = 0.0
        cur_buff_active = buff_remain_norm > 0.01
        if cur_buff_active and not self.last_buff_active:
            buff_reward = 0.2
        self.last_buff_active = cur_buff_active

        # 9. 接近Buff奖励（安全条件：怪物距离>0.4时才引导，避免与risk_penalty冲突）
        buff_proximity_reward = 0.0
        if 0.0 < buff_feat[2] < 0.3 and cur_min_dist_norm > 0.4:
            buff_proximity_reward = 0.03 * (1.0 - buff_feat[2])

        # 10. 移动探索奖励
        movement_reward = 0.0
        if self.last_hero_pos is not None:
            lx, lz = self.last_hero_pos
            move_dist = np.sqrt((hx - lx) ** 2 + (hz - lz) ** 2)
            movement_reward = 0.02 * _norm(move_dist, 2.0) * (1.0 + progress_ratio * 0.5)
        self.last_hero_pos = (hx, hz)

        # 11. 技能使用奖励（缩放到 0.05~0.2，与存活奖励同量级）
        skill_reward = 0.0
        skill_mult = 1.0 + progress_ratio * 0.5
        if last_action == 8:  # 闪现
            if max_collision_risk_norm > 0.5:
                skill_reward = 0.15 * skill_mult
            elif max_collision_risk_norm > 0.3:
                skill_reward = 0.08 * skill_mult
            else:
                skill_reward = 0.02 * skill_mult
        elif last_action == 9:  # 天赋
            skill_reward = 0.10 if cur_min_dist_norm < 0.5 else 0.05

        # 更新状态
        self.last_min_monster_dist_norm = cur_min_dist_norm

        total_reward = (
            survive_reward
            + risk_penalty
            + milestone_reward
            + treasure_reward
            + treasure_proximity_reward
            + buff_reward
            + buff_proximity_reward
            + movement_reward
            + skill_reward
        )

        return feature, legal_action, [total_reward]
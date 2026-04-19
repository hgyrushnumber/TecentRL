#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Feature preprocessor and reward design for Gorge Chase SAC.
峡谷追猎 SAC 特征预处理与奖励设计。
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


def _norm_signed(v, abs_max):
    """Normalize signed value to [0, 1] via [-abs_max, abs_max]."""
    return _norm(v, abs_max, -abs_max)


class Preprocessor:
    def __init__(self):
        self.reset()

    def reset(self):
        self.step_no = 0
        self.max_step = 1000

        self.last_min_monster_dist_norm = 0.5
        self.last_action = -1
        self.last_flash_cd = 0

        # 怪物历史位置（用于相对速度计算）：list of [pos_x, pos_z] or None
        self.monster_prev_pos = [None, None]
        self.last_hero_pos = None

        self.last_treasure_count = -1   # -1 表示未初始化
        self.last_treasure_dist_norm = -1.0  # -1 表示未初始化，用于差分

    def feature_process(self, env_obs, last_action):
        """Process env_obs into feature vector, legal_action mask, and reward."""
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
            1.0 if flash_cd == 0 else 0.0,
            1.0 if talent_cd == 0 else 0.0,
        ], dtype=np.float32)

        # ── 怪物特征 (11D × 2) ───────────────────────────────────────────
        monsters = frame_state.get("monsters", [])
        monster_feats = []
        cur_min_dist_norm = 1.0
        max_collision_risk_norm = 0.0
        eta_list = [1.0, 1.0]

        for i in range(2):
            if i < len(monsters) and monsters[i].get("is_in_view", 0):
                m = monsters[i]
                mx = m["pos"]["x"]
                mz = m["pos"]["z"]
                spd = m.get("speed", 1)

                raw_dist = np.sqrt((hx - mx) ** 2 + (hz - mz) ** 2)
                dist_norm = _norm(raw_dist, MAX_DIST)
                cur_min_dist_norm = min(cur_min_dist_norm, dist_norm)

                if self.monster_prev_pos[i] is not None:
                    pvx, pvz = self.monster_prev_pos[i]
                    vel_x = mx - pvx
                    vel_z = mz - pvz
                else:
                    vel_x = vel_z = 0.0
                self.monster_prev_pos[i] = (mx, mz)

                pred_x = np.clip(mx + vel_x * 3, 0, MAP_SIZE)
                pred_z = np.clip(mz + vel_z * 3, 0, MAP_SIZE)
                pred_dist_norm = _norm(np.sqrt((hx - pred_x) ** 2 + (hz - pred_z) ** 2), MAX_DIST)

                cr_norm = _norm(spd / max(raw_dist, 1.0), MAX_COLLISION_RISK)
                max_collision_risk_norm = max(max_collision_risk_norm, cr_norm)

                angle = np.arctan2(mz - hz, mx - hx)
                dir_sin = (np.sin(angle) + 1.0) * 0.5
                dir_cos = (np.cos(angle) + 1.0) * 0.5

                # ETA（怪物到达英雄所需步数，归一化到[0,1]，越小越危险）
                eta = raw_dist / max(float(spd), 1e-6)
                eta_list[i] = _norm(eta, 80.0)

                monster_feats.append(np.array([
                    1.0,
                    _norm(mx, MAP_SIZE),
                    _norm(mz, MAP_SIZE),
                    _norm(spd, MAX_MONSTER_SPEED),
                    dist_norm,
                    _norm_signed(vel_x, MAX_REL_VELOCITY),
                    _norm_signed(vel_z, MAX_REL_VELOCITY),
                    pred_dist_norm,
                    cr_norm,
                    dir_sin,
                    dir_cos,
                ], dtype=np.float32))
            else:
                if self.monster_prev_pos[i] is not None:
                    est_mx, est_mz = self.monster_prev_pos[i]
                    est_raw_dist = np.sqrt((hx - est_mx) ** 2 + (hz - est_mz) ** 2)
                    est_dist_norm = _norm(est_raw_dist, MAX_DIST)
                    cur_min_dist_norm = min(cur_min_dist_norm, est_dist_norm)
                    eta_list[i] = _norm(est_raw_dist, 80.0)
                    monster_feats.append(np.array([
                        0., 0., 0., 0.,
                        est_dist_norm,
                        0.5, 0.5,
                        est_dist_norm,
                        0., 0.5, 0.5,
                    ], dtype=np.float32))
                else:
                    monster_feats.append(np.array([0., 0., 0., 0., 1., 0.5, 0.5, 1., 0., 0.5, 0.5], dtype=np.float32))

        # ── 宝箱特征 (6D) ────────────────────────────────────────────────
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
                    dx = (tx - hx) / max(min_td, 1e-6)
                    dz = (tz - hz) / max(min_td, 1e-6)
                    treasure_feat[4] = (dx + 1.0) * 0.5
                    treasure_feat[5] = (dz + 1.0) * 0.5
        except Exception:
            pass

        # ── Buff特征 (5D) ────────────────────────────────────────────────
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
                    dx = (bx - hx) / max(min_bd, 1e-6)
                    dz = (bz - hz) / max(min_bd, 1e-6)
                    buff_feat[3] = (dx + 1.0) * 0.5
                    buff_feat[4] = (dz + 1.0) * 0.5
        except Exception:
            pass

        # ── 局部地图特征 (49D) ───────────────────────────────────────────
        map_feat = np.zeros(49, dtype=np.float32)
        center = 0
        if map_info is not None and len(map_info) >= 7:
            center = len(map_info) // 2
            flat_idx = 0
            for row in range(center - 3, center + 4):
                for col in range(center - 3, center + 4):
                    if 0 <= row < len(map_info) and 0 <= col < len(map_info[0]):
                        map_feat[flat_idx] = float(map_info[row][col] != 0)
                    flat_idx += 1

        # ── 合法动作掩码 (16D) ──────────────────────────────────────────
        legal_action = [1] * 16
        if isinstance(legal_act_raw, list) and legal_act_raw:
            if isinstance(legal_act_raw[0], bool):
                for j in range(min(16, len(legal_act_raw))):
                    legal_action[j] = int(legal_act_raw[j])
            else:
                valid_set = {int(a) for a in legal_act_raw if 0 <= int(a) < 16}
                legal_action = [1 if j in valid_set else 0 for j in range(16)]
        else:
            # 回退逻辑：移动恒可用，闪现由CD控制
            for j in range(8, 16):
                legal_action[j] = 1 if flash_cd == 0 else 0

        if sum(legal_action) == 0:
            legal_action = [1] * 8 + [1 if flash_cd == 0 else 0] * 8

        # ── 规划特征 (10D) ───────────────────────────────────────────────
        step_norm = _norm(self.step_no, self.max_step)
        eta_min = min(eta_list)

        # 逃逸方向比：在8邻域中可通行方向占比
        escape_ratio = 1.0
        local_block_ratio = 0.0
        corridor_len_norm = 0.0
        if map_info is not None and len(map_info) > 0:
            h = len(map_info)
            w = len(map_info[0]) if h > 0 else 0
            c = center if center > 0 else h // 2
            dirs = [(0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1)]
            open_cnt = 0
            corridor_lengths = []
            for dr, dc in dirs:
                nr, nc = c + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and map_info[nr][nc] != 0:
                    open_cnt += 1
                    # 该方向最大直行通路长度（最多5格）
                    length = 0
                    for k in range(1, 6):
                        rr, cc = c + dr * k, c + dc * k
                        if 0 <= rr < h and 0 <= cc < w and map_info[rr][cc] != 0:
                            length += 1
                        else:
                            break
                    corridor_lengths.append(length)
            escape_ratio = open_cnt / 8.0
            local_block_ratio = 1.0 - float(np.mean(map_feat))
            corridor_len_norm = _norm(max(corridor_lengths) if corridor_lengths else 0, 5.0)

        flash_ready = 1.0 if flash_cd == 0 else 0.0
        post_half = 1.0 if self.step_no > self.max_step * 0.5 else 0.0

        planning_feat = np.array([
            step_norm,
            cur_min_dist_norm,
            max_collision_risk_norm,
            post_half,
            eta_list[0],
            eta_list[1],
            eta_min,
            escape_ratio,
            local_block_ratio,
            corridor_len_norm * flash_ready,
        ], dtype=np.float32)

        # ── 拼接特征 (114D) ──────────────────────────────────────────────
        # 6 + 11 + 11 + 6 + 5 + 49 + 16 + 10 = 114
        feature = np.concatenate([
            hero_feat,
            monster_feats[0],
            monster_feats[1],
            treasure_feat,
            buff_feat,
            map_feat,
            np.array(legal_action, dtype=np.float32),
            planning_feat,
        ])

        # ── 奖励设计：生存优先策略（简化版）─────────────────────────────────
        terminated = bool(env_obs.get("terminated", False))

        # 1) 生存奖励（与游戏规则对齐：每步1.5分）
        survival_reward = 1.5

        # 2) 宝箱收集奖励（与游戏规则对齐：每个宝箱100分）
        treasure_event_reward = 0.0
        try:
            treasures_remain = len(frame_state.get("treasures", []))
            if self.last_treasure_count == -1:
                self.last_treasure_count = treasures_remain
            elif treasures_remain < self.last_treasure_count:
                collected = self.last_treasure_count - treasures_remain
                treasure_event_reward = 100.0 * collected  # 每个宝箱100分（游戏规则）
                self.last_treasure_count = treasures_remain
            else:
                self.last_treasure_count = treasures_remain
        except Exception:
            pass

        # 3) 风险惩罚（适度，鼓励合理冒险）
        risk_penalty = 0.0
        if cur_min_dist_norm < 0.05:  # 缩小危险检测范围
            danger = (0.05 - cur_min_dist_norm) / 0.05
            risk_penalty = -2.0 * danger  # 降低惩罚力度

        # 4) 终局惩罚（大幅强化，让智能体真正害怕被捕获）
        terminal_penalty = -50.0 if terminated else 0.0

        # 5) 移除复杂奖励信号（PBRS、闪现质量等），简化学习目标

        total_reward = (
            survival_reward
            + treasure_event_reward
            + risk_penalty
            + terminal_penalty
        )

        # 更新历史
        self.last_min_monster_dist_norm = cur_min_dist_norm
        self.last_treasure_dist_norm = treasure_feat[2] if treasure_feat[2] > 0.0 else -1.0
        self.last_hero_pos = (hx, hz)
        self.last_action = int(last_action) if last_action is not None else -1
        self.last_flash_cd = flash_cd

        return feature, legal_action, [float(total_reward)]

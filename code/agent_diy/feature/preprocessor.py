#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Feature preprocessor and reward design for Gorge Chase SAC.
峡谷追猎 SAC 特征预处理与奖励设计（多通道 21x21 空间特征版）。

空间特征通道定义（6 x 21 x 21）:
  0: passable      可通行层
  1: out_of_bound  越界层
  2: hero          英雄层
  3: monster       怪物层
  4: treasure      宝箱层
  5: buff          Buff层

最终特征维度:
  标量特征:
    hero(6)
    monster1(11)
    monster2(11)
    treasure(6)
    buff(5)
    legal_action(16)
    planning(10)
    => 65
  空间特征:
    6 * 21 * 21 = 2646

  总维度:
    65 + 2646 = 2711
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

# Map diagonal / 地图对角线最大距离
MAX_DIST = MAP_SIZE * 1.41

# Spatial map config
LOCAL_MAP_SIZE = 21
LOCAL_MAP_RADIUS = 10

# Spatial channels
CH_PASSABLE = 0
CH_OOB = 1
CH_HERO = 2
CH_MONSTER = 3
CH_TREASURE = 4
CH_BUFF = 5
SPATIAL_CHANNELS = 6


def _norm(v, v_max, v_min=0.0):
    v = float(np.clip(v, v_min, v_max))
    return (v - v_min) / (v_max - v_min) if (v_max - v_min) > 1e-6 else 0.0


def _norm_signed(v, abs_max):
    v = float(np.clip(v, -abs_max, abs_max))
    return (v + abs_max) / (2.0 * abs_max) if abs_max > 1e-6 else 0.5


class Preprocessor:
    def __init__(self):
        self.reset()

    def reset(self):
        self.step_no = 0
        self.max_step = 200

        self.last_min_monster_dist_norm = 0.5

        # 怪物上一帧位置，用于估计速度
        self.monster_prev_pos = [None, None]

        # 历史状态
        self.last_treasure_count = -1
        self.last_treasure_dist_norm = -1.0
        self.last_hero_pos = None
        self.last_action = -1
        self.last_flash_cd = 0

    def _build_spatial_feat(self, map_info, frame_state, hx, hz):
        """
        构建 6x21x21 多通道空间特征。
        默认假设：
          1. map_info 是以英雄为中心的局部视野
          2. 世界坐标 (x, z) 与局部格子近似一一对应（通过相对位移 round 投影）
        如果环境真实映射不同，只需修改 _project_to_local().
        """
        spatial_feat = np.zeros((SPATIAL_CHANNELS, LOCAL_MAP_SIZE, LOCAL_MAP_SIZE), dtype=np.float32)

        # 1) passable / out_of_bound
        if map_info is not None and len(map_info) > 0 and len(map_info[0]) > 0:
            h = len(map_info)
            w = len(map_info[0])

            center_r = h // 2
            center_c = w // 2

            for i, row in enumerate(range(center_r - LOCAL_MAP_RADIUS, center_r + LOCAL_MAP_RADIUS + 1)):
                for j, col in enumerate(range(center_c - LOCAL_MAP_RADIUS, center_c + LOCAL_MAP_RADIUS + 1)):
                    if 0 <= row < h and 0 <= col < w:
                        spatial_feat[CH_PASSABLE, i, j] = float(map_info[row][col] != 0)
                        spatial_feat[CH_OOB, i, j] = 0.0
                    else:
                        spatial_feat[CH_PASSABLE, i, j] = 0.0
                        spatial_feat[CH_OOB, i, j] = 1.0
        else:
            # 没有 map_info 时，全部按越界/不可通行处理
            spatial_feat[CH_PASSABLE, :, :] = 0.0
            spatial_feat[CH_OOB, :, :] = 1.0

        # 2) hero
        spatial_feat[CH_HERO, LOCAL_MAP_RADIUS, LOCAL_MAP_RADIUS] = 1.0

        def _project_to_local(obj_x, obj_z):
            """
            将世界坐标投影到以英雄为中心的 21x21 局部图。
            默认:
              dx > 0 => 右侧
              dz > 0 => 下方
            若你的环境行列方向与此相反，只需改这里。
            """
            dx = int(round(obj_x - hx))
            dz = int(round(obj_z - hz))

            rr = LOCAL_MAP_RADIUS + dz
            cc = LOCAL_MAP_RADIUS + dx
            return rr, cc

        # 3) monster
        for m in frame_state.get("monsters", []):
            if not isinstance(m, dict):
                continue
            if not m.get("is_in_view", 0):
                continue
            if "pos" not in m:
                continue

            mx = m["pos"]["x"]
            mz = m["pos"]["z"]

            rr, cc = _project_to_local(mx, mz)
            if 0 <= rr < LOCAL_MAP_SIZE and 0 <= cc < LOCAL_MAP_SIZE:
                spatial_feat[CH_MONSTER, rr, cc] = min(
                    1.0, spatial_feat[CH_MONSTER, rr, cc] + 1.0
                )

        # 4) treasure
        for t in frame_state.get("treasures", []):
            if not isinstance(t, dict) or "pos" not in t:
                continue

            tx = t["pos"]["x"]
            tz = t["pos"]["z"]

            rr, cc = _project_to_local(tx, tz)
            if 0 <= rr < LOCAL_MAP_SIZE and 0 <= cc < LOCAL_MAP_SIZE:
                spatial_feat[CH_TREASURE, rr, cc] = 1.0

        # 5) buff
        for b in frame_state.get("buffs", []):
            if not isinstance(b, dict) or "pos" not in b:
                continue

            bx = b["pos"]["x"]
            bz = b["pos"]["z"]

            rr, cc = _project_to_local(bx, bz)
            if 0 <= rr < LOCAL_MAP_SIZE and 0 <= cc < LOCAL_MAP_SIZE:
                spatial_feat[CH_BUFF, rr, cc] = 1.0

        return spatial_feat

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

                # ETA（怪物到达英雄所需步数，越小越危险）
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
                    monster_feats.append(
                        np.array([0., 0., 0., 0., 1., 0.5, 0.5, 1., 0., 0.5, 0.5], dtype=np.float32)
                    )

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

        # ── 多通道空间特征 (6 x 21 x 21 = 2646D) ───────────────────────
        spatial_feat = self._build_spatial_feat(map_info, frame_state, hx, hz)
        spatial_flat = spatial_feat.reshape(-1).astype(np.float32)

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

        # 用 passable 层替代旧 7x7 map_feat 的局部统计
        passable_map = spatial_feat[CH_PASSABLE]
        center = LOCAL_MAP_RADIUS

        escape_ratio = 1.0
        local_block_ratio = 0.0
        corridor_len_norm = 0.0

        dirs = [(0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1)]
        open_cnt = 0
        corridor_lengths = []

        for dr, dc in dirs:
            nr, nc = center + dr, center + dc
            if 0 <= nr < LOCAL_MAP_SIZE and 0 <= nc < LOCAL_MAP_SIZE and passable_map[nr, nc] > 0.5:
                open_cnt += 1
                length = 0
                for k in range(1, 6):
                    rr, cc = center + dr * k, center + dc * k
                    if 0 <= rr < LOCAL_MAP_SIZE and 0 <= cc < LOCAL_MAP_SIZE and passable_map[rr, cc] > 0.5:
                        length += 1
                    else:
                        break
                corridor_lengths.append(length)

        escape_ratio = open_cnt / 8.0
        local_block_ratio = 1.0 - float(np.mean(passable_map))
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

        # ── 拼接特征 (2711D) ────────────────────────────────────────────
        # hero(6) + monster(22) + treasure(6) + buff(5)
        # + spatial(2646) + legal(16) + planning(10) = 2711
        feature = np.concatenate([
            hero_feat,
            monster_feats[0],
            monster_feats[1],
            treasure_feat,
            buff_feat,
            spatial_flat,
            np.array(legal_action, dtype=np.float32),
            planning_feat,
        ]).astype(np.float32)

        # ── 奖励设计：生存优先策略（简化版）──────────────────────────────
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
                treasure_event_reward = 100.0 * collected
                self.last_treasure_count = treasures_remain
            else:
                self.last_treasure_count = treasures_remain
        except Exception:
            pass

        # 3) 风险惩罚
        risk_penalty = 0.0
        if cur_min_dist_norm < 0.05:
            danger = (0.05 - cur_min_dist_norm) / 0.05
            risk_penalty = -2.0 * danger

        # 4) 终局惩罚
        terminal_penalty = -50.0 if terminated else 0.0

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
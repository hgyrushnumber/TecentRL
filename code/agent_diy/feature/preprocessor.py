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

from collections import deque

import numpy as np
from agent_diy.conf.conf import Config

# Map size / 地图尺寸（128×128）
MAP_SIZE = 128.0
# Max monster speed / 最大怪物速度
MAX_MONSTER_SPEED = 5.0
# Max distance bucket / 距离桶最大值
MAX_DIST_BUCKET = 5.0
# Max flash cooldown / 最大闪现冷却步数
MAX_FLASH_CD = 2000.0
# Max buff duration / buff最大持续时间
MAX_BUFF_DURATION = 50.0
# Local map window size / 局部地图窗口边长
LOCAL_MAP_WINDOW = Config.LOCAL_MAP_WINDOW
MAP_CHANNELS = Config.MAP_CHANNELS


def _norm(v, v_max, v_min=0.0):
    """Normalize value to [0, 1].

    将值归一化到 [0, 1]。
    """
    v = float(np.clip(v, v_min, v_max))
    return (v - v_min) / (v_max - v_min) if (v_max - v_min) > 1e-6 else 0.0


class Preprocessor:
    def __init__(self):
        self.reset()

    def reset(self):
        self.step_no = 0
        self.max_step = 200
        self.last_min_monster_dist_norm = 0.5
        self.last_min_treasure_dist_norm = 1.0
        self.last_step_score = 0.0
        self.last_treasure_score = 0.0
        self.last_buff_count = 0.0
        self.last_hero_pos = None
        self.last_flash_cd = 0.0
        self.visit_counter = {}
        self.recent_positions = deque(maxlen=20)

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
        self.max_step = env_info.get("max_step", 200)

        # Hero self features (4D) / 英雄自身特征
        hero = frame_state["heroes"]
        hero_pos = hero["pos"]
        hero_x_norm = _norm(hero_pos["x"], MAP_SIZE)
        hero_z_norm = _norm(hero_pos["z"], MAP_SIZE)
        flash_cd_norm = _norm(hero["flash_cooldown"], MAX_FLASH_CD)
        buff_remain_norm = _norm(hero["buff_remaining_time"], MAX_BUFF_DURATION)

        hero_feat = np.array([hero_x_norm, hero_z_norm, flash_cd_norm, buff_remain_norm], dtype=np.float32)

        # Monster features (5D x 2) / 怪物特征
        monsters = frame_state.get("monsters", [])
        monster_feats = []
        for i in range(2):
            if i < len(monsters):
                m = monsters[i]
                is_in_view = float(m.get("is_in_view", 0))
                m_pos = m["pos"]
                if is_in_view:
                    m_x_norm = _norm(m_pos["x"], MAP_SIZE)
                    m_z_norm = _norm(m_pos["z"], MAP_SIZE)
                    m_speed_norm = _norm(m.get("speed", 1), MAX_MONSTER_SPEED)

                    # Euclidean distance / 欧式距离
                    raw_dist = np.sqrt((hero_pos["x"] - m_pos["x"]) ** 2 + (hero_pos["z"] - m_pos["z"]) ** 2)
                    dist_norm = _norm(raw_dist, MAP_SIZE * 1.41)
                else:
                    m_x_norm = 0.0
                    m_z_norm = 0.0
                    m_speed_norm = 0.0
                    dist_norm = 1.0
                monster_feats.append(
                    np.array([is_in_view, m_x_norm, m_z_norm, m_speed_norm, dist_norm], dtype=np.float32)
                )
            else:
                monster_feats.append(np.zeros(5, dtype=np.float32))

        # Out-of-vision monster relative info (2 x [dx, dz, dist]) / 视野外怪物相对信息
        rel_monster_feat = []
        for i in range(2):
            if i < len(monsters):
                rel_monster_feat.extend(self._extract_monster_relative(monsters[i], hero_pos))
            else:
                rel_monster_feat.extend([0.0, 0.0, 1.0])
        rel_monster_feat = np.array(rel_monster_feat, dtype=np.float32)

        # Spatial map features (C x 21 x 21) / 空间特征图（多通道）
        # channel 0: hero, 1: monster, 2: treasure, 3: obstacle
        map_tensor = np.zeros((MAP_CHANNELS, LOCAL_MAP_WINDOW, LOCAL_MAP_WINDOW), dtype=np.float32)
        center = LOCAL_MAP_WINDOW // 2
        map_tensor[0, center, center] = 1.0

        # Place monsters on monster channel / 将怪物投影到怪物通道
        for m in monsters:
            if float(m.get("is_in_view", 0)) <= 0:
                continue
            m_pos = m.get("pos", {})
            self._place_entity(
                map_tensor[1],
                hero_pos.get("x", 0.0),
                hero_pos.get("z", 0.0),
                m_pos.get("x", 0.0),
                m_pos.get("z", 0.0),
            )

        # Place treasures on treasure channel / 将宝箱投影到宝箱通道
        treasure_list = frame_state.get("treasures", frame_state.get("treasure", []))
        if isinstance(treasure_list, dict):
            treasure_list = [treasure_list]
        for t in treasure_list:
            t_pos = t.get("pos", {}) if isinstance(t, dict) else {}
            self._place_entity(
                map_tensor[2],
                hero_pos.get("x", 0.0),
                hero_pos.get("z", 0.0),
                t_pos.get("x", 0.0),
                t_pos.get("z", 0.0),
            )

        # Build obstacle channel from local map occupancy / 障碍物通道
        # map_info 定义：1=可通行，0=障碍物
        obstacle_channel = np.zeros((LOCAL_MAP_WINDOW, LOCAL_MAP_WINDOW), dtype=np.float32)
        if map_info is not None and len(map_info) >= LOCAL_MAP_WINDOW:
            radius = LOCAL_MAP_WINDOW // 2
            for row in range(center - radius, center + radius + 1):
                for col in range(center - radius, center + radius + 1):
                    rr = row - (center - radius)
                    cc = col - (center - radius)
                    if 0 <= row < len(map_info) and 0 <= col < len(map_info[0]):
                        obstacle_channel[rr, cc] = float(map_info[row][col] == 0)
        map_tensor[3] = obstacle_channel
        map_feat = map_tensor.reshape(-1)

        # Legal action mask (8D) / 合法动作掩码
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

        # Progress features (2D) / 进度特征
        step_norm = _norm(self.step_no, self.max_step)
        survival_ratio = step_norm
        progress_feat = np.array([step_norm, survival_ratio], dtype=np.float32)

        # Concatenate features / 拼接特征
        feature = np.concatenate(
            [
                hero_feat,
                monster_feats[0],
                monster_feats[1],
                rel_monster_feat,
                map_feat,
                np.array(legal_action, dtype=np.float32),
                progress_feat,
            ]
        )

        # Step reward / 即时奖励
        cur_min_dist_norm = 1.0
        cur_second_dist_norm = 1.0
        visible_monster_dists = []
        for m_feat in monster_feats:
            if m_feat[0] > 0:
                cur_min_dist_norm = min(cur_min_dist_norm, m_feat[4])
                visible_monster_dists.append(m_feat[4])
        if len(visible_monster_dists) >= 2:
            cur_second_dist_norm = sorted(visible_monster_dists)[1]

        cur_min_treasure_dist_norm = 1.0
        for t in treasure_list:
            if not isinstance(t, dict):
                continue
            t_pos = t.get("pos", {})
            tx, tz = t_pos.get("x"), t_pos.get("z")
            if tx is None or tz is None:
                continue
            raw_dist = np.sqrt((hero_pos["x"] - tx) ** 2 + (hero_pos["z"] - tz) ** 2)
            cur_min_treasure_dist_norm = min(cur_min_treasure_dist_norm, _norm(raw_dist, MAP_SIZE * 1.41))

        # Score-like signals / 分数型信号
        step_score = float(env_info.get("step_score", env_info.get("score", self.last_step_score)))
        treasure_score = float(env_info.get("treasure_score", self.last_treasure_score))
        buff_count = float(env_info.get("buff_count", env_info.get("buff_num", self.last_buff_count)))

        # Dense rewards / 稠密奖励
        survive_reward = 0.01
        step_score_reward = 0.02 if step_score > self.last_step_score else 0.0
        dist_shaping = 0.12 * (cur_min_dist_norm - self.last_min_monster_dist_norm)
        treasure_approach_reward = 0.06 * (self.last_min_treasure_dist_norm - cur_min_treasure_dist_norm)

        # Speed-up stage shaping / 怪物加速前后强化
        monster_speedup_step = float(env_info.get("monster_speedup", 500))
        near_speedup_ratio = _norm(self.step_no, max(monster_speedup_step, 1.0))
        near_speedup_bonus = 0.04 * near_speedup_ratio * max(0.0, cur_min_dist_norm - 0.25)
        late_survival_bonus = 0.06 * max(0.0, near_speedup_ratio - 0.6) * max(0.0, cur_min_dist_norm - 0.2)

        # Corridor openness reward / corridor开阔奖励
        corridor_reward = 0.03 * self._compute_openness(obstacle_channel)

        # Sparse rewards / 稀疏奖励
        treasure_score_reward = 0.2 if treasure_score > self.last_treasure_score else 0.0
        buff_reward = 0.15 if buff_count > self.last_buff_count else 0.0

        # Risk penalties / 风险惩罚
        danger_penalty = -0.18 * max(0.0, 0.25 - cur_min_dist_norm)
        second_monster_penalty = -0.08 * max(0.0, 0.28 - cur_second_dist_norm)
        corner_penalty = -0.08 * max(0.0, 0.2 - self._compute_openness(obstacle_channel))
        encircle_penalty = -0.06 * self._compute_encircle_penalty(hero_pos, monsters)

        # Movement penalties / 移动相关惩罚
        invalid_move_penalty = self._compute_invalid_move_penalty(hero_pos)
        repeat_explore_penalty = self._compute_repeat_penalty(hero_pos)

        # Flash-related reward / 闪现相关奖励
        flash_escape_reward, flash_abuse_penalty = self._compute_flash_reward(
            hero.get("flash_cooldown", 0.0), cur_min_dist_norm, obstacle_channel
        )

        reward_value = (
            survive_reward
            + step_score_reward
            + treasure_score_reward
            + buff_reward
            + treasure_approach_reward
            + dist_shaping
            + near_speedup_bonus
            + late_survival_bonus
            + corridor_reward
            + encircle_penalty
            + corner_penalty
            + danger_penalty
            + invalid_move_penalty
            + repeat_explore_penalty
            + flash_escape_reward
            + flash_abuse_penalty
            + second_monster_penalty
        )
        reward_value = float(np.clip(reward_value, -1.0, 1.0))

        self.last_min_monster_dist_norm = cur_min_dist_norm
        self.last_min_treasure_dist_norm = cur_min_treasure_dist_norm
        self.last_step_score = step_score
        self.last_treasure_score = treasure_score
        self.last_buff_count = buff_count
        self.last_flash_cd = float(hero.get("flash_cooldown", 0.0))
        self.last_hero_pos = (float(hero_pos.get("x", 0.0)), float(hero_pos.get("z", 0.0)))

        reward = [reward_value]

        return feature, legal_action, reward

    def _place_entity(self, channel, hero_x, hero_z, obj_x, obj_z):
        radius = LOCAL_MAP_WINDOW // 2
        dx = float(obj_x) - float(hero_x)
        dz = float(obj_z) - float(hero_z)
        col = int(np.round((dx / MAP_SIZE) * (LOCAL_MAP_WINDOW - 1))) + radius
        row = int(np.round((dz / MAP_SIZE) * (LOCAL_MAP_WINDOW - 1))) + radius
        if 0 <= row < LOCAL_MAP_WINDOW and 0 <= col < LOCAL_MAP_WINDOW:
            channel[row, col] = 1.0

    def _compute_openness(self, obstacle_channel):
        passable = 1.0 - obstacle_channel
        center = LOCAL_MAP_WINDOW // 2
        inner = passable[max(0, center - 2) : center + 3, max(0, center - 2) : center + 3]
        mid = passable[max(0, center - 4) : center + 5, max(0, center - 4) : center + 5]
        inner_ratio = float(np.mean(inner)) if inner.size > 0 else 0.0
        mid_ratio = float(np.mean(mid)) if mid.size > 0 else 0.0
        return 0.6 * inner_ratio + 0.4 * mid_ratio

    def _compute_encircle_penalty(self, hero_pos, monsters):
        visible = []
        hx, hz = float(hero_pos.get("x", 0.0)), float(hero_pos.get("z", 0.0))
        for m in monsters:
            if float(m.get("is_in_view", 0)) <= 0:
                continue
            mp = m.get("pos", {})
            dx = float(mp.get("x", 0.0)) - hx
            dz = float(mp.get("z", 0.0)) - hz
            dist = np.sqrt(dx * dx + dz * dz) + 1e-6
            visible.append((dx / dist, dz / dist, dist))
        if len(visible) < 2:
            return 0.0
        dot = visible[0][0] * visible[1][0] + visible[0][1] * visible[1][1]
        close_factor = max(0.0, 1.0 - min(visible[0][2], visible[1][2]) / (MAP_SIZE * 0.35))
        return max(0.0, -dot) * close_factor

    def _compute_invalid_move_penalty(self, hero_pos):
        hx, hz = float(hero_pos.get("x", 0.0)), float(hero_pos.get("z", 0.0))
        if self.last_hero_pos is None:
            return 0.0
        dx = hx - self.last_hero_pos[0]
        dz = hz - self.last_hero_pos[1]
        disp = np.sqrt(dx * dx + dz * dz)
        return -0.08 if disp < 0.2 else 0.0

    def _compute_repeat_penalty(self, hero_pos):
        hx, hz = float(hero_pos.get("x", 0.0)), float(hero_pos.get("z", 0.0))
        cell = (int(hx // 4), int(hz // 4))
        self.visit_counter[cell] = self.visit_counter.get(cell, 0) + 1
        self.recent_positions.append(cell)
        unique_ratio = len(set(self.recent_positions)) / max(1, len(self.recent_positions))
        revisit = self.visit_counter[cell]
        return -0.04 * max(0.0, revisit / 20.0) - 0.05 * max(0.0, 0.5 - unique_ratio)

    def _compute_flash_reward(self, flash_cd, cur_min_dist_norm, obstacle_channel):
        flash_cd = float(flash_cd)
        flashed = flash_cd > self.last_flash_cd + 100.0
        if not flashed:
            return 0.0, 0.0
        openness = self._compute_openness(obstacle_channel)
        escaped = cur_min_dist_norm - self.last_min_monster_dist_norm
        if escaped > 0.05 or openness > 0.6:
            return 0.2, 0.0
        return 0.0, -0.15

    def _extract_monster_relative(self, monster, hero_pos):
        rel = monster.get("relative_pos", {}) if isinstance(monster, dict) else {}
        dx = rel.get("x", rel.get("dx", monster.get("relative_x", None)))
        dz = rel.get("z", rel.get("dz", monster.get("relative_z", None)))
        dist = rel.get("dist", rel.get("distance", monster.get("relative_distance", None)))

        if dx is None or dz is None:
            m_pos = monster.get("pos", {}) if isinstance(monster, dict) else {}
            if "x" in m_pos and "z" in m_pos:
                dx = float(m_pos["x"]) - float(hero_pos.get("x", 0.0))
                dz = float(m_pos["z"]) - float(hero_pos.get("z", 0.0))
            else:
                dx, dz = 0.0, 0.0

        if dist is None:
            dist = np.sqrt(float(dx) * float(dx) + float(dz) * float(dz))

        dx_norm = float(np.clip(float(dx) / MAP_SIZE, -1.0, 1.0))
        dz_norm = float(np.clip(float(dz) / MAP_SIZE, -1.0, 1.0))
        dist_norm = _norm(float(dist), MAP_SIZE * 1.41)
        return [dx_norm, dz_norm, dist_norm]

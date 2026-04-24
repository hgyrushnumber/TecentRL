#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Process env_obs into feature vector and legal action mask.

注意：
1. feature_process 只处理观测特征和合法动作。
2. reward 统一由 compute_reward_from_transition(prev_obs, curr_obs, action) 计算。
3. CNN 局部地图使用 4 通道：
   channel 0: obstacle
   channel 1: treasure
   channel 2: monster
   channel 3: buff
"""
import numpy as np
from agent_diy.conf.conf import Config


MAP_SIZE = 128.0
MAX_MONSTER_SPEED = 5.0
MAX_DIST_BUCKET = 5.0
MAX_FLASH_CD = 2000.0
MAX_BUFF_DURATION = 50.0

LOCAL_MAP_WINDOW = Config.LOCAL_MAP_WINDOW
MAP_CHANNELS = Config.MAP_CHANNELS


def _norm(v, v_max, v_min=0.0):
    """Normalize value to [0, 1]."""
    v = float(np.clip(v, v_min, v_max))
    return (v - v_min) / (v_max - v_min) if (v_max - v_min) > 1e-6 else 0.0


class Preprocessor:
    def __init__(self):
        self.reset()

    def reset(self):
        self.step_no = 0
        self.max_step = 200

        self.visit_counter = {}

        self.last_reward_components = {}
        self.last_env_info = {}

    def feature_process(self, env_obs, last_action):
        """
        Process env_obs into feature vector and legal action mask.
        """
        observation = env_obs["observation"]
        frame_state = observation["frame_state"]
        env_info = observation["env_info"]
        map_info = observation["map_info"]
        legal_act_raw = observation["legal_action"]

        self.step_no = observation["step_no"]
        self.max_step = env_info.get("max_step", 200)

        # ------------------------------------------------------------------
        # 1) Hero self features
        # ------------------------------------------------------------------
        hero = frame_state["heroes"]
        hero_pos = hero["pos"]

        hero_x_norm = _norm(hero_pos["x"], MAP_SIZE)
        hero_z_norm = _norm(hero_pos["z"], MAP_SIZE)
        flash_cd_norm = _norm(hero.get("flash_cooldown", 0), MAX_FLASH_CD)
        buff_remain_norm = _norm(hero.get("buff_remaining_time", 0), MAX_BUFF_DURATION)

        hero_feat = np.array(
            [hero_x_norm, hero_z_norm, flash_cd_norm, buff_remain_norm],
            dtype=np.float32,
        )

        # ------------------------------------------------------------------
        # 2) Monster scalar features
        # ------------------------------------------------------------------
        monsters = frame_state.get("monsters", [])
        monster_feats = []

        for i in range(2):
            if i < len(monsters):
                m = monsters[i]
                is_in_view = float(m.get("is_in_view", 0))
                m_pos = m.get("pos", {})

                if is_in_view and "x" in m_pos and "z" in m_pos:
                    m_speed_norm = _norm(m.get("speed", 1), MAX_MONSTER_SPEED)

                    dx = float(m_pos["x"]) - float(hero_pos["x"])
                    dz = float(m_pos["z"]) - float(hero_pos["z"])
                    raw_dist = np.sqrt(dx * dx + dz * dz)

                    if raw_dist > 1e-6:
                        m_dir_x = float(np.clip(dx / raw_dist, -1.0, 1.0))
                        m_dir_z = float(np.clip(dz / raw_dist, -1.0, 1.0))
                    else:
                        m_dir_x, m_dir_z = 0.0, 0.0

                    dist_bucket = m.get("hero_l2_distance", None)
                    if dist_bucket is not None:
                        dist_norm = _norm(float(dist_bucket), MAX_DIST_BUCKET)
                    else:
                        dist_norm = _norm(raw_dist, MAP_SIZE * 1.41)
                else:
                    m_dir_x = 0.0
                    m_dir_z = 0.0
                    m_speed_norm = 0.0
                    dist_norm = 1.0

                monster_feats.append(
                    np.array(
                        [is_in_view, m_dir_x, m_dir_z, m_speed_norm, dist_norm],
                        dtype=np.float32,
                    )
                )
            else:
                monster_feats.append(np.zeros(5, dtype=np.float32))

        # ------------------------------------------------------------------
        # 3) Relative monster features
        # ------------------------------------------------------------------
        rel_monster_feat = []
        for i in range(2):
            if i < len(monsters):
                rel_monster_feat.extend(
                    self._extract_monster_relative(monsters[i], hero_pos)
                )
            else:
                rel_monster_feat.extend([0.0, 0.0, 1.0])

        rel_monster_feat = np.array(rel_monster_feat, dtype=np.float32)

        # ------------------------------------------------------------------
        # 4) Local semantic map: obstacle / treasure / monster / buff
        # ------------------------------------------------------------------
        local_map = self._build_local_semantic_map(map_info, hero_pos, monsters)
        map_feat = local_map.reshape(-1).astype(np.float32)

        # ------------------------------------------------------------------
        # 5) Legal action mask
        # ------------------------------------------------------------------
        legal_action = [1] * Config.ACTION_NUM

        if isinstance(legal_act_raw, list) and len(legal_act_raw) >= Config.ACTION_NUM:
            legal_action = [
                1 if float(legal_act_raw[j]) > 0 else 0
                for j in range(Config.ACTION_NUM)
            ]

        if sum(legal_action) == 0:
            legal_action = [1] * Config.ACTION_NUM

        # ------------------------------------------------------------------
        # 6) Progress features
        # ------------------------------------------------------------------
        step_norm = _norm(self.step_no, self.max_step)
        survival_ratio = step_norm
        progress_feat = np.array([step_norm, survival_ratio], dtype=np.float32)

        # ------------------------------------------------------------------
        # 7) Nearest visible treasure direction from local treasure channel
        # ------------------------------------------------------------------
        treasure_dir_feat = self._nearest_treasure_direction_from_local_map(local_map)

        # ------------------------------------------------------------------
        # 8) Concatenate feature
        # ------------------------------------------------------------------
        feature = np.concatenate(
            [
                hero_feat,
                monster_feats[0],
                monster_feats[1],
                rel_monster_feat,
                map_feat,
                np.array(legal_action, dtype=np.float32),
                progress_feat,
                treasure_dir_feat,
            ]
        ).astype(np.float32)

        return feature, legal_action

    def compute_reward_from_transition(self, prev_obs, curr_obs, action):
        """
        Compute reward for transition.

        Args:
            prev_obs: Previous observation dict.
            curr_obs: Current observation dict returned by env.step.
            action: Reserved for future action-dependent reward. Currently unused.

        Returns:
            reward_value: float
            reward_components: dict
        """
        prev_observation = prev_obs["observation"]
        prev_frame_state = prev_observation["frame_state"]
        prev_env_info = prev_observation["env_info"]

        curr_observation = curr_obs["observation"]
        curr_frame_state = curr_observation["frame_state"]
        curr_env_info = curr_observation["env_info"]

        prev_hero_pos = prev_frame_state["heroes"]["pos"]
        curr_hero_pos = curr_frame_state["heroes"]["pos"]

        # ------------------------------------------------------------------
        # 1) Monster distance progress
        # ------------------------------------------------------------------
        prev_min_dist_norm = self._min_monster_dist_norm(prev_frame_state)
        curr_min_dist_norm = self._min_monster_dist_norm(curr_frame_state)

        monster_progress = curr_min_dist_norm - prev_min_dist_norm
        monster_distance_reward = 18.0 * monster_progress

        danger_penalty = -25.0 * max(0.0, 0.22 - curr_min_dist_norm)

        # ------------------------------------------------------------------
        # 2) Visible local treasure approach
        # ------------------------------------------------------------------
        treasure_value = Config.MAP_VALUE_TREASURE

        prev_min_treasure_dist_norm, prev_has_treasure = self._nearest_local_object_dist_norm(
            prev_obs,
            treasure_value,
        )
        curr_min_treasure_dist_norm, curr_has_treasure = self._nearest_local_object_dist_norm(
            curr_obs,
            treasure_value,
        )

        if prev_has_treasure and curr_has_treasure:
            treasure_progress = prev_min_treasure_dist_norm - curr_min_treasure_dist_norm
        else:
            treasure_progress = 0.0

        if curr_min_dist_norm > 0.32:
            treasure_approach_reward = 22.0 * max(0.0, treasure_progress)
            treasure_approach_reward -= 4.0 * max(0.0, -treasure_progress)
        else:
            treasure_approach_reward = -3.0 * max(0.0, treasure_progress)

        # 吃到宝箱奖励：如果 env_info 提供 treasure_id，可以继续用数量变化判断。
        # 如果环境不提供，则该项自然为 0。
        prev_treasure_ids = self._get_treasure_ids(prev_env_info)
        curr_treasure_ids = self._get_treasure_ids(curr_env_info)

        treasure_delta = max(0.0, float(len(prev_treasure_ids) - len(curr_treasure_ids)))
        treasure_score_reward = 80.0 * treasure_delta
        treasure_reward = treasure_approach_reward + treasure_score_reward

        # ------------------------------------------------------------------
        # 3) Terminal reward
        # ------------------------------------------------------------------
        done = bool(curr_env_info.get("done", False))
        terminal_reward = 0.0

        if done:
            is_caught = bool(curr_env_info.get("is_caught", False))
            is_success = bool(curr_env_info.get("is_success", False))
            is_timeout = bool(curr_env_info.get("is_timeout", False))

            if is_caught:
                terminal_reward = -100.0
            elif is_success:
                terminal_reward = 100.0
            elif is_timeout:
                terminal_reward = -5.0
            else:
                curr_step_no = int(curr_observation.get("step_no", 0))
                max_step = int(curr_env_info.get("max_step", 200))
                terminal_reward = 100.0 if curr_step_no >= max_step - 1 else -100.0

        # ------------------------------------------------------------------
        # 4) Invalid move penalty
        # ------------------------------------------------------------------
        prev_hx = float(prev_hero_pos.get("x", 0.0))
        prev_hz = float(prev_hero_pos.get("z", 0.0))
        curr_hx = float(curr_hero_pos.get("x", 0.0))
        curr_hz = float(curr_hero_pos.get("z", 0.0))

        disp = np.sqrt((curr_hx - prev_hx) ** 2 + (curr_hz - prev_hz) ** 2)
        invalid_move_penalty = -5.0 if disp < 0.2 else 0.0

        # ------------------------------------------------------------------
        # 5) Repeat visit penalty
        # ------------------------------------------------------------------
        cell = (int(curr_hx // 4), int(curr_hz // 4))

        self.visit_counter[cell] = self.visit_counter.get(cell, 0) + 1

        revisit = self.visit_counter[cell]
        repeat_penalty = 0.0

        if revisit > 3:
            repeat_penalty -= 1.5 * min(5, revisit - 3)

        # ------------------------------------------------------------------
        # 6) Stage reward
        # ------------------------------------------------------------------
        curr_step_no = int(curr_observation.get("step_no", 0))
        stage_reward = 0.0

        if 0 < curr_step_no <= 1000 and curr_step_no % 100 == 0:
            stage_reward = 20.0 * (curr_step_no // 100)

        # ------------------------------------------------------------------
        # 7) Env score delta
        # ------------------------------------------------------------------
        prev_total_score = float(prev_env_info.get("total_score", 0.0))
        curr_total_score = float(curr_env_info.get("total_score", 0.0))
        total_score_delta = curr_total_score - prev_total_score

        # ------------------------------------------------------------------
        # 8) Reward scale
        # ------------------------------------------------------------------
        raw_reward = (
            terminal_reward
            + monster_distance_reward
            + danger_penalty
            + treasure_reward
            + invalid_move_penalty
            + repeat_penalty
            + stage_reward
            + total_score_delta
        )

        reward_scale = float(getattr(Config, "REWARD_SCALE", 0.005))
        scaled_reward = raw_reward * reward_scale
        reward_value = float(scaled_reward)

        reward_components = {
            "terminal_reward": float(terminal_reward * reward_scale),
            "monster_distance_reward": float(monster_distance_reward * reward_scale),
            "dist_shaping": float(monster_distance_reward * reward_scale),
            "danger_penalty": float(danger_penalty * reward_scale),
            "treasure_reward": float(treasure_reward * reward_scale),
            "treasure_approach_reward": float(treasure_approach_reward * reward_scale),
            "treasure_score_reward": float(treasure_score_reward * reward_scale),
            "invalid_move_penalty": float(invalid_move_penalty * reward_scale),
            "repeat_penalty": float(repeat_penalty * reward_scale),
            "total_score_delta": float(total_score_delta * reward_scale),
            "stage_reward": float(stage_reward * reward_scale),

            "raw_reward_total": float(raw_reward),
            "scaled_reward_total": float(scaled_reward),
            "reward_total": float(reward_value),

            # debug
            "debug_prev_monster_dist": float(prev_min_dist_norm),
            "debug_curr_monster_dist": float(curr_min_dist_norm),
            "debug_monster_progress": float(monster_progress),

            "debug_prev_treasure_dist": float(prev_min_treasure_dist_norm),
            "debug_curr_treasure_dist": float(curr_min_treasure_dist_norm),
            "debug_prev_has_treasure": float(prev_has_treasure),
            "debug_curr_has_treasure": float(curr_has_treasure),
            "debug_treasure_progress": float(treasure_progress),
            "debug_treasure_delta": float(treasure_delta),
        }

        self.last_reward_components = dict(reward_components)
        self.last_env_info = dict(curr_env_info) if isinstance(curr_env_info, dict) else {}

        return reward_value, reward_components

    # ======================================================================
    # Local map helpers
    # ======================================================================
    def _build_local_semantic_map(self, map_info, hero_pos, monsters=None):
        """
        Build local semantic map with shape [4, 21, 21].

        当前环境约定：
        map_info 已经是以 hero 为中心的 21x21 局部视野，
        不存在全局地图，也不做全局地图裁剪。

        channel 0: obstacle
        channel 1: treasure
        channel 2: monster
        channel 3: buff
        """
        local_map = np.zeros(
            (Config.MAP_CHANNELS, Config.LOCAL_MAP_WINDOW, Config.LOCAL_MAP_WINDOW),
            dtype=np.float32,
        )

        if map_info is None or len(map_info) == 0 or len(map_info[0]) == 0:
            return local_map

        map_arr = np.array(map_info)

        if map_arr.shape != (Config.LOCAL_MAP_WINDOW, Config.LOCAL_MAP_WINDOW):
            return local_map

        local_map[0] = (map_arr == Config.MAP_VALUE_OBSTACLE).astype(np.float32)
        local_map[1] = (map_arr == Config.MAP_VALUE_TREASURE).astype(np.float32)
        local_map[2] = (map_arr == Config.MAP_VALUE_MONSTER).astype(np.float32)
        local_map[3] = (map_arr == Config.MAP_VALUE_BUFF).astype(np.float32)

        # 如果 map_info 里怪兽编码不稳定，但 frame_state.monsters 能提供可见怪兽位置，则补充到 monster channel。
        if monsters is not None:
            self._overlay_visible_monsters(local_map, hero_pos, monsters)

        return local_map


    def _overlay_visible_monsters(self, local_map, hero_pos, monsters):
        center = LOCAL_MAP_WINDOW // 2

        for m in monsters:
            if not isinstance(m, dict):
                continue

            if m.get("is_in_view", 0) <= 0:
                continue

            m_pos = m.get("pos", {})
            if "x" not in m_pos or "z" not in m_pos:
                continue

            dc = int(round(float(m_pos["x"]) - float(hero_pos.get("x", 0.0))))
            dr = int(round(float(m_pos["z"]) - float(hero_pos.get("z", 0.0))))

            rr = center + dr
            cc = center + dc

            if 0 <= rr < LOCAL_MAP_WINDOW and 0 <= cc < LOCAL_MAP_WINDOW:
                local_map[2, rr, cc] = 1.0

    def _nearest_treasure_direction_from_local_map(self, local_map):
        """
        Return [dx, dz, dist_norm] of nearest visible treasure.
        If no visible treasure exists, return [0, 0, 1].
        """
        treasure_dir_feat = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        treasure_positions = np.argwhere(local_map[1] > 0.5)
        if len(treasure_positions) == 0:
            return treasure_dir_feat

        center = Config.LOCAL_MAP_WINDOW // 2
        deltas = treasure_positions - np.array([center, center])
        dists = np.sqrt((deltas ** 2).sum(axis=1))

        idx = int(np.argmin(dists))
        dr, dc = deltas[idx]
        dist = dists[idx]

        if dist > 1e-6:
            treasure_dir_feat[0] = float(np.clip(dc / dist, -1.0, 1.0))
            treasure_dir_feat[1] = float(np.clip(dr / dist, -1.0, 1.0))
            treasure_dir_feat[2] = float(np.clip(dist / center, 0.0, 1.0))

        return treasure_dir_feat

    def _nearest_local_object_dist_norm(self, env_obs, target_value):
        """
        Find nearest target object in local 21x21 map.

        当前环境约定：
        map_info 已经是以 hero 为中心的 21x21 局部视野。

        Returns:
            dist_norm: normalized distance to nearest object, default 1.0
            found: whether object exists in local view
        """
        observation = env_obs["observation"]
        map_info = observation["map_info"]

        if map_info is None or len(map_info) == 0 or len(map_info[0]) == 0:
            return 1.0, False

        map_arr = np.array(map_info)

        if map_arr.shape != (Config.LOCAL_MAP_WINDOW, Config.LOCAL_MAP_WINDOW):
            return 1.0, False

        positions = np.argwhere(map_arr == target_value)
        if len(positions) == 0:
            return 1.0, False

        center = Config.LOCAL_MAP_WINDOW // 2
        deltas = positions - np.array([center, center])
        dists = np.sqrt((deltas ** 2).sum(axis=1))

        min_dist = float(np.min(dists))
        return float(np.clip(min_dist / center, 0.0, 1.0)), True

    # ======================================================================
    # Scalar feature helpers
    # ======================================================================
    def _min_monster_dist_norm(self, frame_state):
        monsters = frame_state.get("monsters", [])
        min_dist_norm = 1.0

        for m in monsters:
            if not isinstance(m, dict):
                continue

            if m.get("is_in_view", 0) <= 0:
                continue

            dist_bucket = m.get("hero_l2_distance", None)
            if dist_bucket is not None:
                min_dist_norm = min(
                    min_dist_norm,
                    _norm(float(dist_bucket), MAX_DIST_BUCKET),
                )

        return float(min_dist_norm)

    def _get_treasure_ids(self, env_info):
        treasure_ids = env_info.get("treasure_id", [])
        if not isinstance(treasure_ids, list):
            return []
        return treasure_ids

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

    def get_last_reward_components(self):
        return dict(self.last_reward_components)

    def get_last_env_info(self):
        return dict(self.last_env_info)
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
        self.init_remaining_treasure_count = None
        self.last_remaining_treasure_count = None
        self.last_hero_pos = None
        self.last_flash_cd = 0.0
        self.visit_counter = {}
        self.recent_positions = deque(maxlen=20)
        self.last_reward_components = {}
        self.last_env_info = {}
        self.last_survival_stage = 0
        # Post-flash behavior window / 闪现后行为约束窗口
        self.post_flash_window = 0

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

        # Keep 4D shape unchanged and include hero absolute coordinates.
        # 保持4维不变，并记录英雄绝对坐标（归一化）。
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
                    m_speed_norm = _norm(m.get("speed", 1), MAX_MONSTER_SPEED)
                    dx = float(m_pos["x"]) - float(hero_pos["x"])
                    dz = float(m_pos["z"]) - float(hero_pos["z"])
                    raw_dist = np.sqrt(dx * dx + dz * dz)
                    if raw_dist > 1e-6:
                        m_dir_x = float(np.clip(dx / raw_dist, -1.0, 1.0))
                        m_dir_z = float(np.clip(dz / raw_dist, -1.0, 1.0))
                    else:
                        m_dir_x, m_dir_z = 0.0, 0.0

                    # Prefer env-provided bucket distance when available.
                    # 优先使用环境直接提供的距离桶（hero_l2_distance: 0~5）。
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
                    np.array([is_in_view, m_dir_x, m_dir_z, m_speed_norm, dist_norm], dtype=np.float32)
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

        # Spatial map features (1 x 21 x 21) / 空间特征图（仅障碍）
        # monster / treasure 改由标量关系特征学习。

        # Treasure list is still used by scalar features and reward.
        # 宝箱列表仍用于标量特征与奖励计算。
        treasure_list = frame_state.get("treasures", frame_state.get("treasure", []))
        if isinstance(treasure_list, dict):
            treasure_list = [treasure_list]

        # Build obstacle channel from local map occupancy / 障碍物通道
        # map_info 定义：1=可通行，0=障碍物。
        # 以英雄坐标为窗口中心，越界区域按障碍处理。
        obstacle_channel = np.zeros((LOCAL_MAP_WINDOW, LOCAL_MAP_WINDOW), dtype=np.float32)
        if map_info is not None and len(map_info) > 0 and len(map_info[0]) > 0:
            radius = LOCAL_MAP_WINDOW // 2
            hero_row = int(round(float(hero_pos.get("z", 0.0))))
            hero_col = int(round(float(hero_pos.get("x", 0.0))))
            map_rows = len(map_info)
            map_cols = len(map_info[0])
            for rr in range(LOCAL_MAP_WINDOW):
                for cc in range(LOCAL_MAP_WINDOW):
                    src_row = hero_row - radius + rr
                    src_col = hero_col - radius + cc
                    if 0 <= src_row < map_rows and 0 <= src_col < map_cols:
                        obstacle_channel[rr, cc] = float(map_info[src_row][src_col] == 0)
                    else:
                        obstacle_channel[rr, cc] = 1.0
        map_feat = obstacle_channel.reshape(-1)

        # Legal action mask (16D) / 合法动作掩码
        # 仅使用环境提供的 bool[16] 掩码，不做兼容映射。
        legal_action = [1] * 16
        if isinstance(legal_act_raw, list) and len(legal_act_raw) >= 16 and isinstance(legal_act_raw[0], bool):
            legal_action = [int(legal_act_raw[j]) for j in range(16)]

        if sum(legal_action) == 0:
            legal_action = [1] * 16

        # Progress features (2D) / 进度特征
        step_norm = _norm(self.step_no, self.max_step)
        survival_ratio = step_norm
        progress_feat = np.array([step_norm, survival_ratio], dtype=np.float32)
        treasure_dir_feat = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        nearest_treasure_dist_norm = float("inf")
        for t in treasure_list:
            if not isinstance(t, dict):
                continue
            bucket = t.get("hero_l2_distance", None)
            direction = t.get("hero_relative_direction", None)
            if bucket is not None and direction is not None:
                bucket = float(bucket)
                dist_norm = _norm(bucket, MAX_DIST_BUCKET)
                if dist_norm < nearest_treasure_dist_norm:
                    nearest_treasure_dist_norm = dist_norm
                    dir_x, dir_z = self._direction_id_to_vec(int(direction))
                    treasure_dir_feat[0] = dir_x
                    treasure_dir_feat[1] = dir_z
                    treasure_dir_feat[2] = dist_norm
                continue

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
                treasure_dir_feat,
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
            dist_bucket = t.get("hero_l2_distance", None)
            if dist_bucket is not None:
                cur_min_treasure_dist_norm = min(cur_min_treasure_dist_norm, _norm(float(dist_bucket), MAX_DIST_BUCKET))
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
        treasure_ids = env_info.get("treasure_id", [])
        if not isinstance(treasure_ids, list):
            treasure_ids = []
        remaining_treasure_count = float(len(treasure_ids))
        if self.init_remaining_treasure_count is None:
            self.init_remaining_treasure_count = remaining_treasure_count
        if self.last_remaining_treasure_count is None:
            self.last_remaining_treasure_count = remaining_treasure_count

        # Dense rewards / 稠密奖励
        survive_reward = float(getattr(Config, "REWARD_SURVIVE", 0.01))
        step_score_reward = float(getattr(Config, "REWARD_STEP_SCORE", 0.05)) if step_score > self.last_step_score else 0.0
        dist_shaping = float(getattr(Config, "REWARD_DIST_SHAPING", 0.08)) * (
            cur_min_dist_norm - self.last_min_monster_dist_norm
        )
        treasure_progress = self.last_min_treasure_dist_norm - cur_min_treasure_dist_norm
        treasure_approach_reward = float(getattr(Config, "REWARD_TREASURE_APPROACH", 0.12)) * max(0.0, treasure_progress)
        treasure_approach_reward -= float(getattr(Config, "PENALTY_TREASURE_AWAY", 0.03)) * max(0.0, -treasure_progress)
        if cur_min_dist_norm > float(getattr(Config, "TREASURE_SAFE_DISTANCE_TH", 0.35)) and treasure_progress > 0.0:
            treasure_approach_reward += min(
                float(getattr(Config, "REWARD_TREASURE_SAFE_BONUS_CAP", 0.04)),
                float(getattr(Config, "REWARD_TREASURE_SAFE_BONUS_COEF", 0.16)) * treasure_progress,
            )

        # Progressive survival reward / 生存递进奖励（步数越高奖励越大）
        progressive_step_reward = float(getattr(Config, "REWARD_PROGRESSIVE_STEP", 0.02)) * step_norm

        # Progressive survival milestone reward / 生存里程碑递进奖励（默认20步更新一次）
        milestone_step = int(max(1, getattr(Config, "MILESTONE_STEP", 20)))
        current_stage = int(min(50, self.step_no // milestone_step))
        stage_progress_reward = float(getattr(Config, "REWARD_STAGE_PROGRESS", 0.04)) * max(
            0, current_stage - self.last_survival_stage
        )
        self.last_survival_stage = current_stage

        # Speed-up stage shaping / 怪物加速前后强化
        monster_speedup_step = float(env_info.get("monster_speedup", 500))
        near_speedup_ratio = _norm(self.step_no, max(monster_speedup_step, 1.0))
        near_speedup_bonus = float(getattr(Config, "REWARD_NEAR_SPEEDUP", 0.03)) * near_speedup_ratio * max(
            0.0, cur_min_dist_norm - 0.25
        )
        late_survival_bonus = float(getattr(Config, "REWARD_LATE_SURVIVAL", 0.05)) * max(
            0.0, near_speedup_ratio - 0.6
        ) * max(0.0, cur_min_dist_norm - 0.2)

        # Corridor openness reward / corridor开阔奖励
        corridor_reward = float(getattr(Config, "REWARD_CORRIDOR", 0.05)) * self._compute_openness(obstacle_channel)

        # Sparse rewards / 稀疏奖励
        # 以剩余宝箱ID列表长度为核心：长度越短，奖励越高；
        # 且在宝箱数量发生减少时给予增量奖励。
        treasure_delta = max(0.0, self.last_remaining_treasure_count - remaining_treasure_count)
        init_count = max(1.0, float(self.init_remaining_treasure_count))
        completion_ratio = float(np.clip((init_count - remaining_treasure_count) / init_count, 0.0, 1.0))
        treasure_score_reward = float(getattr(Config, "REWARD_TREASURE_SCORE", 0.6)) * treasure_delta * (1.0 + completion_ratio)
        buff_reward = float(getattr(Config, "REWARD_BUFF", 0.2)) if buff_count > self.last_buff_count else 0.0

        # Risk penalties / 风险惩罚
        danger_penalty = -float(getattr(Config, "PENALTY_DANGER", 0.08)) * max(0.0, 0.25 - cur_min_dist_norm)
        second_monster_penalty = -float(getattr(Config, "PENALTY_SECOND_MONSTER", 0.04)) * max(
            0.0, 0.28 - cur_second_dist_norm
        )
        corner_penalty = -float(getattr(Config, "PENALTY_CORNER", 0.04)) * max(
            0.0, 0.2 - self._compute_openness(obstacle_channel)
        )
        encircle_penalty = -float(getattr(Config, "PENALTY_ENCIRCLE", 0.03)) * self._compute_encircle_penalty(
            hero_pos, monsters
        )

        # Movement penalties / 移动相关惩罚
        invalid_move_penalty = self._compute_invalid_move_penalty(hero_pos)
        repeat_explore_penalty = self._compute_repeat_penalty(hero_pos)

        # Flash-related reward / 闪现相关奖励
        flash_escape_reward, flash_abuse_penalty = self._compute_flash_reward(
            hero.get("flash_cooldown", 0.0), cur_min_dist_norm, obstacle_channel
        )
        post_flash_move_bonus, post_flash_idle_penalty = self._compute_post_flash_momentum(
            hero_pos=hero_pos,
            cur_min_dist_norm=cur_min_dist_norm,
            cur_min_treasure_dist_norm=cur_min_treasure_dist_norm,
        )

        reward_value = (
            survive_reward
            + step_score_reward
            + treasure_score_reward
            + buff_reward
            + treasure_approach_reward
            + progressive_step_reward
            + stage_progress_reward
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
            + post_flash_move_bonus
            + post_flash_idle_penalty
            + second_monster_penalty
        )
        reward_value = float(np.clip(reward_value, -1.5, 1.5))
        self.last_reward_components = {
            "survive_reward": float(survive_reward),
            "step_score_reward": float(step_score_reward),
            "treasure_score_reward": float(treasure_score_reward),
            "buff_reward": float(buff_reward),
            "treasure_approach_reward": float(treasure_approach_reward),
            "progressive_step_reward": float(progressive_step_reward),
            "stage_progress_reward": float(stage_progress_reward),
            "dist_shaping": float(dist_shaping),
            "near_speedup_bonus": float(near_speedup_bonus),
            "late_survival_bonus": float(late_survival_bonus),
            "corridor_reward": float(corridor_reward),
            "encircle_penalty": float(encircle_penalty),
            "corner_penalty": float(corner_penalty),
            "danger_penalty": float(danger_penalty),
            "invalid_move_penalty": float(invalid_move_penalty),
            "repeat_explore_penalty": float(repeat_explore_penalty),
            "flash_escape_reward": float(flash_escape_reward),
            "flash_abuse_penalty": float(flash_abuse_penalty),
            "post_flash_move_bonus": float(post_flash_move_bonus),
            "post_flash_idle_penalty": float(post_flash_idle_penalty),
            "second_monster_penalty": float(second_monster_penalty),
            "reward_total": float(reward_value),
        }

        self.last_min_monster_dist_norm = cur_min_dist_norm
        self.last_min_treasure_dist_norm = cur_min_treasure_dist_norm
        self.last_step_score = step_score
        self.last_treasure_score = treasure_score
        self.last_buff_count = buff_count
        self.last_remaining_treasure_count = remaining_treasure_count
        self.last_flash_cd = float(hero.get("flash_cooldown", 0.0))
        self.last_hero_pos = (float(hero_pos.get("x", 0.0)), float(hero_pos.get("z", 0.0)))
        self.last_env_info = dict(env_info) if isinstance(env_info, dict) else {}

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
        invalid_move_penalty = float(getattr(Config, "PENALTY_INVALID_MOVE", 0.03))
        return -invalid_move_penalty if disp < 0.2 else 0.0

    def _compute_repeat_penalty(self, hero_pos):
        hx, hz = float(hero_pos.get("x", 0.0)), float(hero_pos.get("z", 0.0))
        cell = (int(hx // 4), int(hz // 4))
        self.visit_counter[cell] = self.visit_counter.get(cell, 0) + 1
        self.recent_positions.append(cell)
        unique_ratio = len(set(self.recent_positions)) / max(1, len(self.recent_positions))
        revisit = self.visit_counter[cell]
        revisit_penalty = float(getattr(Config, "PENALTY_REPEAT_VISIT", 0.02))
        unique_penalty = float(getattr(Config, "PENALTY_REPEAT_UNIQUE", 0.02))
        return -revisit_penalty * max(0.0, revisit / 20.0) - unique_penalty * max(0.0, 0.5 - unique_ratio)

    def _compute_flash_reward(self, flash_cd, cur_min_dist_norm, obstacle_channel):
        flash_cd = float(flash_cd)
        flashed = flash_cd > self.last_flash_cd + 100.0
        if not flashed:
            return 0.0, 0.0
        # Start a short post-flash behavior window to discourage "flash then idle".
        self.post_flash_window = int(getattr(Config, "POST_FLASH_WINDOW", 8))
        openness = self._compute_openness(obstacle_channel)
        escaped = cur_min_dist_norm - self.last_min_monster_dist_norm
        if escaped > 0.05 or openness > 0.6:
            return float(getattr(Config, "FLASH_ESCAPE_REWARD", 0.25)), 0.0
        return 0.0, -float(getattr(Config, "FLASH_ABUSE_PENALTY", 0.08))

    def _compute_post_flash_momentum(self, hero_pos, cur_min_dist_norm, cur_min_treasure_dist_norm):
        if self.post_flash_window <= 0:
            return 0.0, 0.0

        self.post_flash_window -= 1

        if self.last_hero_pos is None:
            return 0.0, 0.0

        hx, hz = float(hero_pos.get("x", 0.0)), float(hero_pos.get("z", 0.0))
        dx = hx - self.last_hero_pos[0]
        dz = hz - self.last_hero_pos[1]
        disp = np.sqrt(dx * dx + dz * dz)

        move_bonus = 0.0
        idle_penalty = 0.0

        # Encourage sustained movement for a few steps after flash.
        if disp > float(getattr(Config, "THRESH_POST_FLASH_MOVE", 0.35)):
            move_bonus += float(getattr(Config, "REWARD_POST_FLASH_MOVE", 0.02))
        elif disp < float(getattr(Config, "THRESH_POST_FLASH_IDLE", 0.2)):
            idle_penalty -= float(getattr(Config, "PENALTY_POST_FLASH_IDLE", 0.04))

        # If monsters are relatively far, emphasize "don't stand still".
        if (
            cur_min_dist_norm > float(getattr(Config, "TREASURE_SAFE_DISTANCE_TH", 0.35))
            and disp < float(getattr(Config, "THRESH_POST_FLASH_SAFE_IDLE", 0.25))
        ):
            idle_penalty -= float(getattr(Config, "PENALTY_POST_FLASH_SAFE_IDLE", 0.03))

        # Small extra incentive to keep approaching treasure after a successful escape.
        treasure_progress = self.last_min_treasure_dist_norm - cur_min_treasure_dist_norm
        if treasure_progress > 0.0:
            move_bonus += min(
                float(getattr(Config, "REWARD_POST_FLASH_TREASURE_CAP", 0.03)),
                float(getattr(Config, "REWARD_POST_FLASH_TREASURE_COEF", 0.12)) * treasure_progress,
            )

        return move_bonus, idle_penalty

    def _compute_post_flash_momentum(self, hero_pos, cur_min_dist_norm, cur_min_treasure_dist_norm):
        if self.post_flash_window <= 0:
            return 0.0, 0.0

        self.post_flash_window -= 1

        if self.last_hero_pos is None:
            return 0.0, 0.0

        hx, hz = float(hero_pos.get("x", 0.0)), float(hero_pos.get("z", 0.0))
        dx = hx - self.last_hero_pos[0]
        dz = hz - self.last_hero_pos[1]
        disp = np.sqrt(dx * dx + dz * dz)

        move_bonus = 0.0
        idle_penalty = 0.0

        # Encourage sustained movement for a few steps after flash.
        if disp > 0.35:
            move_bonus += 0.02
        elif disp < 0.2:
            idle_penalty -= 0.04

        # If monsters are relatively far, emphasize "don't stand still".
        if cur_min_dist_norm > 0.35 and disp < 0.25:
            idle_penalty -= 0.03

        # Small extra incentive to keep approaching treasure after a successful escape.
        treasure_progress = self.last_min_treasure_dist_norm - cur_min_treasure_dist_norm
        if treasure_progress > 0.0:
            move_bonus += min(0.03, 0.12 * treasure_progress)

        return move_bonus, idle_penalty

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

    def _direction_id_to_vec(self, direction_id):
        mapping = {
            1: (1.0, 0.0),  # 东
            2: (0.7071, 0.7071),  # 东北
            3: (0.0, 1.0),  # 北
            4: (-0.7071, 0.7071),  # 西北
            5: (-1.0, 0.0),  # 西
            6: (-0.7071, -0.7071),  # 西南
            7: (0.0, -1.0),  # 南
            8: (0.7071, -0.7071),  # 东南
        }
        return mapping.get(direction_id, (0.0, 0.0))

    def get_last_reward_components(self):
        return dict(self.last_reward_components)

    def get_last_env_info(self):
        return dict(self.last_env_info)

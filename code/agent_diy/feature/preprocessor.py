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
3. CNN 输入分为两部分：
local_map：21×21×1，只包含 obstacle
global_entity_map：128×128×4，包含 hero / treasure / monster / buff
"""
import heapq
from collections import deque
import numpy as np
from agent_diy.conf.conf import Config

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

        # 最近 N 步位置历史：
        # 1. 用于 anti-stuck reward
        # 2. 用于构造 10 步前相对位移特征
        self.position_history = deque(maxlen=Config.ANTI_STUCK_WINDOW + 1)

        # 本局已经第一次看见过的宝箱位置
        self.seen_treasure_keys = set()

        self.last_reward_components = {}
        self.last_env_info = {}

    def feature_process(self, env_obs, last_action):
        observation = env_obs.get("observation", {})
        obs_frame_state = observation.get("frame_state", {})
        extra_frame_state = self._get_extra_frame_state(env_obs)

        env_info = observation.get("env_info", {})
        map_info = observation.get("map_info", [])
        legal_act_raw = observation.get("legal_action", [])

        hero = obs_frame_state.get("heroes", {})
        hero_pos = hero.get("pos", {})

        # extra_info 优先，用于全局实体图、怪兽特征、宝箱/Buff方向
        organs = extra_frame_state.get("organs", obs_frame_state.get("organs", []))

        flash_cooldown_max = float(
            env_info.get("flash_cooldown_max", Config.MAX_FLASH_CD)
        )

        hero_feat = np.array(
            [
                _norm(hero_pos.get("x", 0.0), Config.MAP_SIZE),
                _norm(hero_pos.get("z", 0.0), Config.MAP_SIZE),
                _norm(hero.get("flash_cooldown", 0.0), flash_cooldown_max),
                _norm(hero.get("buff_remaining_time", 0.0), Config.MAX_BUFF_DURATION),
            ],
            dtype=np.float32,
        )

        monster_feat = self._build_monster_features(extra_frame_state)

        # 1) 局部图：只保留 obstacle，负责局部避障
        local_map = self._build_local_obstacle_map(map_info)
        local_map_feat = local_map.reshape(-1).astype(np.float32)

        # 2) 全局图：hero / treasure / monster / buff 高斯热力图
        global_map = self._build_global_entity_map(extra_frame_state)
        global_map_feat = global_map.reshape(-1).astype(np.float32)

        # 3) legal action
        legal_action = [1] * Config.ACTION_NUM
        if isinstance(legal_act_raw, list) and len(legal_act_raw) >= Config.ACTION_NUM:
            legal_action = [
                1 if float(legal_act_raw[j]) > 0 else 0
                for j in range(Config.ACTION_NUM)
            ]

        if sum(legal_action) == 0:
            legal_action = [1] * Config.ACTION_NUM

        legal_action_feat = np.array(legal_action, dtype=np.float32)

        # 4) scalar status
        status_feat = self._build_status_features(obs_frame_state, env_info)

        # 5) anti-stuck feature: [dx10_norm, dz10_norm]
        anti_stuck_feat = self._build_anti_stuck_feature(hero_pos)
        # 6) 目标方向仍然保留：用于给 MLP 一个明确的最近目标方向
        treasure_dir_feat = self._nearest_organ_path_direction_feature(
            map_info,
            extra_frame_state,
            organs,
            Config.ORGAN_TYPE_TREASURE,
        )

        buff_dir_feat = self._nearest_organ_path_direction_feature(
            map_info,
            extra_frame_state,
            organs,
            Config.ORGAN_TYPE_BUFF,
        )

        feature = np.concatenate(
            [
                hero_feat,
                monster_feat,
                local_map_feat,
                global_map_feat,
                legal_action_feat,
                status_feat,
                anti_stuck_feat,
                treasure_dir_feat,
                buff_dir_feat,
            ]
        ).astype(np.float32)

        if feature.shape[0] != Config.DIM_OF_OBSERVATION:
            raise ValueError(
                f"feature dim mismatch: got {feature.shape[0]}, "
                f"expected {Config.DIM_OF_OBSERVATION}"
            )

        return feature, legal_action
    
    def _nearest_organ_direction_feature(self, frame_state, organs, sub_type):
        hero = frame_state.get("heroes", {})
        hero_pos = hero.get("pos", {})

        hx = float(hero_pos.get("x", 0.0))
        hz = float(hero_pos.get("z", 0.0))

        best_dist = None
        best_dx = 0.0
        best_dz = 0.0

        for organ in organs:
            if not isinstance(organ, dict):
                continue

            if int(organ.get("status", 0)) <= 0:
                continue

            if int(organ.get("sub_type", -1)) != int(sub_type):
                continue

            pos = organ.get("pos", {})
            if "x" not in pos or "z" not in pos:
                continue

            dx = float(pos["x"]) - hx
            dz = float(pos["z"]) - hz
            dist = np.sqrt(dx * dx + dz * dz)

            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_dx = dx
                best_dz = dz

        if best_dist is None:
            return np.array([0.0, 0.0, 1.0], dtype=np.float32)

        if best_dist > 1e-6:
            dir_x = float(np.clip(best_dx / best_dist, -1.0, 1.0))
            dir_z = float(np.clip(best_dz / best_dist, -1.0, 1.0))
        else:
            dir_x, dir_z = 0.0, 0.0

        dist_norm = float(np.clip(best_dist / (Config.MAP_SIZE* 1.41), 0.0, 1.0))

        return np.array([dir_x, dir_z, dist_norm], dtype=np.float32)

    def compute_reward_from_transition(self, prev_obs, curr_obs, action):
        """
        Compute reward for transition.

        Reward 统一使用 extra_info.frame_state 中的全局实体信息：
        - monsters: 怪兽位置、速度
        - organs: 宝箱 / buff 位置
        - observation.env_info: 分数、收集数量、终止信息

        map_info 只表示障碍/可通行，不参与宝箱、怪兽、buff reward 计算。
        """
        prev_observation = prev_obs["observation"]
        curr_observation = curr_obs["observation"]

        prev_obs_frame_state = prev_observation["frame_state"]
        curr_obs_frame_state = curr_observation["frame_state"]

        prev_frame_state = self._get_extra_frame_state(prev_obs)
        curr_frame_state = self._get_extra_frame_state(curr_obs)

        prev_env_info = prev_observation["env_info"]
        curr_env_info = curr_observation["env_info"]

        prev_hero_pos = prev_obs_frame_state["heroes"]["pos"]
        curr_hero_pos = curr_obs_frame_state["heroes"]["pos"]

        # ------------------------------------------------------------------
        # 1) Monster distance progress from extra_info.monsters
        # ------------------------------------------------------------------
        prev_min_dist_norm = self._min_monster_dist_norm(prev_frame_state)
        curr_min_dist_norm = self._min_monster_dist_norm(curr_frame_state)

        prev_has_monster = prev_min_dist_norm < 1.0
        curr_has_monster = curr_min_dist_norm < 1.0

        if prev_has_monster or curr_has_monster:
            monster_progress = curr_min_dist_norm - prev_min_dist_norm
        else:
            monster_progress = 0.0

        # ------------------------------------------------------------------
        # Global field shaping
        # 全局势场塑形：替代旧的 monster_distance / treasure_approach / buff_approach
        # ------------------------------------------------------------------
        prev_monster_heat = self._entity_field_value_at_hero(
            prev_frame_state,
            entity_type="monster",
        )
        curr_monster_heat = self._entity_field_value_at_hero(
            curr_frame_state,
            entity_type="monster",
        )

        prev_treasure_heat = self._entity_field_value_at_hero(
            prev_frame_state,
            entity_type="organ",
            sub_type=Config.ORGAN_TYPE_TREASURE,
        )
        curr_treasure_heat = self._entity_field_value_at_hero(
            curr_frame_state,
            entity_type="organ",
            sub_type=Config.ORGAN_TYPE_TREASURE,
        )

        prev_buff_heat = self._entity_field_value_at_hero(
            prev_frame_state,
            entity_type="organ",
            sub_type=Config.ORGAN_TYPE_BUFF,
        )
        curr_buff_heat = self._entity_field_value_at_hero(
            curr_frame_state,
            entity_type="organ",
            sub_type=Config.ORGAN_TYPE_BUFF,
        )

        monster_field_progress = prev_monster_heat - curr_monster_heat
        treasure_field_progress = curr_treasure_heat - prev_treasure_heat
        buff_field_progress = curr_buff_heat - prev_buff_heat

        field_eps = getattr(Config, "FIELD_PROGRESS_EPS", 0.005)

        if abs(monster_field_progress) < field_eps:
            monster_field_progress = 0.0

        if abs(treasure_field_progress) < field_eps:
            treasure_field_progress = 0.0

        if abs(buff_field_progress) < field_eps:
            buff_field_progress = 0.0
        global_monster_field_reward = (
            Config.REWARD_GLOBAL_MONSTER_FIELD * max(0.0, monster_field_progress)
            - Config.PENALTY_GLOBAL_MONSTER_FIELD * max(0.0, -monster_field_progress)
        )

        if curr_min_dist_norm > Config.TREASURE_SAFE_DISTANCE_TH:
            global_treasure_field_reward = (
                Config.REWARD_GLOBAL_TREASURE_FIELD * max(0.0, treasure_field_progress)
            )
        else:
            global_treasure_field_reward = (
                -Config.PENALTY_GLOBAL_TREASURE_GREED * max(0.0, treasure_field_progress)
            )

        if curr_min_dist_norm > Config.BUFF_SAFE_DISTANCE_TH:
            global_buff_field_reward = (
                Config.REWARD_GLOBAL_BUFF_FIELD * max(0.0, buff_field_progress)
            )
        else:
            global_buff_field_reward = (
                -Config.PENALTY_GLOBAL_BUFF_GREED * max(0.0, buff_field_progress)
            )

        monster_speed_factor = self._visible_monster_speed_factor(curr_frame_state)
        danger_penalty = (
            -Config.PENALTY_DANGER
            * monster_speed_factor
            * max(0.0, Config.DANGER_DISTANCE_TH - curr_min_dist_norm)
        )

        # ------------------------------------------------------------------
        # Flash reward / penalty
        # ------------------------------------------------------------------
        is_flash_action = int(action) >= Config.FLASH_ACTION_START
        in_flash_danger = prev_min_dist_norm < Config.FLASH_DANGER_DIST_TH

        flash_escape_reward = 0.0
        flash_toward_monster_penalty = 0.0
        flash_waste_penalty = 0.0

        if is_flash_action:
            # 危险时闪现，并且闪现后远离怪兽：给基础奖励 + 距离增量奖励
            if in_flash_danger and monster_progress > Config.FLASH_ESCAPE_PROGRESS_TH:
                flash_escape_reward = (
                    Config.REWARD_FLASH_ESCAPE_BASE
                    + Config.REWARD_FLASH_ESCAPE * monster_progress
                )

            # 闪现后反而更靠近怪兽：固定惩罚，不再乘很小的 progress
            elif curr_min_dist_norm < prev_min_dist_norm:
                flash_toward_monster_penalty = -Config.PENALTY_FLASH_TOWARD_MONSTER

            # 不危险时乱用闪现
            elif not in_flash_danger:
                flash_waste_penalty = -Config.PENALTY_FLASH_WASTE
        
        # ------------------------------------------------------------------
        # First seen treasure reward
        # 第一次看见某个宝箱位置时给奖励
        # ------------------------------------------------------------------
        first_seen_treasure_reward = 0.0
        new_seen_treasure_count = 0

        if not hasattr(self, "seen_treasure_keys"):
            self.seen_treasure_keys = set()

        for organ in curr_frame_state.get("organs", []):
            if not isinstance(organ, dict):
                continue

            if int(organ.get("status", 0)) <= 0:
                continue

            if int(organ.get("sub_type", -1)) != Config.ORGAN_TYPE_TREASURE:
                continue

            pos = organ.get("pos", {})
            if "x" not in pos or "z" not in pos:
                continue

            treasure_key = (
                int(round(float(pos["x"]))),
                int(round(float(pos["z"]))),
            )

            if treasure_key not in self.seen_treasure_keys:
                self.seen_treasure_keys.add(treasure_key)
                new_seen_treasure_count += 1

        first_seen_treasure_reward = (
            Config.REWARD_FIRST_SEEN_TREASURE * float(new_seen_treasure_count)
        )
        # ------------------------------------------------------------------
        # 2) Treasure collect reward from extra_info/env_info
        # ------------------------------------------------------------------
        prev_treasure_count = float(prev_env_info.get("treasures_collected", 0.0))
        curr_treasure_count = float(curr_env_info.get("treasures_collected", 0.0))
        treasure_delta = max(0.0, curr_treasure_count - prev_treasure_count)

        # 兜底：treasure_id 列表减少也认为吃到宝箱
        if treasure_delta <= 0.0:
            prev_treasure_ids = self._get_treasure_ids(prev_env_info)
            curr_treasure_ids = self._get_treasure_ids(curr_env_info)
            treasure_delta = max(0.0, float(len(prev_treasure_ids) - len(curr_treasure_ids)))

        treasure_score_reward = Config.REWARD_TREASURE_SCORE * treasure_delta

        # 危险区吃宝箱不完全禁止，但降低收益，避免模型为了宝箱送死
        if curr_min_dist_norm <= Config.TREASURE_SAFE_DISTANCE_TH:
            treasure_score_reward *= Config.TREASURE_DANGER_SCORE_SCALE

        # 靠近宝箱交给 global_treasure_field_reward，这里只保留真正吃到宝箱奖励
        treasure_reward = treasure_score_reward

      
        # ------------------------------------------------------------------
        # 3) Buff collect reward from extra_info/env_info
        # ------------------------------------------------------------------
        prev_buff_time = float(prev_obs_frame_state["heroes"].get("buff_remaining_time", 0.0))
        curr_buff_time = float(curr_obs_frame_state["heroes"].get("buff_remaining_time", 0.0))

        prev_collected_buff = float(prev_env_info.get("collected_buff", 0.0))
        curr_collected_buff = float(curr_env_info.get("collected_buff", 0.0))
        buff_collect_delta = max(0.0, curr_collected_buff - prev_collected_buff)

        # 兜底：buff_remaining_time 从 0 变大，也认为吃到 buff
        if buff_collect_delta <= 0.0 and prev_buff_time <= 0.0 and curr_buff_time > 0.0:
            buff_collect_delta = 1.0

        buff_collect_reward = Config.REWARD_BUFF_COLLECT * buff_collect_delta

        # 靠近 Buff 交给 global_buff_field_reward，这里只保留真正吃到 Buff 奖励
        buff_reward = buff_collect_reward

        # ------------------------------------------------------------------
        # 4) Survival reward
        # ------------------------------------------------------------------
        survival_reward = Config.REWARD_SURVIVAL

        # ------------------------------------------------------------------
        # 5) Terminal reward
        # ------------------------------------------------------------------
        done = bool(curr_obs.get("terminated", False) or curr_obs.get("truncated", False))
        terminal_reward = 0.0

        if done:
            is_caught = bool(curr_env_info.get("is_caught", False))
            is_success = bool(curr_env_info.get("is_success", False))
            is_timeout = bool(curr_env_info.get("is_timeout", False))

            if is_caught:
                terminal_reward = Config.TERMINAL_CAUGHT_PENALTY
            elif is_success:
                terminal_reward = Config.TERMINAL_SUCCESS_REWARD
            elif is_timeout:
                terminal_reward = Config.TERMINAL_TIMEOUT_PENALTY
            else:
                curr_step_no = int(curr_observation.get("step_no", 0))
                max_step = int(curr_env_info.get("max_step", 1000))
                terminal_reward = (
                    Config.TERMINAL_SUCCESS_REWARD
                    if curr_step_no >= max_step - 1
                    else Config.TERMINAL_CAUGHT_PENALTY
                )

        # ------------------------------------------------------------------
        # 6) Invalid move penalty
        # ------------------------------------------------------------------
        prev_hx = float(prev_hero_pos.get("x", 0.0))
        prev_hz = float(prev_hero_pos.get("z", 0.0))
        curr_hx = float(curr_hero_pos.get("x", 0.0))
        curr_hz = float(curr_hero_pos.get("z", 0.0))

        disp = np.sqrt((curr_hx - prev_hx) ** 2 + (curr_hz - prev_hz) ** 2)
        invalid_move_penalty = -Config.PENALTY_INVALID_MOVE if disp < 0.2 else 0.0

        # ------------------------------------------------------------------
        # Anti-stuck penalty
        # 如果当前坐标和 N 步前坐标距离太小，说明存在卡墙/磨蹭/局部摩擦
        # ------------------------------------------------------------------
        anti_stuck_penalty = 0.0
        anti_stuck_dist_10 = Config.ANTI_STUCK_DISTANCE_TH

        if not hasattr(self, "position_history"):
            self.position_history = deque(maxlen=Config.ANTI_STUCK_WINDOW + 1)

        self.position_history.append((curr_hx, curr_hz))

        if len(self.position_history) >= Config.ANTI_STUCK_WINDOW + 1:
            old_hx, old_hz = self.position_history[0]
            anti_stuck_dist_10 = float(
                np.sqrt((curr_hx - old_hx) ** 2 + (curr_hz - old_hz) ** 2)
            )

            if anti_stuck_dist_10 < Config.ANTI_STUCK_DISTANCE_TH:
                anti_stuck_penalty = -Config.PENALTY_ANTI_STUCK
        # 只奖励“不靠近怪兽”的移动，避免模型为了 move_reward 乱跑
        if curr_min_dist_norm >= prev_min_dist_norm:
            move_reward = Config.REWARD_MOVE * min(float(disp), 1.0)
        else:
            move_reward = 0.0
        # ------------------------------------------------------------------
        # 7) Repeat visit penalty
        # ------------------------------------------------------------------
        cell = (int(curr_hx // 4), int(curr_hz // 4))
        self.visit_counter[cell] = self.visit_counter.get(cell, 0) + 1

        revisit = self.visit_counter[cell]
        repeat_penalty = 0.0

        if revisit > 3:
            repeat_penalty -= Config.PENALTY_REPEAT_VISIT * min(5, revisit - 3)

        # ------------------------------------------------------------------
        # 8) Stage reward
        # ------------------------------------------------------------------
        curr_step_no = int(curr_observation.get("step_no", 0))
        stage_reward = 0.0

        if 0 < curr_step_no <= 1000 and curr_step_no % 100 == 0:
            stage_reward = 20.0 * (curr_step_no // 100)

        # ------------------------------------------------------------------
        # 9) Env score delta
        # ------------------------------------------------------------------
        prev_total_score = float(prev_env_info.get("total_score", 0.0))
        curr_total_score = float(curr_env_info.get("total_score", 0.0))
        total_score_delta = curr_total_score - prev_total_score

        # ------------------------------------------------------------------
        # 10) Reward scale
        # ------------------------------------------------------------------
        raw_reward = (
            survival_reward
            + terminal_reward

            # 全局势场塑形
            + global_monster_field_reward
            + global_treasure_field_reward
            + global_buff_field_reward

            # 近身硬危险惩罚
            + danger_penalty

            # 闪现动作质量
            + flash_escape_reward
            + flash_toward_monster_penalty
            + flash_waste_penalty

            # 真实事件奖励
            + treasure_reward
            + buff_reward
            + first_seen_treasure_reward

            # 行为约束
            + move_reward
            + invalid_move_penalty
            + repeat_penalty
            + anti_stuck_penalty

            # 阶段与环境分数
            + stage_reward
            + total_score_delta
        )
        reward_scale = float(getattr(Config, "REWARD_SCALE", 0.005))
        scaled_reward = raw_reward * reward_scale
        reward_value = float(scaled_reward)

        reward_components = {
            "survival_reward": float(survival_reward * reward_scale),
            "terminal_reward": float(terminal_reward * reward_scale),

            # global field shaping
            "global_monster_field_reward": float(global_monster_field_reward * reward_scale),
            "global_treasure_field_reward": float(global_treasure_field_reward * reward_scale),
            "global_buff_field_reward": float(global_buff_field_reward * reward_scale),

            # 兼容旧 monitor：dist_shaping 现在表示全局怪兽势场
            "dist_shaping": float(global_monster_field_reward * reward_scale),

            "danger_penalty": float(danger_penalty * reward_scale),
            "monster_speed_factor": float(monster_speed_factor),

            "treasure_reward": float(treasure_reward * reward_scale),
            "treasure_score_reward": float(treasure_score_reward * reward_scale),
            "first_seen_treasure_reward": float(first_seen_treasure_reward * reward_scale),
            "debug_new_seen_treasure_count": float(new_seen_treasure_count),

            "buff_reward": float(buff_reward * reward_scale),
            "buff_collect_reward": float(buff_collect_reward * reward_scale),

            "invalid_move_penalty": float(invalid_move_penalty * reward_scale),
            "repeat_penalty": float(repeat_penalty * reward_scale),
            "stage_reward": float(stage_reward * reward_scale),
            "total_score_delta": float(total_score_delta * reward_scale),

            "raw_reward_total": float(raw_reward),
            "scaled_reward_total": float(scaled_reward),
            "reward_total": float(reward_value),

            # debug: monster distance
            "debug_prev_monster_dist": float(prev_min_dist_norm),
            "debug_curr_monster_dist": float(curr_min_dist_norm),
            "debug_prev_has_monster": float(prev_has_monster),
            "debug_curr_has_monster": float(curr_has_monster),
            "debug_monster_progress": float(monster_progress),

            # debug: global field
            "debug_prev_monster_heat": float(prev_monster_heat),
            "debug_curr_monster_heat": float(curr_monster_heat),
            "debug_monster_heat_progress": float(monster_field_progress),

            "debug_prev_treasure_heat": float(prev_treasure_heat),
            "debug_curr_treasure_heat": float(curr_treasure_heat),
            "debug_treasure_heat_progress": float(treasure_field_progress),

            "debug_prev_buff_heat": float(prev_buff_heat),
            "debug_curr_buff_heat": float(curr_buff_heat),
            "debug_buff_heat_progress": float(buff_field_progress),

            # debug: real events
            "debug_treasure_delta": float(treasure_delta),
            "debug_prev_buff_time": float(prev_buff_time),
            "debug_curr_buff_time": float(curr_buff_time),
            "debug_buff_collect_delta": float(buff_collect_delta),

            "move_reward": float(move_reward * reward_scale),

            "flash_escape_reward": float(flash_escape_reward * reward_scale),
            "flash_toward_monster_penalty": float(flash_toward_monster_penalty * reward_scale),
            "flash_waste_penalty": float(flash_waste_penalty * reward_scale),

            "debug_is_flash_action": float(is_flash_action),
            "debug_in_flash_danger": float(in_flash_danger),
            "anti_stuck_penalty": float(anti_stuck_penalty * reward_scale),
            "debug_anti_stuck_dist_10": float(anti_stuck_dist_10),
        }

        self.last_reward_components = dict(reward_components)
        self.last_env_info = dict(curr_env_info) if isinstance(curr_env_info, dict) else {}

        return reward_value, reward_components

  
    
    def _build_monster_features(self, frame_state):
        hero = frame_state.get("heroes", {})
        hero_pos = hero.get("pos", {})
        hx = float(hero_pos.get("x", 0.0))
        hz = float(hero_pos.get("z", 0.0))

        monsters = frame_state.get("monsters", [])
        feats = []

        for i in range(2):
            if i >= len(monsters) or not isinstance(monsters[i], dict):
                feats.extend([0.0, 0.0, 0.0, 1.0, 0.0, 1.0])
                continue

            m = monsters[i]
            pos = m.get("pos", {})

            mx = float(pos.get("x", -1)) if isinstance(pos, dict) else -1
            mz = float(pos.get("z", -1)) if isinstance(pos, dict) else -1

            if mx < 0 or mz < 0:
                feats.extend([0.0, 0.0, 0.0, 1.0, 0.0, 1.0])
                continue

            dx = mx - hx
            dz = mz - hz
            dist = np.sqrt(dx * dx + dz * dz)

            if dist > 1e-6:
                dir_x = float(np.clip(dx / dist, -1.0, 1.0))
                dir_z = float(np.clip(dz / dist, -1.0, 1.0))
            else:
                dir_x, dir_z = 0.0, 0.0

            dist_norm = float(np.clip(dist / (Config.MAP_SIZE* 1.41), 0.0, 1.0))

            speed = float(m.get("speed", 0.0))
            speed_norm = 0.0 if speed < 0 else _norm(speed, Config.MAX_MONSTER_SPEED)

            interval = float(m.get("monster_interval", 300.0))
            interval_norm = _norm(interval, 300.0)

            feats.extend([
                1.0,        # valid_info
                dir_x,
                dir_z,
                dist_norm,
                speed_norm,
                interval_norm,
            ])

        return np.array(feats, dtype=np.float32)
    def _build_status_features(self, frame_state, env_info):
        hero = frame_state.get("heroes", {})

        step_no = float(env_info.get("step_no", 0))
        max_step = float(env_info.get("max_step", 1000))

        step_norm = _norm(step_no, max_step)

        total_treasure = float(env_info.get("total_treasure", 10))
        treasures_collected = float(env_info.get("treasures_collected", 0))
        treasure_ratio = treasures_collected / max(total_treasure, 1.0)

        treasure_score = float(env_info.get("treasure_score", 0))
        treasure_score_norm = _norm(treasure_score, 1000.0)

        total_buff = float(env_info.get("total_buff", 2))
        collected_buff = float(env_info.get("collected_buff", 0))
        buff_ratio = collected_buff / max(total_buff, 1.0)

        buff_remaining = float(hero.get("buff_remaining_time", 0))
        buff_remaining_norm = _norm(buff_remaining, Config.MAX_BUFF_DURATION)

        return np.array(
            [
                step_norm,
                treasure_ratio,
                treasure_score_norm,
                buff_ratio,
                buff_remaining_norm,
                float(step_no / max(max_step, 1.0)),
            ],
            dtype=np.float32,
        )

    def _build_anti_stuck_feature(self, hero_pos):
        """
        Build anti-stuck feature.

        Returns:
            [dx10_norm, dz10_norm]

        含义：
            当前英雄位置相对 N 步前位置的位移。
            如果历史不足 N 步，则返回 [0, 0]。
        """
        curr_x = float(hero_pos.get("x", 0.0))
        curr_z = float(hero_pos.get("z", 0.0))

        if (
            not hasattr(self, "position_history")
            or len(self.position_history) < Config.ANTI_STUCK_WINDOW+1
        ):
            return np.array([0.0, 0.0], dtype=np.float32)

        old_x, old_z = self.position_history[0]

        dx10_norm = float(np.clip((curr_x - old_x) / Config.MAP_SIZE, -1.0, 1.0))
        dz10_norm = float(np.clip((curr_z - old_z) / Config.MAP_SIZE, -1.0, 1.0))

        return np.array([dx10_norm, dz10_norm], dtype=np.float32)

    # ======================================================================
    # Scalar feature helpers
    # ======================================================================
   
    def _min_monster_dist_norm(self, frame_state):
        hero = frame_state.get("heroes", {})
        hero_pos = hero.get("pos", {})

        hx = float(hero_pos.get("x", 0.0))
        hz = float(hero_pos.get("z", 0.0))

        min_dist_norm = 1.0

        for m in frame_state.get("monsters", []):
            if not isinstance(m, dict):
                continue

            pos = m.get("pos", {})
            if "x" not in pos or "z" not in pos:
                continue

            mx = float(pos.get("x", -1))
            mz = float(pos.get("z", -1))

            if mx < 0 or mz < 0:
                continue

            dx = mx - hx
            dz = mz - hz
            dist = np.sqrt(dx * dx + dz * dz)

            dist_norm = float(np.clip(dist / (Config.MAP_SIZE* 1.41), 0.0, 1.0))
            min_dist_norm = min(min_dist_norm, dist_norm)

        return float(min_dist_norm)

    def _get_treasure_ids(self, env_info):
        treasure_ids = env_info.get("treasure_id", [])
        if not isinstance(treasure_ids, list):
            return []
        return treasure_ids

    def get_last_reward_components(self):
        return dict(self.last_reward_components)

    def get_last_env_info(self):
        return dict(self.last_env_info)

   
    def _visible_monster_speed_factor(self, frame_state):
        monsters = frame_state.get("monsters", [])
        max_speed_norm = 0.0

        for m in monsters:
            if not isinstance(m, dict):
                continue

            pos = m.get("pos", {})
            mx = float(pos.get("x", -1)) if isinstance(pos, dict) else -1
            mz = float(pos.get("z", -1)) if isinstance(pos, dict) else -1

            if mx < 0 or mz < 0:
                continue

            speed = float(m.get("speed", 1.0))
            if speed < 0:
                continue

            max_speed_norm = max(max_speed_norm, _norm(speed, Config.MAX_MONSTER_SPEED))

        return 1.0 + max_speed_norm
  
    def _get_extra_frame_state(self, env_obs):
        """
        extra_info 允许使用时，优先使用 extra_info.frame_state。
        如果不存在，则回退到 observation.frame_state。
        """
        if isinstance(env_obs, dict):
            extra_info = env_obs.get("extra_info", {})
            if isinstance(extra_info, dict):
                frame_state = extra_info.get("frame_state", None)
                if isinstance(frame_state, dict):
                    return frame_state

            observation = env_obs.get("observation", {})
            if isinstance(observation, dict):
                return observation.get("frame_state", {})

        return {}
    def _build_local_obstacle_map(self, map_info):
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

        return local_map
   
    def _organ_global_to_local(self, hero_pos, organ_pos):
        """
        Convert global organ position to local 21x21 map index.

        Returns:
            (rr, cc) if inside local map, otherwise None.
        """
        center = Config.LOCAL_MAP_WINDOW // 2

        hx = float(hero_pos.get("x", 0.0))
        hz = float(hero_pos.get("z", 0.0))

        ox = float(organ_pos.get("x", -1))
        oz = float(organ_pos.get("z", -1))

        if ox < 0 or oz < 0:
            return None

        dx = ox - hx
        dz = oz - hz

        if abs(dx) > center or abs(dz) > center:
            return None

        cc = center + int(round(dx))
        rr = center + int(round(dz))

        if 0 <= rr < Config.LOCAL_MAP_WINDOW and 0 <= cc < Config.LOCAL_MAP_WINDOW:
            return rr, cc

        return None


    def _get_local_organ_targets(self, frame_state, organs, sub_type):
        """
        Get local map cells of organs with given sub_type.
        """
        hero = frame_state.get("heroes", {})
        hero_pos = hero.get("pos", {})

        targets = []

        for organ in organs:
            if not isinstance(organ, dict):
                continue

            if int(organ.get("status", 0)) <= 0:
                continue

            if int(organ.get("sub_type", -1)) != int(sub_type):
                continue

            pos = organ.get("pos", {})
            if "x" not in pos or "z" not in pos:
                continue

            local_pos = self._organ_global_to_local(hero_pos, pos)
            if local_pos is not None:
                targets.append(local_pos)

        return targets


    def _local_walkable_grid(self, map_info):
        """
        Build walkable grid from local map_info.

        map_info:
            0 = obstacle
            1 = walkable
        """
        if map_info is None or len(map_info) == 0 or len(map_info[0]) == 0:
            return None

        map_arr = np.array(map_info)

        if map_arr.shape != (Config.LOCAL_MAP_WINDOW, Config.LOCAL_MAP_WINDOW):
            return None

        return map_arr == Config.MAP_VALUE_WALKABLE


    def _shortest_path_to_targets(self, walkable, targets):
        """
        Dijkstra on local 21x21 grid.

        Returns:
            found: bool
            path_cost: float
            first_step: tuple[int, int] or None
                first_step = (dr, dc), direction from center to next cell.
        """
        if walkable is None or not targets:
            return False, 1.0, None

        center = Config.LOCAL_MAP_WINDOW // 2
        start = (center, center)
        target_set = set(targets)

        if start in target_set:
            return True, 0.0, (0, 0)

        walkable = walkable.copy()
        walkable[start[0], start[1]] = True

        # 允许目标格可达，避免目标格因为地图编码边界问题被视为不可走
        for rr, cc in targets:
            if 0 <= rr < Config.LOCAL_MAP_WINDOW and 0 <= cc < Config.LOCAL_MAP_WINDOW:
                walkable[rr, cc] = True

        inf = 1e9
        dist = np.full(
            (Config.LOCAL_MAP_WINDOW, Config.LOCAL_MAP_WINDOW),
            inf,
            dtype=np.float32,
        )
        dist[start[0], start[1]] = 0.0

        # 8方向移动，斜向代价 sqrt(2)
        directions = [
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (-1, -1, 1.4142),
            (-1, 1, 1.4142),
            (1, -1, 1.4142),
            (1, 1, 1.4142),
        ]

        heap = []

        for dr, dc, cost in directions:
            nr = start[0] + dr
            nc = start[1] + dc

            if not (0 <= nr < Config.LOCAL_MAP_WINDOW and 0 <= nc < Config.LOCAL_MAP_WINDOW):
                continue

            if not walkable[nr, nc]:
                continue

            dist[nr, nc] = cost
            heapq.heappush(heap, (cost, nr, nc, dr, dc))

        while heap:
            cost, r, c, first_dr, first_dc = heapq.heappop(heap)

            if cost > dist[r, c] + 1e-6:
                continue

            if (r, c) in target_set:
                return True, float(cost), (int(first_dr), int(first_dc))

            for dr, dc, move_cost in directions:
                nr = r + dr
                nc = c + dc

                if not (0 <= nr < Config.LOCAL_MAP_WINDOW and 0 <= nc < Config.LOCAL_MAP_WINDOW):
                    continue

                if not walkable[nr, nc]:
                    continue

                new_cost = cost + move_cost

                if new_cost < dist[nr, nc]:
                    dist[nr, nc] = new_cost
                    heapq.heappush(heap, (new_cost, nr, nc, first_dr, first_dc))

        return False, 1.0, None
    def _nearest_organ_path_direction_feature(self, map_info, frame_state, organs, sub_type):
        """
        Return [dir_x, dir_z, dist_norm] based on local reachable path.

        If no reachable target exists inside local 21x21 map, fallback to Euclidean direction.
        """
        walkable = self._local_walkable_grid(map_info)
        targets = self._get_local_organ_targets(frame_state, organs, sub_type)

        found, path_cost, first_step = self._shortest_path_to_targets(walkable, targets)

        if found and first_step is not None:
            dr, dc = first_step

            if dr == 0 and dc == 0:
                return np.array([0.0, 0.0, 0.0], dtype=np.float32)

            norm = np.sqrt(float(dr * dr + dc * dc))
            dir_x = float(np.clip(dc / norm, -1.0, 1.0))
            dir_z = float(np.clip(dr / norm, -1.0, 1.0))

            # 局部 21x21 内的路径代价归一化
            dist_norm = float(
                np.clip(
                    path_cost / max(Config.LOCAL_MAP_WINDOW * 1.5, 1.0),
                    0.0,
                    1.0,
                )
            )

            return np.array([dir_x, dir_z, dist_norm], dtype=np.float32)

        # fallback：局部不可达或目标不在局部视野内时，继续用全局欧氏方向
        return self._nearest_organ_direction_feature(frame_state, organs, sub_type)


    
    def _make_gaussian_kernel(self, sigma):
        radius = int(np.ceil(3 * sigma))
        ax = np.arange(-radius, radius + 1, dtype=np.float32)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
        kernel /= np.max(kernel) + 1e-8
        return kernel.astype(np.float32)


    def _paint_gaussian(self, grid, channel, r, c, sigma):
        kernel = self._make_gaussian_kernel(sigma)
        radius = kernel.shape[0] // 2

        h, w = grid.shape[1], grid.shape[2]

        r0 = max(0, r - radius)
        r1 = min(h, r + radius + 1)
        c0 = max(0, c - radius)
        c1 = min(w, c + radius + 1)

        kr0 = r0 - (r - radius)
        kr1 = kr0 + (r1 - r0)
        kc0 = c0 - (c - radius)
        kc1 = kc0 + (c1 - c0)

        patch = kernel[kr0:kr1, kc0:kc1]

        grid[channel, r0:r1, c0:c1] = np.maximum(
            grid[channel, r0:r1, c0:c1],
            patch,
        )


    def _build_global_entity_map(self, frame_state):
        global_map = np.zeros(
            (
                Config.GLOBAL_MAP_CHANNELS,
                Config.GLOBAL_MAP_SIZE,
                Config.GLOBAL_MAP_SIZE,
            ),
            dtype=np.float32,
        )

        def to_grid(pos):
            if not isinstance(pos, dict):
                return None

            x = float(pos.get("x", -1))
            z = float(pos.get("z", -1))

            if x < 0 or z < 0:
                return None

            gx = int(np.clip(round(x), 0, Config.GLOBAL_MAP_SIZE - 1))
            gz = int(np.clip(round(z), 0, Config.GLOBAL_MAP_SIZE - 1))

            return gz, gx

        hero = frame_state.get("heroes", {})
        cell = to_grid(hero.get("pos", {}))
        if cell is not None:
            r, c = cell
            self._paint_gaussian(
                global_map,
                Config.GLOBAL_CHANNEL_HERO,
                r,
                c,
                Config.HERO_SIGMA,
            )

        for organ in frame_state.get("organs", []):
            if not isinstance(organ, dict):
                continue

            if int(organ.get("status", 0)) <= 0:
                continue

            cell = to_grid(organ.get("pos", {}))
            if cell is None:
                continue

            r, c = cell
            sub_type = int(organ.get("sub_type", -1))

            if sub_type == Config.ORGAN_TYPE_TREASURE:
                self._paint_gaussian(
                    global_map,
                    Config.GLOBAL_CHANNEL_TREASURE,
                    r,
                    c,
                    Config.TREASURE_SIGMA,
                )
            elif sub_type == Config.ORGAN_TYPE_BUFF:
                self._paint_gaussian(
                    global_map,
                    Config.GLOBAL_CHANNEL_BUFF,
                    r,
                    c,
                    Config.BUFF_SIGMA,
                )

        for monster in frame_state.get("monsters", []):
            if not isinstance(monster, dict):
                continue

            cell = to_grid(monster.get("pos", {}))
            if cell is None:
                continue

            r, c = cell

            speed = float(monster.get("speed", 1.0))
            speed_norm = np.clip(speed / max(Config.MAX_MONSTER_SPEED, 1e-6), 0.0, 1.0)
            sigma = Config.MONSTER_SIGMA + Config.MONSTER_SIGMA_SPEED_COEF * speed_norm

            self._paint_gaussian(
                global_map,
                Config.GLOBAL_CHANNEL_MONSTER,
                r,
                c,
                sigma,
            )

        return global_map
    def _gaussian_value_at_distance(self, dist, sigma):
        sigma = max(float(sigma), 1e-6)
        return float(np.exp(-(dist * dist) / (2.0 * sigma * sigma)))
   
    def _entity_field_value_at_hero(self, frame_state, sub_type=None, entity_type="organ"):
        """
        Return hero's value in a global entity potential field.

        注意：
        这里用于 reward 势场，不复用 CNN heatmap sigma。
        CNN sigma 负责定位，reward sigma 负责更大范围的方向引导。
        """
        hero = frame_state.get("heroes", {})
        hero_pos = hero.get("pos", {})

        hx = float(hero_pos.get("x", 0.0))
        hz = float(hero_pos.get("z", 0.0))

        best_heat = 0.0

        if entity_type == "monster":
            for m in frame_state.get("monsters", []):
                if not isinstance(m, dict):
                    continue

                pos = m.get("pos", {})
                if "x" not in pos or "z" not in pos:
                    continue

                mx = float(pos.get("x", -1))
                mz = float(pos.get("z", -1))
                if mx < 0 or mz < 0:
                    continue

                speed = float(m.get("speed", 1.0))
                speed_norm = np.clip(
                    speed / max(Config.MAX_MONSTER_SPEED, 1e-6),
                    0.0,
                    1.0,
                )

                sigma = Config.REWARD_MONSTER_FIELD_SIGMA * (1.0 + 0.3 * speed_norm)

                dist = np.sqrt((mx - hx) ** 2 + (mz - hz) ** 2)
                best_heat = max(
                    best_heat,
                    self._gaussian_value_at_distance(dist, sigma),
                )

            return float(best_heat)

        for organ in frame_state.get("organs", []):
            if not isinstance(organ, dict):
                continue
            if int(organ.get("status", 0)) <= 0:
                continue
            if int(organ.get("sub_type", -1)) != int(sub_type):
                continue

            pos = organ.get("pos", {})
            if "x" not in pos or "z" not in pos:
                continue

            ox = float(pos.get("x", -1))
            oz = float(pos.get("z", -1))
            if ox < 0 or oz < 0:
                continue

            if int(sub_type) == Config.ORGAN_TYPE_TREASURE:
                sigma = Config.REWARD_TREASURE_FIELD_SIGMA
            elif int(sub_type) == Config.ORGAN_TYPE_BUFF:
                sigma = Config.REWARD_BUFF_FIELD_SIGMA
            else:
                sigma = 20.0

            dist = np.sqrt((ox - hx) ** 2 + (oz - hz) ** 2)
            best_heat = max(
                best_heat,
                self._gaussian_value_at_distance(dist, sigma),
            )

        return float(best_heat)
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
# Max distance bucket / 距离桶最大值
MAX_DIST_BUCKET = 5.0
# Max flash cooldown / 最大闪现冷却步数
MAX_FLASH_CD = 2000.0
# Max buff duration / buff最大持续时间
MAX_BUFF_DURATION = 50.0
# Max acceleration / 最大加速度
MAX_ACCELERATION = 2.0
# Max relative velocity / 最大相对速度
MAX_REL_VELOCITY = 10.0
# Max collision risk / 最大碰撞风险
MAX_COLLISION_RISK = 10.0


def _norm(v, v_max, v_min=0.0):
    """Normalize value to [0, 1].

    将值归一化到 [0, 1]。
    """
    v = float(np.clip(v, v_min, v_max))
    return (v - v_min) / (v_max - v_min) if (v_max - v_min) > 1e-6 else 0.0


class Preprocessor:
    def __init__(self):
        self.reset()
        # 存储历史信息用于轨迹预测
        self.monster_history = []  # 存储过去几步的怪物信息
        self.history_length = 5  # 保留5步历史

    def reset(self):
        self.step_no = 0
        self.max_step = 200
        self.last_min_monster_dist_norm = 0.5
        self.monster_history = []

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

        # Monster features with advanced features (10D x 2) / 怪物特征（含高级特征）
        monsters = frame_state.get("monsters", [])
        monster_feats = []
        current_monster_info = []
        
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
                    
                    # Store for trajectory prediction
                    current_monster_info.append({
                        'pos': m_pos,
                        'speed': m.get("speed", 1),
                        'dist': raw_dist
                    })
                else:
                    m_x_norm = 0.0
                    m_z_norm = 0.0
                    m_speed_norm = 0.0
                    dist_norm = 1.0
                    current_monster_info.append(None)
                
                # Advanced features for visible monsters
                if is_in_view > 0:
                    # Relative velocity features (2D)
                    rel_vel_x, rel_vel_z = self._calculate_relative_velocity(i, m_pos)
                    rel_vel_x_norm = _norm(rel_vel_x, MAX_REL_VELOCITY)
                    rel_vel_z_norm = _norm(rel_vel_z, MAX_REL_VELOCITY)
                    
                    # Trajectory prediction features (2D)
                    pred_x, pred_z = self._predict_monster_trajectory(i, m_pos)
                    pred_dist = np.sqrt((hero_pos["x"] - pred_x) ** 2 + (hero_pos["z"] - pred_z) ** 2)
                    pred_dist_norm = _norm(pred_dist, MAP_SIZE * 1.41)
                    
                    # Collision risk feature (1D)
                    collision_risk = self._calculate_collision_risk(
                        hero_pos, m_pos, m.get("speed", 1), raw_dist
                    )
                    collision_risk_norm = _norm(collision_risk, MAX_COLLISION_RISK)
                    
                    monster_feats.append(
                        np.array([
                            is_in_view, m_x_norm, m_z_norm, m_speed_norm, dist_norm,
                            rel_vel_x_norm, rel_vel_z_norm, pred_dist_norm, collision_risk_norm
                        ], dtype=np.float32)
                    )
                else:
                    monster_feats.append(
                        np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
                    )
            else:
                monster_feats.append(np.zeros(9, dtype=np.float32))
                current_monster_info.append(None)
        
        # Update monster history for next step
        self._update_monster_history(current_monster_info)

        # Local map features (16D) / 局部地图特征
        map_feat = np.zeros(16, dtype=np.float32)
        if map_info is not None and len(map_info) >= 13:
            center = len(map_info) // 2
            flat_idx = 0
            for row in range(center - 2, center + 2):
                for col in range(center - 2, center + 2):
                    if 0 <= row < len(map_info) and 0 <= col < len(map_info[0]):
                        map_feat[flat_idx] = float(map_info[row][col] != 0)
                    flat_idx += 1

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

        # Progress features (4D) / 进度特征（增加了高级特征）
        step_norm = _norm(self.step_no, self.max_step)
        survival_ratio = step_norm
        
        # Time-based features / 时间相关特征
        time_pressure = _norm(self.step_no / max(self.max_step, 1), 1.0)  # 时间压力
        survival_advantage = _norm(self.step_no - self.max_step * 0.5, self.max_step)  # 生存优势
        
        progress_feat = np.array([
            step_norm, survival_ratio, time_pressure, survival_advantage
        ], dtype=np.float32)

        # Concatenate features / 拼接特征
        feature = np.concatenate(
            [
                hero_feat,
                monster_feats[0],
                monster_feats[1],
                map_feat,
                np.array(legal_action, dtype=np.float32),
                progress_feat,
            ]
        )

        # Advanced reward design / 高级奖励设计
        cur_min_dist_norm = 1.0
        max_collision_risk = 0.0
        for m_feat in monster_feats:
            if m_feat[0] > 0:
                cur_min_dist_norm = min(cur_min_dist_norm, m_feat[4])
                max_collision_risk = max(max_collision_risk, m_feat[8])  # 使用碰撞风险特征

        # Base survival reward / 基础生存奖励
        survive_reward = 0.01
        
        # Distance shaping reward / 距离塑形奖励
        dist_shaping = 0.1 * (cur_min_dist_norm - self.last_min_monster_dist_norm)
        
        # Collision risk penalty / 碰撞风险惩罚
        risk_penalty = -0.05 * max_collision_risk
        
        # Milestone rewards / 阶段性里程碑奖励
        milestone_reward = 0.0
        if self.step_no == 50:
            milestone_reward = 0.5  # 存活50步奖励
        elif self.step_no == 100:
            milestone_reward = 1.0  # 存活100步奖励
        elif self.step_no == 150:
            milestone_reward = 2.0  # 存活150步奖励
        elif self.step_no == 200:
            milestone_reward = 5.0  # 存活200步奖励
            
        # Emergency avoidance reward / 紧急避险奖励
        avoidance_reward = 0.0
        if (self.last_min_monster_dist_norm < 0.3 and 
            cur_min_dist_norm > self.last_min_monster_dist_norm + 0.1):
            avoidance_reward = 0.2  # 成功远离危险区域奖励
            
        # Time pressure bonus / 时间压力奖励
        time_bonus = 0.005 * step_norm  # 随时间增加的生存奖励

        self.last_min_monster_dist_norm = cur_min_dist_norm

        # Total reward / 总奖励
        total_reward = (
            survive_reward + 
            dist_shaping + 
            risk_penalty + 
            milestone_reward + 
            avoidance_reward + 
            time_bonus
        )
        
        reward = [total_reward]

        return feature, legal_action, reward

    def _update_monster_history(self, current_monsters):
        """Update monster position history for trajectory prediction.
        
        更新怪物位置历史用于轨迹预测。
        """
        self.monster_history.append(current_monsters)
        if len(self.monster_history) > self.history_length:
            self.monster_history.pop(0)
    
    def _calculate_relative_velocity(self, monster_idx, current_pos):
        """Calculate relative velocity between hero and monster.
        
        计算英雄与怪物之间的相对速度。
        """
        if len(self.monster_history) < 2:
            return 0.0, 0.0
        
        # Get previous position
        prev_info = self.monster_history[-2]
        if (prev_info is None or monster_idx >= len(prev_info) or 
            prev_info[monster_idx] is None):
            return 0.0, 0.0
        
        prev_pos = prev_info[monster_idx]['pos']
        current_pos_dict = current_pos
        
        # Calculate velocity (assuming 1 step time difference)
        vel_x = current_pos_dict['x'] - prev_pos['x']
        vel_z = current_pos_dict['z'] - prev_pos['z']
        
        return vel_x, vel_z
    
    def _predict_monster_trajectory(self, monster_idx, current_pos):
        """Predict monster future position based on trajectory.
        
        基于轨迹预测怪物未来位置。
        """
        if len(self.monster_history) < 2:
            return current_pos['x'], current_pos['z']
        
        # Simple linear prediction: current_pos + velocity
        vel_x, vel_z = self._calculate_relative_velocity(monster_idx, current_pos)
        
        # Predict 3 steps ahead
        pred_x = current_pos['x'] + vel_x * 3
        pred_z = current_pos['z'] + vel_z * 3
        
        # Clamp to map boundaries
        pred_x = np.clip(pred_x, 0, MAP_SIZE)
        pred_z = np.clip(pred_z, 0, MAP_SIZE)
        
        return pred_x, pred_z
    
    def _calculate_collision_risk(self, hero_pos, monster_pos, monster_speed, distance):
        """Calculate collision risk based on relative positions and velocities.
        
        基于相对位置和速度计算碰撞风险。
        """
        if distance < 1e-6:
            return MAX_COLLISION_RISK
        
        # Calculate direction vector from hero to monster
        dx = monster_pos['x'] - hero_pos['x']
        dz = monster_pos['z'] - hero_pos['z']
        
        # Normalize direction
        dist = np.sqrt(dx*dx + dz*dz)
        if dist > 0:
            dx /= dist
            dz /= dist
        
        # Get monster velocity
        vel_x, vel_z = self._calculate_relative_velocity(0, monster_pos)
        
        # Calculate approach velocity (negative means approaching)
        approach_vel = -(vel_x * dx + vel_z * dz)
        
        # Risk increases with speed and decreases with distance
        risk = (monster_speed + max(0, approach_vel)) / max(distance, 1.0)
        
        return risk

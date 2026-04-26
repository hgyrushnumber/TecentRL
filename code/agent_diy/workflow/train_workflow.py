#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Training workflow for Gorge Chase SAC.
峡谷追猎 SAC 训练工作流。
"""

import os
import time


import numpy as np
from agent_diy.feature.definition import SampleData
from tools.metrics_utils import get_training_metrics
from tools.train_env_conf_validate import read_usr_conf
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    last_save_model_time = time.time()
    env = envs[0]
    agent = agents[0]

    usr_conf = read_usr_conf("agent_diy/conf/train_env_conf.toml", logger)
    if usr_conf is None:
        logger.error("usr_conf is None, please check agent_diy/conf/train_env_conf.toml")
        return

    episode_runner = EpisodeRunner(
        env=env,
        agent=agent,
        usr_conf=usr_conf,
        logger=logger,
        monitor=monitor,
    )

    while True:
        for g_data in episode_runner.run_episodes():
            if not g_data:
                continue

            agent.send_sample_data(g_data)
            g_data.clear()

            now = time.time()
            if now - last_save_model_time >= 1800:
                agent.save_model()
                last_save_model_time = now


class EpisodeRunner:
    def __init__(self, env, agent, usr_conf, logger, monitor):
        self.env = env
        self.agent = agent
        self.usr_conf = usr_conf
        self.logger = logger
        self.monitor = monitor
        self.episode_cnt = 0
        self.last_report_monitor_time = 0
        self.last_get_training_metrics_time = 0

    def run_episodes(self):
        """Run episodes and yield collected samples."""
        while True:
            now = time.time()
            if now - self.last_get_training_metrics_time >= 60:
                self.last_get_training_metrics_time = now
                
            env_reset_result = self.env.reset(usr_conf=self.usr_conf)
            # self.logger.info(f"env.reset() returned type: {type(env_reset_result)}")
            env_obs = env_reset_result

            if handle_disaster_recovery(env_obs, self.logger):
                continue

            self.agent.reset(env_obs)
            self.agent.load_model(id="latest")

            obs_data, _ = self.agent.observation_process(env_obs)
            if obs_data is None:
                self.logger.error("observation_process returned None on reset")
                continue

            collector = []
            self.episode_cnt += 1
            done = False
            step = 0
            total_reward = 0.0
            reward_component_sums = {}
            reward_debug_latest = {}
            reward_debug_max = {}
            reward_debug_abs_sum = {}
           
            self.logger.info(f"Episode {self.episode_cnt} start")

            obs = env_obs

            while not done:
                try:
                    predict_ret = self.agent.predict(list_obs_data=[obs_data])
                    if predict_ret is None or len(predict_ret) == 0:
                        self.logger.error("agent.predict returned empty result")
                        break
                    act_data = predict_ret[0]
                except Exception as e:
                    self.logger.error(f"predict() Exception {e}")
                    break

                try:
                    act = self.agent.action_process(act_data)
                except Exception as e:
                    self.logger.error(f"action_process() Exception {e}")
                    break

                try:
                    _env_reward, _obs = self.env.step(act)
                except Exception as e:
                    self.logger.error(f"env.step() Exception {e}")
                    break

                if handle_disaster_recovery(_obs, self.logger):
                    break

                step += 1
                terminated = _obs["terminated"]
                truncated = _obs["truncated"]
                done = terminated or truncated

                try:
                    reward, reward_components = self.agent.preprocessor.compute_reward_from_transition(
                        prev_obs=obs,
                        curr_obs=_obs,
                        action=act,
                    )
                except Exception as e:
                    self.logger.error(f"compute_reward_from_transition Exception {e}")
                    break

                reward = np.array([reward], dtype=np.float32)

                try:
                    next_obs_data, next_remain_info = self.agent.observation_process(_obs)
                except Exception as e:
                    self.logger.error(f"observation_process(next) Exception {e}")
                    break

                if next_obs_data is None:
                    self.logger.error("next observation_process returned None")
                    break

                total_reward += float(reward[0])

                for k, v in reward_components.items():
                    reward_component_sums[k] = reward_component_sums.get(k, 0.0) + float(v)

                debug_keys_latest = [
                    "debug_prev_monster_dist",
                    "debug_curr_monster_dist",
                    "debug_monster_progress",

                    "debug_prev_monster_heat",
                    "debug_curr_monster_heat",
                    "debug_monster_heat_progress",

                    "debug_prev_treasure_heat",
                    "debug_curr_treasure_heat",
                    "debug_treasure_heat_progress",

                    "debug_prev_buff_heat",
                    "debug_curr_buff_heat",
                    "debug_buff_heat_progress",

                    "debug_anti_stuck_dist_10",

                    "monster_speed_factor",
                ]

                debug_keys_max = [
                    "debug_curr_has_monster",
                    "debug_is_flash_action",
                    "debug_in_flash_danger",
                ]
                debug_keys_abs_sum = [
                    "debug_monster_progress",
                    "debug_monster_heat_progress",
                    "debug_treasure_heat_progress",
                    "debug_buff_heat_progress",
                ]


                for k in debug_keys_latest:
                    if k in reward_components:
                        reward_debug_latest[k] = float(reward_components[k])

                for k in debug_keys_max:
                    if k in reward_components:
                        reward_debug_max[k] = max(
                            reward_debug_max.get(k, 0.0),
                            float(reward_components[k]),
                        )
                for k in debug_keys_abs_sum:
                    if k in reward_components:
                        reward_debug_abs_sum[k] = reward_debug_abs_sum.get(k, 0.0) + abs(
                            float(reward_components[k])
                        )

                frame = SampleData(
                    obs=np.array(obs_data.feature, dtype=np.float32),
                    legal_action=np.array(obs_data.legal_action, dtype=np.float32),
                    act=np.array([act_data.action[0]], dtype=np.float32),
                    reward=reward.copy(),
                    done=np.array([float(done)], dtype=np.float32),
                    next_obs=np.array(next_obs_data.feature, dtype=np.float32),
                    next_legal_action=np.array(next_obs_data.legal_action, dtype=np.float32),
                )
                collector.append(frame)

                if done:
                    now = time.time()
                    if now - self.last_report_monitor_time >= 60 and self.monitor:
                        monitor_data = {
                            "episode_reward": round(total_reward, 4),
                            "episode_steps": step,
                            "episode_cnt": self.episode_cnt,

                            # reward components
                            "comp_treasure_score": round(
                                reward_component_sums.get("treasure_score_reward", 0.0), 4
                            ),
                            "comp_danger_penalty": round(
                                reward_component_sums.get("danger_penalty", 0.0), 4
                            ),
                            "comp_repeat_penalty": round(
                                reward_component_sums.get("repeat_penalty", 0.0), 4
                            ),
                            "comp_buff_reward": round(
                                reward_component_sums.get("buff_reward", 0.0), 4
                            ),
                            "comp_buff_collect": round(
                                reward_component_sums.get("buff_collect_reward", 0.0), 4
                            ),
                       
                            # local map debug: whether objects appeared in this episode
                            "debug_has_monster_max": round(
                                reward_debug_max.get("debug_curr_has_monster", 0.0), 4
                            ),

                            # latest progress debug
                            "debug_monster_progress_last": round(
                                reward_debug_latest.get("debug_monster_progress", 0.0), 4
                            ),

                            "monster_speed_factor_last": round(
                                reward_debug_latest.get("monster_speed_factor", 1.0), 4
                            ),
                            "debug_monster_progress_abs_sum": round(
                                reward_debug_abs_sum.get("debug_monster_progress", 0.0), 4
                            ),
                            "comp_survival": round(
                                reward_component_sums.get("survival_reward", 0.0), 4
                            ),  
                            "comp_move_reward": round(
                                reward_component_sums.get("move_reward", 0.0), 4
                            ),
                            "comp_flash_escape": round(
                                reward_component_sums.get("flash_escape_reward", 0.0), 4
                            ),
                            "comp_flash_toward_monster": round(
                                reward_component_sums.get("flash_toward_monster_penalty", 0.0), 4
                            ),
                            "comp_flash_waste": round(
                                reward_component_sums.get("flash_waste_penalty", 0.0), 4
                            ),
                            "debug_is_flash_action_max": round(
                                reward_debug_max.get("debug_is_flash_action", 0.0), 4
                            ),
                            "debug_in_flash_danger_max": round(
                                reward_debug_max.get("debug_in_flash_danger", 0.0), 4
                            ),
                            "comp_global_monster_field": round(
                                reward_component_sums.get("global_monster_field_reward", 0.0), 4
                            ),
                            "comp_global_treasure_field": round(
                                reward_component_sums.get("global_treasure_field_reward", 0.0), 4
                            ),
                            "comp_global_buff_field": round(
                                reward_component_sums.get("global_buff_field_reward", 0.0), 4
                            ),

                            "debug_monster_heat_progress_last": round(
                                reward_debug_latest.get("debug_monster_heat_progress", 0.0), 4
                            ),
                            "debug_treasure_heat_progress_last": round(
                                reward_debug_latest.get("debug_treasure_heat_progress", 0.0), 4
                            ),
                            "debug_buff_heat_progress_last": round(
                                reward_debug_latest.get("debug_buff_heat_progress", 0.0), 4
                            ),

                            "debug_monster_heat_progress_abs_sum": round(
                                reward_debug_abs_sum.get("debug_monster_heat_progress", 0.0), 4
                            ),
                            "debug_treasure_heat_progress_abs_sum": round(
                                reward_debug_abs_sum.get("debug_treasure_heat_progress", 0.0), 4
                            ),
                            "debug_buff_heat_progress_abs_sum": round(
                                reward_debug_abs_sum.get("debug_buff_heat_progress", 0.0), 4
                            ),
                            "comp_first_seen_treasure": round(
                                reward_component_sums.get("first_seen_treasure_reward", 0.0), 4
                            ),
                            "debug_new_seen_treasure_count": round(
                                reward_component_sums.get("debug_new_seen_treasure_count", 0.0), 4
                            ),

                            "comp_anti_stuck": round(
                                reward_component_sums.get("anti_stuck_penalty", 0.0), 4
                            ),
                            "debug_anti_stuck_dist_10_last": round(
                                reward_debug_latest.get("debug_anti_stuck_dist_10", 5.0), 4
                            ),

                        }

                        self.monitor.put_data({os.getpid(): monitor_data})
                        self.last_report_monitor_time = now

                    if collector:
                        yield collector
                    break

                obs = _obs
                obs_data = next_obs_data
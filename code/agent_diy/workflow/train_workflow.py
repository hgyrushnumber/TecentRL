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
from agent_diy.feature.definition import SampleData, sample_process
from tools.metrics_utils import get_training_metrics
from tools.train_env_conf_validate import read_usr_conf
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
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
        """Run a single episode and yield collected samples."""
        while True:
            now = time.time()
            if now - self.last_get_training_metrics_time >= 60:
                training_metrics = get_training_metrics()
                self.last_get_training_metrics_time = now
                if training_metrics is not None:
                    self.logger.info(f"training_metrics is {training_metrics}")

            env_obs = self.env.reset(self.usr_conf)

            if handle_disaster_recovery(env_obs, self.logger):
                continue

            self.agent.reset(env_obs)

            obs_data, remain_info = self.agent.observation_process(env_obs)
            if obs_data is None:
                self.logger.error("observation_process returned None on reset")
                continue

            collector = []
            self.episode_cnt += 1
            done = False
            step = 0
            total_reward = 0.0
            reward_component_sums = {}

            self.logger.info(f"Episode {self.episode_cnt} start")

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
                    env_reward, env_obs = self.env.step(act)
                except Exception as e:
                    self.logger.error(f"env.step() Exception {e}")
                    break

                if handle_disaster_recovery(env_obs, self.logger):
                    break

                try:
                    terminated = bool(env_obs["terminated"])
                    truncated = bool(env_obs["truncated"])
                except Exception as e:
                    self.logger.error(f"env_obs parse Exception {e}, env_obs={env_obs}")
                    break

                step += 1
                done = terminated or truncated

                try:
                    next_obs_data, next_remain_info = self.agent.observation_process(env_obs)
                except Exception as e:
                    self.logger.error(f"observation_process(next) Exception {e}")
                    break

                if next_obs_data is None:
                    self.logger.error("next observation_process returned None")
                    break

                reward = np.array(next_remain_info.get("reward", [0.0]), dtype=np.float32)
                total_reward += float(reward[0])
                reward_components = next_remain_info.get("reward_components", {})
                for k, v in reward_components.items():
                    reward_component_sums[k] = reward_component_sums.get(k, 0.0) + float(v)

                final_reward = np.zeros(1, dtype=np.float32)
                if done:
                    env_info = env_obs.get("observation", {}).get("env_info", {})
                    total_score = env_info.get("total_score", 0)

                    if terminated:
                        result_str = "DEAD"
                    elif truncated:
                        result_str = "TIMEOUT_DONE"
                    else:
                        result_str = "ABNORMAL"

                    self.logger.info(
                        f"[GAMEOVER] episode:{self.episode_cnt} steps:{step} "
                        f"result:{result_str} sim_score:{total_score:.1f} "
                        f"total_reward:{total_reward:.3f}"
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
                            "reward": round(total_reward + float(final_reward[0]), 4),
                            "episode_steps": step,
                            "episode_cnt": self.episode_cnt,
                            "final_reward": round(float(final_reward[0]), 4),
                            "comp_survive": round(reward_component_sums.get("survive_reward", 0.0), 4),
                            "comp_treasure_score": round(reward_component_sums.get("treasure_score_reward", 0.0), 4),
                            "comp_treasure_approach": round(
                                reward_component_sums.get("treasure_approach_reward", 0.0), 4
                            ),
                            "comp_danger_penalty": round(reward_component_sums.get("danger_penalty", 0.0), 4),
                            "comp_dist_shaping": round(reward_component_sums.get("dist_shaping", 0.0), 4),
                            "comp_repeat_penalty": round(reward_component_sums.get("repeat_explore_penalty", 0.0), 4),
                        }
                        self.monitor.put_data({os.getpid(): monitor_data})
                        self.last_report_monitor_time = now

                    if collector:
                        collector = sample_process(collector)
                        yield collector
                    break

                obs_data = next_obs_data
                remain_info = next_remain_info

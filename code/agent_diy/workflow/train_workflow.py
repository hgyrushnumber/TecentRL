#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Training workflow for Gorge Chase SAC — distributed-compatible.
峡谷追猎 SAC 训练工作流（兼容分布式架构）。

当前语义（路线B）：
  - Actor 侧按“整局”采样
  - episode 结束后 yield 一整局 SampleData
  - Learner 侧 receive 的 list_sample_data 即“一整局样本列表”
  - 训练奖励以 preprocessor.py 返回的 shaping reward 为准
  - total_score 仅用于评估、日志和监控，不直接参与训练
"""

import os
import time

import numpy as np
from agent_diy.feature.definition import SampleData, sample_process
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
            agent.send_sample_data(g_data)
            g_data.clear()

            now = time.time()
            if now - last_save_model_time >= 600:
                agent.save_model(id="latest")
                if logger:
                    logger.info("[workflow] saved latest checkpoint")
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
        """Collect one whole episode and yield SampleData list."""
        while True:
            now = time.time()
            if now - self.last_get_training_metrics_time >= 60:
                training_metrics = get_training_metrics()
                self.last_get_training_metrics_time = now
                if training_metrics is not None and self.logger:
                    self.logger.info(f"training_metrics is {training_metrics}")

            # 重置环境
            env_obs = self.env.reset(self.usr_conf)
            if handle_disaster_recovery(env_obs, self.logger):
                continue

            # 重置 Agent，并拉取最新模型
            self.agent.reset(env_obs)
            self.agent.load_model(id="latest")

            obs_data, remain_info = self.agent.observation_process(env_obs)

            collector = []
            self.episode_cnt += 1
            done = False
            step = 0
            total_reward = 0.0
            treasure_count = 0
            total_score = 0.0

            if self.logger:
                self.logger.info(f"Episode {self.episode_cnt} start")

            while not done:
                # Actor 推理动作
                act_data = self.agent.predict(list_obs_data=[obs_data])[0]
                act = self.agent.action_process(act_data)

                # 环境交互
                env_reward, env_obs = self.env.step(act)
                if handle_disaster_recovery(env_obs, self.logger):
                    break

                terminated = env_obs["terminated"]
                truncated = env_obs["truncated"]
                step += 1
                done = terminated or truncated

                # 下一帧观测（SAC 需要 next_obs）
                next_obs_data, next_remain_info = self.agent.observation_process(env_obs)

                # 路线B：训练奖励直接使用 preprocessor 返回的 shaping reward
                shaping_reward = float(
                    np.array(next_remain_info.get("reward", [0.0]), dtype=np.float32)[0]
                )
                reward = np.array([shaping_reward], dtype=np.float32)
                total_reward += float(reward[0])

                # 环境真实得分仅用于评估/日志
                env_info = env_obs["observation"]["env_info"]
                total_score = float(env_info.get("total_score", 0.0))
                treasure_count = int(env_info.get("treasure_count", 0))

                if done:
                    result_str = "FAIL" if terminated else "WIN"

                    if self.logger:
                        self.logger.info(
                            f"[GAMEOVER] episode:{self.episode_cnt} steps:{step} "
                            f"result:{result_str} sim_score:{total_score:.1f} "
                            f"treasure:{treasure_count} "
                            f"total_reward:{total_reward:.3f}"
                        )

                frame = SampleData(
                    obs=np.array(obs_data.feature, dtype=np.float32),
                    legal_action=np.array(obs_data.legal_action, dtype=np.float32),
                    act=np.array([act_data.action[0]], dtype=np.float32),
                    reward=reward,
                    done=np.array([float(done)], dtype=np.float32),
                    next_obs=np.array(next_obs_data.feature, dtype=np.float32),
                    next_legal_action=np.array(next_obs_data.legal_action, dtype=np.float32),

                    # 兼容字段（SAC 不使用）
                    reward_sum=np.zeros(1, dtype=np.float32),
                    value=np.zeros(1, dtype=np.float32),
                    next_value=np.zeros(1, dtype=np.float32),
                    advantage=np.zeros(1, dtype=np.float32),
                    prob=np.array(act_data.prob, dtype=np.float32),
                )
                collector.append(frame)

                if done:
                    now = time.time()
                    if now - self.last_report_monitor_time >= 60 and self.monitor:
                        monitor_data = {
                            "reward": round(total_reward, 4),         # shaping累计回报
                            "sim_score": round(total_score, 4),       # 环境真实得分
                            "episode_steps": step,
                            "episode_cnt": self.episode_cnt,
                            "finish_step": step,
                            "treasure": treasure_count,
                        }
                        self.monitor.put_data({os.getpid(): monitor_data})
                        self.last_report_monitor_time = now

                    collector = sample_process(collector)
                    yield collector
                    break

                obs_data = next_obs_data
                remain_info = next_remain_info
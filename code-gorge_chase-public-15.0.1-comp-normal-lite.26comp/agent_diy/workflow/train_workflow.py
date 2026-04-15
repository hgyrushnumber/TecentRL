#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Training workflow for Gorge Chase DIY Agent (Enhanced PPO).
峡谷追猎 DIY Agent 训练工作流（增强版 PPO）。

与 agent_ppo 工作流对齐，额外改进：
  - 终局奖励更细粒度（按 steps/max_step 比例奖励存活效率）
  - 支持每 30 分钟自动保存
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
        """Run a single episode and yield collected samples.

        执行单局对局并 yield 训练样本。
        """
        while True:
            now = time.time()
            if now - self.last_get_training_metrics_time >= 60:
                training_metrics = get_training_metrics()
                self.last_get_training_metrics_time = now
                if training_metrics is not None:
                    self.logger.info(f"training_metrics is {training_metrics}")

            # 重置环境
            env_obs = self.env.reset(self.usr_conf)

            # 容灾处理
            if handle_disaster_recovery(env_obs, self.logger):
                continue

            # 重置 Agent 并加载最新模型
            self.agent.reset(env_obs)
            self.agent.load_model(id="latest")

            # 初始观测
            obs_data, remain_info = self.agent.observation_process(env_obs)

            collector = []
            self.episode_cnt += 1
            done = False
            step = 0
            total_reward = 0.0

            self.logger.info(f"Episode {self.episode_cnt} start")

            while not done:
                # Agent 推理（随机采样）
                act_data = self.agent.predict(list_obs_data=[obs_data])[0]
                act = self.agent.action_process(act_data)

                # 与环境交互
                env_reward, env_obs = self.env.step(act)

                # 容灾处理
                if handle_disaster_recovery(env_obs, self.logger):
                    break

                terminated = env_obs["terminated"]
                truncated = env_obs["truncated"]
                step += 1
                done = terminated or truncated

                # 处理下一步观测
                _obs_data, _remain_info = self.agent.observation_process(env_obs)

                # 即时奖励
                reward = np.array(_remain_info.get("reward", [0.0]), dtype=np.float32)
                total_reward += float(reward[0])

                # 终局奖励
                final_reward = np.zeros(1, dtype=np.float32)
                if done:
                    env_info = env_obs["observation"]["env_info"]
                    total_score = env_info.get("total_score", 0)
                    max_step = env_info.get("max_step", 1000)

                    if terminated:
                        # 被抓：惩罚，但考虑已存活步数（存活越久惩罚越轻）
                        survival_ratio = float(step) / max(max_step, 1)
                        final_reward[0] = -10.0 + 5.0 * survival_ratio
                        result_str = "FAIL"
                    else:
                        # 成功存活：奖励
                        final_reward[0] = 10.0
                        result_str = "WIN"

                    self.logger.info(
                        f"[GAMEOVER] episode:{self.episode_cnt} steps:{step} "
                        f"result:{result_str} sim_score:{total_score:.1f} "
                        f"total_reward:{total_reward:.3f}"
                    )

                # 构造样本帧
                frame = SampleData(
                    obs=np.array(obs_data.feature, dtype=np.float32),
                    legal_action=np.array(obs_data.legal_action, dtype=np.float32),
                    act=np.array([act_data.action[0]], dtype=np.float32),
                    reward=reward,
                    done=np.array([float(done)], dtype=np.float32),
                    reward_sum=np.zeros(1, dtype=np.float32),
                    value=np.array(act_data.value, dtype=np.float32).flatten()[:1],
                    next_value=np.zeros(1, dtype=np.float32),
                    advantage=np.zeros(1, dtype=np.float32),
                    prob=np.array(act_data.prob, dtype=np.float32),
                )
                collector.append(frame)

                if done:
                    if collector:
                        collector[-1].reward = collector[-1].reward + final_reward

                    # 监控上报
                    now = time.time()
                    if now - self.last_report_monitor_time >= 60 and self.monitor:
                        monitor_data = {
                            "reward": round(total_reward + float(final_reward[0]), 4),
                            "episode_steps": step,
                            "episode_cnt": self.episode_cnt,
                        }
                        self.monitor.put_data({os.getpid(): monitor_data})
                        self.last_report_monitor_time = now

                    if collector:
                        collector = sample_process(collector)
                        yield collector
                    break

                # 状态更新
                obs_data = _obs_data
                remain_info = _remain_info
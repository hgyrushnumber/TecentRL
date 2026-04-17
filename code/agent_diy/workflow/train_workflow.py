#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Training workflow for Gorge Chase SAC — distributed-compatible.
峡谷追猎 SAC 训练工作流（兼容分布式架构）。

与原 PPO workflow 的唯一差异：
  SampleData 额外记录 next_obs / next_legal_action 字段，
  其余流程（yield collector → send_sample_data → learn）完全不变。
  ReplayBuffer 由 Algorithm（Learner 侧）内置管理，workflow 无感知。
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
        # yield 一局的 collector，框架通过 send_sample_data 发给 Learner
        for g_data in episode_runner.run_episodes():
            agent.send_sample_data(g_data)
            g_data.clear()

            now = time.time()
            if now - last_save_model_time >= 300:   # 每5分钟保存一次，Actor更快拿到最新模型
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
        """Collect one episode and yield SampleData list.

        采集一局数据并 yield，由框架发送给 Learner 侧训练。
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
            if handle_disaster_recovery(env_obs, self.logger):
                continue

            # 重置 Agent，拉取最新模型
            self.agent.reset(env_obs)
            self.agent.load_model(id="latest")

            obs_data, remain_info = self.agent.observation_process(env_obs)

            collector = []
            self.episode_cnt += 1
            done = False
            step = 0
            total_reward = 0.0

            self.logger.info(f"Episode {self.episode_cnt} start")

            while not done:
                # Actor 侧：推理动作
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
                _obs_data, _remain_info = self.agent.observation_process(env_obs)

                reward = np.array(_remain_info.get("reward", [0.0]), dtype=np.float32)
                total_reward += float(reward[0])

                # 终局奖励
                final_reward = np.zeros(1, dtype=np.float32)
                if done:
                    env_info = env_obs["observation"]["env_info"]
                    total_score = env_info.get("total_score", 0)
                    treasure_count = env_info.get("treasure_count", 0)

                    if terminated:
                        final_reward[0] = -5.0   # 被抓：强惩罚，明确告知模型死亡代价
                        result_str = "FAIL"
                    else:
                        final_reward[0] = 5.0    # 生存至终：强奖励，引导模型优先存活
                        result_str = "WIN"

                    self.logger.info(
                        f"[GAMEOVER] episode:{self.episode_cnt} steps:{step} "
                        f"result:{result_str} sim_score:{total_score:.1f} "
                        f"treasure:{treasure_count} "
                        f"total_reward:{total_reward:.3f}"
                    )

                # 构造 SampleData（含 next_obs，SAC 所需）
                frame = SampleData(
                    obs=np.array(obs_data.feature, dtype=np.float32),
                    legal_action=np.array(obs_data.legal_action, dtype=np.float32),
                    act=np.array([act_data.action[0]], dtype=np.float32),
                    reward=reward,
                    done=np.array([float(done)], dtype=np.float32),
                    next_obs=np.array(_obs_data.feature, dtype=np.float32),
                    next_legal_action=np.array(_obs_data.legal_action, dtype=np.float32),
                    # 以下字段 SAC 不使用，置0保持兼容
                    reward_sum=np.zeros(1, dtype=np.float32),
                    value=np.zeros(1, dtype=np.float32),
                    next_value=np.zeros(1, dtype=np.float32),
                    advantage=np.zeros(1, dtype=np.float32),
                    prob=np.array(act_data.prob, dtype=np.float32),
                )
                collector.append(frame)

                if done:
                    # 终局奖励叠加到最后一帧
                    collector[-1].reward = collector[-1].reward + final_reward

                    now = time.time()
                    if now - self.last_report_monitor_time >= 60 and self.monitor:
                        monitor_data = {
                            "reward": round(total_reward + float(final_reward[0]), 4),
                            "episode_steps": step,
                            "episode_cnt": self.episode_cnt,
                            "finish_step": step,
                            "treasure": treasure_count,
                        }
                        self.monitor.put_data({os.getpid(): monitor_data})
                        self.last_report_monitor_time = now

                    # yield 一局数据给框架（send_sample_data → Learner → learn()）
                    collector = sample_process(collector)
                    yield collector
                    break

                obs_data = _obs_data
                remain_info = _remain_info
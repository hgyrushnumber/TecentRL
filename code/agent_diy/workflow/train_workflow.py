#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Training workflow for Gorge Chase SAC.
峡谷追猎 SAC 训练工作流。

与PPO的关键区别：
  - 使用 ReplayBuffer 存储所有历史转换（off-policy）
  - 每局结束后从 buffer 中随机采样 batch 训练（而非用当局数据）
  - 每局训练次数 = 当局步数（环境步 : 训练步 ≈ 1:1）
"""

import os
import time

import numpy as np
from agent_diy.feature.definition import SampleData, ReplayBuffer, sample_process
from tools.metrics_utils import get_training_metrics
from tools.train_env_conf_validate import read_usr_conf
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery
from agent_diy.conf.conf import Config


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    last_save_model_time = time.time()
    env = envs[0]
    agent = agents[0]

    usr_conf = read_usr_conf("agent_diy/conf/train_env_conf.toml", logger)
    if usr_conf is None:
        logger.error("usr_conf is None, please check agent_diy/conf/train_env_conf.toml")
        return

    # SAC ReplayBuffer（全局，跨局共享）
    replay_buffer = ReplayBuffer(Config.REPLAY_BUFFER_SIZE)

    episode_runner = EpisodeRunner(
        env=env,
        agent=agent,
        usr_conf=usr_conf,
        replay_buffer=replay_buffer,
        logger=logger,
        monitor=monitor,
    )

    while True:
        episode_runner.run_one_episode()

        now = time.time()
        if now - last_save_model_time >= 1800:
            agent.save_model()
            last_save_model_time = now


class EpisodeRunner:
    def __init__(self, env, agent, usr_conf, replay_buffer, logger, monitor):
        self.env = env
        self.agent = agent
        self.usr_conf = usr_conf
        self.replay_buffer = replay_buffer
        self.logger = logger
        self.monitor = monitor
        self.episode_cnt = 0
        self.total_steps = 0
        self.last_report_monitor_time = 0
        self.last_get_training_metrics_time = 0

    def run_one_episode(self):
        """Run one episode: collect transitions → train from replay buffer.

        执行一局：收集转换存入回放池 → 从回放池采样训练。
        """
        # 定期获取训练指标
        now = time.time()
        if now - self.last_get_training_metrics_time >= 60:
            training_metrics = get_training_metrics()
            self.last_get_training_metrics_time = now
            if training_metrics is not None:
                self.logger.info(f"training_metrics is {training_metrics}")

        # 重置环境
        env_obs = self.env.reset(self.usr_conf)
        if handle_disaster_recovery(env_obs, self.logger):
            return

        self.agent.reset(env_obs)
        self.agent.load_model(id="latest")

        obs_data, _ = self.agent.observation_process(env_obs)
        self.episode_cnt += 1
        done = False
        step = 0
        total_reward = 0.0
        train_count = 0

        self.logger.info(f"Episode {self.episode_cnt} start | buffer={len(self.replay_buffer)}")

        while not done:
            # 推理动作
            act_data = self.agent.predict(list_obs_data=[obs_data])[0]
            act = self.agent.action_process(act_data)

            # 与环境交互
            env_reward, env_obs = self.env.step(act)
            if handle_disaster_recovery(env_obs, self.logger):
                break

            terminated = env_obs["terminated"]
            truncated = env_obs["truncated"]
            step += 1
            done = terminated or truncated
            self.total_steps += 1

            # 处理下一帧观测
            next_obs_data, next_remain_info = self.agent.observation_process(env_obs)
            reward = np.array(next_remain_info.get("reward", [0.0]), dtype=np.float32)
            total_reward += float(reward[0])

            # 终局奖励
            final_reward = np.zeros(1, dtype=np.float32)
            if done:
                env_info = env_obs["observation"]["env_info"]
                total_score = env_info.get("total_score", 0)
                treasure_count = env_info.get("treasure_count", 0)
                if terminated:
                    final_reward[0] = -10.0
                    result_str = "FAIL"
                else:
                    final_reward[0] = 10.0
                    result_str = "WIN"
                reward = reward + final_reward

                self.logger.info(
                    f"[GAMEOVER] episode:{self.episode_cnt} steps:{step} "
                    f"result:{result_str} sim_score:{total_score:.1f} "
                    f"treasure:{treasure_count} "
                    f"total_reward:{total_reward:.3f} "
                    f"buffer:{len(self.replay_buffer)} "
                    f"train_steps:{train_count}"
                )

            # 构造 SAC 转换帧（包含 next_obs 和 next_legal_action）
            transition = SampleData(
                obs=np.array(obs_data.feature, dtype=np.float32),
                legal_action=np.array(obs_data.legal_action, dtype=np.float32),
                act=np.array([act_data.action[0]], dtype=np.float32),
                reward=reward,
                next_obs=np.array(next_obs_data.feature, dtype=np.float32),
                next_legal_action=np.array(next_obs_data.legal_action, dtype=np.float32),
                done=np.array([float(done)], dtype=np.float32),
                # 以下字段 SAC 不使用
                reward_sum=np.zeros(1, dtype=np.float32),
                value=np.zeros(1, dtype=np.float32),
                next_value=np.zeros(1, dtype=np.float32),
                advantage=np.zeros(1, dtype=np.float32),
                prob=np.array(act_data.prob, dtype=np.float32),
            )
            self.replay_buffer.push(transition)

            # ── SAC 每步训练一次（buffer 足够时）────────────────────
            if self.replay_buffer.ready:
                batch = self.replay_buffer.sample(Config.BATCH_SIZE)
                self.agent.learn(batch)
                train_count += 1

            # 监控上报（每60秒）
            now = time.time()
            if done and now - self.last_report_monitor_time >= 60 and self.monitor:
                monitor_data = {
                    "reward": round(float(total_reward), 4),
                    "episode_steps": step,
                    "episode_cnt": self.episode_cnt,
                    "finish_step": step,
                    "treasure": treasure_count if done else 0,
                    "buffer_size": len(self.replay_buffer),
                    "train_steps": train_count,
                }
                self.monitor.put_data({os.getpid(): monitor_data})
                self.last_report_monitor_time = now

            obs_data = next_obs_data
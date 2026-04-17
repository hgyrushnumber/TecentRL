#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Agent class for Gorge Chase SAC — distributed-compatible.
峡谷追猎 SAC Agent（兼容分布式架构）。

分布式角色说明：
  Actor  进程：调用 predict() 采样动作，observation_process() 提取特征，
               workflow 收集样本后调用 send_sample_data() 发送给 Learner
  Learner进程：框架调用 learn(list_sample_data) 触发 SAC 更新
               完成后调用 save_model() 写入共享存储供 Actor 拉取
"""

import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import numpy as np
from kaiwudrl.interface.agent import BaseAgent

from agent_diy.algorithm.algorithm import Algorithm
from agent_diy.conf.conf import Config
from agent_diy.feature.definition import ActData, ObsData
from agent_diy.feature.preprocessor import Preprocessor
from agent_diy.model.model import Actor, Critic


class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        torch.manual_seed(0)
        self.device = device

        input_dim = Config.DIM_OF_OBSERVATION
        hidden_dim = 256
        mid_dim = 128
        action_num = Config.ACTION_NUM

        # Actor 网络（Actor/Learner 进程均持有，Actor 侧只做推理）
        self.model = Actor(input_dim, hidden_dim, mid_dim, action_num).to(device)
        # Critic 网络（Learner 侧参与训练，Actor 侧不使用）
        self.critic = Critic(input_dim, hidden_dim, mid_dim, action_num).to(device)

        # Algorithm 持有双Q网络、目标网络、优化器、ReplayBuffer（均在 Learner 侧生效）
        self.algorithm = Algorithm(self.model, self.critic, device, logger, monitor)

        self.preprocessor = Preprocessor()
        self.last_action = -1
        self.logger = logger
        self.monitor = monitor
        super().__init__(agent_type, device, logger, monitor)

    # ── 每局重置（Actor 侧）──────────────────────────────────────────
    def reset(self, env_obs=None):
        self.preprocessor.reset()
        self.last_action = -1

    # ── 观测处理（Actor 侧）──────────────────────────────────────────
    def observation_process(self, env_obs, preprocessor=None, extra_info=None):
        feature, legal_action, reward = self.preprocessor.feature_process(
            env_obs, self.last_action
        )
        obs_data = ObsData(feature=list(feature), legal_action=legal_action)
        return obs_data, {"reward": reward}

    # ── 推理（Actor 侧，随机采样用于探索）───────────────────────────
    def predict(self, list_obs_data):
        feature = list_obs_data[0].feature
        legal_action = list_obs_data[0].legal_action

        self.model.set_eval_mode()
        obs_t = torch.tensor(np.array([feature]), dtype=torch.float32).to(self.device)
        la_t = torch.tensor(np.array([legal_action]), dtype=torch.float32).to(self.device)

        with torch.no_grad():
            probs = self.model(obs_t, la_t)[0].cpu().numpy()  # (A,)

        # 随机采样（探索）
        probs = np.clip(probs, 1e-9, None)
        probs /= probs.sum()
        action = int(np.random.choice(len(probs), p=probs))
        d_action = int(np.argmax(probs))

        return [ActData(action=[action], d_action=[d_action], prob=list(probs), value=[0.0])]

    # ── 评估推理（贪心）─────────────────────────────────────────────
    def exploit(self, env_obs):
        obs_data, _ = self.observation_process(env_obs)
        act_data = self.predict([obs_data])
        return self.action_process(act_data[0], is_stochastic=False)

    # ── 训练入口（Learner 侧，由框架调用）────────────────────────────
    def learn(self, list_sample_data):
        return self.algorithm.learn(list_sample_data)

    # ── 动作解包 ─────────────────────────────────────────────────────
    def action_process(self, act_data, is_stochastic=True):
        action = act_data.action if is_stochastic else act_data.d_action
        self.last_action = int(action[0])
        return int(action[0])

    # ── 模型存储（标准接口，单文件存 Actor，兼容框架）────────────────
    def save_model(self, path=None, id="1"):
        """Save actor weights + full training state (optimizer, log_alpha, train_step).

        存储 Actor/Critic 权重及完整训练状态，支持断点续训。
        """
        # 主文件：actor，供 Actor 进程 load_model 拉取参数
        model_file = f"{path}/model.ckpt-{id}.pkl"
        torch.save(
            {k: v.clone().cpu() for k, v in self.model.state_dict().items()},
            model_file,
        )
        # Critic 权重
        critic_file = f"{path}/critic.ckpt-{id}.pkl"
        torch.save(
            {k: v.clone().cpu() for k, v in self.critic.state_dict().items()},
            critic_file,
        )
        # 完整训练状态（优化器 + log_alpha + train_step，断点续训必需）
        train_state_file = f"{path}/train_state.ckpt-{id}.pkl"
        torch.save(
            {
                "actor_opt": self.algorithm.actor_optimizer.state_dict(),
                "critic_opt": self.algorithm.critic_optimizer.state_dict(),
                "alpha_opt": self.algorithm.alpha_optimizer.state_dict(),
                "log_alpha": self.algorithm.log_alpha.item(),
                "train_step": self.algorithm.train_step,
            },
            train_state_file,
        )
        if self.logger:
            self.logger.info(
                f"[SAC] save model {model_file} successfully "
                f"(train_step={self.algorithm.train_step})"
            )

    def load_model(self, path=None, id="1"):
        """Load actor weights (standard interface for Actor process).

        Actor 进程：只加载 Actor 权重用于推理。
        Learner 进程：同时加载完整训练状态（优化器/log_alpha/train_step）实现断点续训。
        文件不存在时静默跳过，从头开始训练。
        """
        model_file = f"{path}/model.ckpt-{id}.pkl"
        try:
            self.model.load_state_dict(
                torch.load(model_file, map_location=self.device)
            )
        except FileNotFoundError:
            if self.logger:
                self.logger.info(f"[SAC] no checkpoint at {model_file}, training from scratch")
            return
        except Exception as e:
            if self.logger:
                self.logger.warning(f"[SAC] load actor failed: {e}, training from scratch")
            return

        # 尝试加载 Critic + 完整训练状态（Learner 侧断点续训）
        critic_file = f"{path}/critic.ckpt-{id}.pkl"
        try:
            self.critic.load_state_dict(
                torch.load(critic_file, map_location=self.device)
            )
            self.algorithm.critic_target.load_state_dict(self.critic.state_dict())
        except (FileNotFoundError, Exception):
            pass

        # 恢复优化器状态（断点续训核心，Actor侧文件可能不存在则跳过）
        train_state_file = f"{path}/train_state.ckpt-{id}.pkl"
        try:
            state = torch.load(train_state_file, map_location=self.device)
            self.algorithm.actor_optimizer.load_state_dict(state["actor_opt"])
            self.algorithm.critic_optimizer.load_state_dict(state["critic_opt"])
            self.algorithm.alpha_optimizer.load_state_dict(state["alpha_opt"])
            with torch.no_grad():
                self.algorithm.log_alpha.fill_(state["log_alpha"])
            self.algorithm.alpha = self.algorithm.log_alpha.exp().clamp(1e-4, 1.0).item()
            self.algorithm.train_step = state.get("train_step", 0)
        except (FileNotFoundError, Exception):
            pass

        if self.logger:
            self.logger.info(
                f"[SAC] load model {model_file} successfully "
                f"(train_step={self.algorithm.train_step})"
            )
#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Agent class for Gorge Chase SAC.
峡谷追猎 SAC Agent 主类。
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

        # SAC 网络：Actor + 双Q Critic
        self.model = Actor(input_dim, hidden_dim, mid_dim, action_num).to(device)
        self.critic = Critic(input_dim, hidden_dim, mid_dim, action_num).to(device)

        # Algorithm 持有所有优化器和目标网络
        self.algorithm = Algorithm(self.model, self.critic, device, logger, monitor)

        self.preprocessor = Preprocessor()
        self.last_action = -1
        self.logger = logger
        self.monitor = monitor
        super().__init__(agent_type, device, logger, monitor)

    # ── 每局重置 ────────────────────────────────────────────────────────
    def reset(self, env_obs=None):
        self.preprocessor.reset()
        self.last_action = -1

    # ── 观测处理 ────────────────────────────────────────────────────────
    def observation_process(self, env_obs):
        feature, legal_action, reward = self.preprocessor.feature_process(
            env_obs, self.last_action
        )
        obs_data = ObsData(feature=list(feature), legal_action=legal_action)
        return obs_data, {"reward": reward}

    # ── 训练推理（随机采样）─────────────────────────────────────────────
    def predict(self, list_obs_data):
        feature = list_obs_data[0].feature
        legal_action = list_obs_data[0].legal_action

        self.model.set_eval_mode()
        obs_t = torch.tensor(np.array([feature]), dtype=torch.float32).to(self.device)
        la_t = torch.tensor(np.array([legal_action]), dtype=torch.float32).to(self.device)

        with torch.no_grad():
            probs = self.model(obs_t, la_t)[0].cpu().numpy()   # (A,)

        # 随机采样（探索）
        action = int(np.random.choice(len(probs), p=probs))
        d_action = int(np.argmax(probs))

        return [ActData(action=[action], d_action=[d_action], prob=list(probs), value=[0.0])]

    # ── 评估推理（贪心）────────────────────────────────────────────────
    def exploit(self, env_obs):
        obs_data, _ = self.observation_process(env_obs)
        act_data = self.predict([obs_data])
        return self.action_process(act_data[0], is_stochastic=False)

    # ── 训练（由 workflow 调用，传入从 ReplayBuffer 采样的 batch）─────
    def learn(self, list_sample_data):
        return self.algorithm.learn(list_sample_data)

    # ── 动作解包 ────────────────────────────────────────────────────────
    def action_process(self, act_data, is_stochastic=True):
        action = act_data.action if is_stochastic else act_data.d_action
        self.last_action = int(action[0])
        return int(action[0])

    # ── 模型存储（委托 Algorithm 处理）────────────────────────────────
    def save_model(self, path=None, id="1"):
        self.algorithm.save_model(path, id)

    def load_model(self, path=None, id="1"):
        self.algorithm.load_model(path, id)

    # ── send_sample_data: workflow 调用接口（SAC 不用，保留空实现）────
    def send_sample_data(self, data):
        pass
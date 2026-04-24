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
from agent_diy.model.model import Model


class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        torch.manual_seed(0)
        np.random.seed(0)

        self.device = device
        self.model = Model(device).to(self.device)
        self.algorithm = Algorithm(self.model, None, self.device, logger, monitor)
        self.preprocessor = Preprocessor()
        self.last_action = -1
        self.logger = logger
        self.monitor = monitor
        super().__init__(agent_type, device, logger, monitor)

    def reset(self, env_obs=None):
        self.preprocessor.reset()
        self.last_action = -1

    def observation_process(self, env_obs):
        feature, legal_action = self.preprocessor.feature_process(env_obs, self.last_action)

        obs_data = ObsData(
            feature=list(feature),
            legal_action=legal_action,
        )

        remain_info = {
            "env_info": env_obs.get("observation", {}).get("env_info", {}),
            "raw_obs": env_obs,
        }

        return obs_data, remain_info

    def predict(self, list_obs_data):
        feature = list_obs_data[0].feature
        legal_action = list_obs_data[0].legal_action

        probs, q1, q2 = self._run_model(feature, legal_action)

        legal_mask = np.array(legal_action, dtype=np.float32)
        min_q = np.minimum(q1, q2)
        masked_q = np.where(legal_mask > 0, min_q, -1e9)

        action = int(np.random.choice(len(probs), p=probs))
        d_action = int(np.argmax(masked_q))

        return [
            ActData(
                action=[action],
                d_action=[d_action],
            )
        ]

    def exploit(self, env_obs):
        obs_data, _ = self.observation_process(env_obs)
        act_data = self.predict([obs_data])
        return self.action_process(act_data[0], is_stochastic=False)

    def learn(self, list_sample_data):
        return self.algorithm.learn(list_sample_data)

    def save_model(self, path=None, id="1"):
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        state_dict_cpu = {k: v.clone().cpu() for k, v in self.model.state_dict().items()}
        torch.save(state_dict_cpu, model_file_path)
        if self.logger:
            self.logger.info(f"save model {model_file_path} successfully")

    def load_model(self, path=None, id="1"):
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        self.model.load_state_dict(torch.load(model_file_path, map_location=self.device))
        if self.logger:
            self.logger.info(f"load model {model_file_path} successfully")

    def action_process(self, act_data, is_stochastic=True):
        action = act_data.action if is_stochastic else act_data.d_action
        self.last_action = int(action[0])
        return int(action[0])

    def _run_model(self, feature, legal_action):
        self.model.set_eval_mode()
        obs_tensor = torch.tensor(np.array([feature]), dtype=torch.float32).to(self.device)
        legal_tensor = torch.tensor(np.array([legal_action]), dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits, q1, q2 = self.model(obs_tensor, inference=True)
            probs = self.algorithm._masked_softmax(logits, legal_tensor)[0].cpu().numpy()
            q1 = q1[0].cpu().numpy()
            q2 = q2[0].cpu().numpy()

        return probs, q1, q2

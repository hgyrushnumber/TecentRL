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
  Actor 进程：
    - observation_process() 提取特征
    - predict() 采样动作
    - workflow 收集整局样本后调用 send_sample_data() 发给 Learner

  Learner 进程：
    - 框架调用 learn(list_sample_data) 触发 SAC 更新
    - 完成后调用 save_model() 写入共享存储供 Actor 拉取

路线B语义说明：
  - preprocessor.py 返回的 reward 是训练主奖励
  - workflow.py 直接使用该 reward 作为 Learner 训练目标
  - total_score 仅用于评估/日志/监控，不直接作为训练 reward
"""

import os
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
        np.random.seed(0)
        self.device = device

        input_dim = Config.DIM_OF_OBSERVATION
        hidden_dim = 512
        mid_dim = 256
        action_num = Config.ACTION_NUM

        # Actor 网络（Actor/Learner 进程均持有，Actor 侧只做推理）
        self.model = Actor(input_dim, hidden_dim, mid_dim, action_num).to(device)

        # Critic 网络（Learner 侧参与训练，Actor 侧不直接使用）
        self.critic = Critic(input_dim, hidden_dim, mid_dim, action_num).to(device)

        # Algorithm 持有双Q网络、目标网络、优化器、ReplayBuffer（Learner侧生效）
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
        """
        提取特征、合法动作以及训练奖励。

        路线B中：
          reward 由 preprocessor.py 定义，
          workflow.py 会直接使用这里返回的 reward 作为训练目标。
        """
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
            probs = self.model(obs_t, la_t)[0].cpu().numpy()  # [A]

        # ε-greedy 探索：防止早期策略锁死
        if np.random.random() < 0.3:
            legal_actions = np.where(np.array(legal_action) == 1)[0]
            if len(legal_actions) > 0:
                action = int(np.random.choice(legal_actions))
                d_action = int(np.argmax(probs))

                probs_out = np.zeros(Config.ACTION_NUM, dtype=np.float32)
                probs_out[action] = 1.0

                return [
                    ActData(
                        action=[action],
                        d_action=[d_action],
                        prob=list(probs_out),
                        value=[0.0],
                    )
                ]

        # 基于策略分布的温度采样，增加探索性
        temperature = 1.5
        logits = np.log(np.clip(probs, 1e-9, None))
        tempered_logits = logits / temperature
        tempered_probs = np.exp(tempered_logits)
        tempered_probs = tempered_probs / np.sum(tempered_probs)

        action = int(np.random.choice(len(tempered_probs), p=tempered_probs))
        d_action = int(np.argmax(probs))

        return [
            ActData(
                action=[action],
                d_action=[d_action],
                prob=list(tempered_probs),   # 记录真实采样分布，更一致
                value=[0.0],
            )
        ]

    # ── 评估推理（贪心）─────────────────────────────────────────────
    def exploit(self, env_obs):
        obs_data, _ = self.observation_process(env_obs)
        act_data = self.predict([obs_data])
        return self.action_process(act_data[0], is_stochastic=False)

    # ── 训练入口（Learner 侧，由框架调用）────────────────────────────
    def learn(self, list_sample_data):
        results = self.algorithm.learn(list_sample_data)
        if results is not None:
            if self.monitor:
                self.monitor.put_data({os.getpid(): results})
            if self.logger:
                self.logger.info(
                    f"[SAC] train_step:{results['train_step']} "
                    f"total_loss:{results['total_loss']} "
                    f"critic_loss:{results['critic_loss']} "
                    f"actor_loss:{results['actor_loss']} "
                    f"entropy:{results['entropy']:.3f} "
                    f"alpha:{results['alpha']:.4f}"
                )
        return results

    # ── 动作解包 ─────────────────────────────────────────────────────
    def action_process(self, act_data, is_stochastic=True):
        action = act_data.action if is_stochastic else act_data.d_action
        self.last_action = int(action[0])
        return int(action[0])

    # ── 模型存储（仅 Actor latest，用于 Actor 拉取推理）─────────────
    def save_model(self, path=None, id="1"):
        """
        Save actor weights only.

        注意：
          这里保存的是 Actor 侧“最新推理模型”，用于 workflow 中定期同步 latest。
          这不是 Learner 侧完整断点续训存储。
        """
        if path is not None:
            self._model_path = path

        model_file = f"{path}/model.ckpt-{id}.pkl"
        torch.save(
            {k: v.clone().cpu() for k, v in self.model.state_dict().items()},
            model_file,
        )

        if self.logger:
            self.logger.info(f"[SAC] save model {model_file} successfully")

    def load_model(self, path=None, id="1"):
        """
        Load actor weights only for inference.

        注意：
          这里只加载 Actor latest 权重用于推理，
          不恢复 Critic / Optimizer / alpha 等完整训练状态。
        """
        model_file = f"{path}/model.ckpt-{id}.pkl"
        try:
            self.model.load_state_dict(
                torch.load(model_file, map_location=self.device)
            )
        except FileNotFoundError:
            if self.logger:
                self.logger.info(f"[SAC] no checkpoint at {model_file}, skip")
            return
        except Exception as e:
            if self.logger:
                self.logger.warning(f"[SAC] load actor failed: {e}, skip")
            return

        if self.logger:
            self.logger.info(f"[SAC] load model {model_file} successfully")
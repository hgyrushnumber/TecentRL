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
        self.device = device

        input_dim = Config.DIM_OF_OBSERVATION
        hidden_dim = 512   # 增大隐藏层（原256），提升复杂空间感知能力
        mid_dim = 256      # 增大中间层（原128），匹配102维输入+49维地图特征
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
        # 断点续训已移除：训练成本低，每次从随机初始化开始更干净

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

        # ε-greedy 探索：训练早期10%随机探索，防止过早收敛
        if np.random.random() < 0.1:
            legal_actions = np.where(np.array(legal_action[:8]) == 1)[0]  # 只考虑移动动作
            if len(legal_actions) > 0:
                action = int(np.random.choice(legal_actions))
                d_action = int(np.argmax(probs))
                probs_out = np.zeros(10, dtype=np.float32)
                probs_out[action] = 1.0
                return [ActData(action=[action], d_action=[d_action], prob=list(probs_out), value=[0.0])]

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
        """Train and return loss dict (aligned with PPO interface).

        训练并返回损失字典，字段与PPO对齐：
          value_loss  → Critic MSE 损失（对应PPO的价值损失）
          policy_loss → Actor 策略损失（对应PPO的策略损失）
          entropy_loss→ 策略熵（对应PPO的熵损失）
          total_loss  → value_loss + policy_loss 聚合
        """
        results = self.algorithm.learn(list_sample_data)
        if results is not None:
            # 实时上报到 monitor（与PPO look相同字段）
            if self.monitor:
                self.monitor.put_data({os.getpid(): results})
            if self.logger:
                self.logger.info(
                    f"[SAC] train_step:{results['train_step']} "
                    f"total_loss:{results['total_loss']} "
                    f"value_loss:{results['value_loss']} "
                    f"policy_loss:{results['policy_loss']} "
                    f"entropy_loss:{results['entropy_loss']:.3f}"
                )
        return results

    # ── 动作解包 ─────────────────────────────────────────────────────
    def action_process(self, act_data, is_stochastic=True):
        action = act_data.action if is_stochastic else act_data.d_action
        self.last_action = int(action[0])
        return int(action[0])

    # ── 模型存储（只存 Actor 权重，供 Actor 侧拉取推理）─────────────
    def save_model(self, path=None, id="1"):
        """Save actor weights only.

        断点续训已移除，仅保存 Actor 权重供 Actor 进程 load_model 拉取。
        减少磁盘 IO，避免影响 Actor 采样效率。
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
        """Load actor weights only (Actor process inference).

        只加载 Actor 权重用于推理，不恢复优化器/Critic 等训练状态。
        文件不存在时静默跳过。
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
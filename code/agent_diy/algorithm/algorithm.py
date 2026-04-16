#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Discrete SAC algorithm for Gorge Chase — distributed-compatible.
峡谷追猎 离散动作 SAC 算法（兼容分布式架构）。

分布式架构适配：
  learn(list_sample_data) 由框架在 Learner 侧调用，接收 Actor 发来的一局数据：
    1. 将本局数据批量推入内置 ReplayBuffer
    2. 若 buffer 已达到 LEARNING_STARTS，执行 N 次 SAC 梯度更新
       N = len(list_sample_data)（与本局步数等量，保持 env_step:train_step ≈ 1:1）

损失：
  critic_loss = MSE(Q_i(s,a), r + γ*(1-d)*V(s'))
  actor_loss  = E_π[α*logπ(a|s) - min(Q1,Q2)(s,a)]
  alpha_loss  = α*(H[π] - H_target)
"""

import os
import time
import copy

import numpy as np
import torch
import torch.nn.functional as F

from agent_diy.conf.conf import Config
from agent_diy.feature.definition import ReplayBuffer


class Algorithm:
    def __init__(self, actor, critic, device=None, logger=None, monitor=None):
        self.device = device
        self.actor = actor
        self.critic = critic
        self.logger = logger
        self.monitor = monitor

        # 目标 Q 网络（软更新，不参与梯度）
        self.critic_target = copy.deepcopy(critic)
        for p in self.critic_target.parameters():
            p.requires_grad = False

        self.gamma = Config.GAMMA
        self.tau = Config.TAU
        self.grad_clip = Config.GRAD_CLIP_RANGE
        self.batch_size = Config.BATCH_SIZE

        # 优化器
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=Config.INIT_LEARNING_RATE_START
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=Config.INIT_LEARNING_RATE_START
        )

        # 自动熵调整 α
        self.target_entropy = Config.TARGET_ENTROPY_RATIO * np.log(Config.ACTION_NUM)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha = self.log_alpha.exp().item()
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=Config.ALPHA_LR)

        # ── Learner 侧内置 ReplayBuffer ──────────────────────────────
        self.replay_buffer = ReplayBuffer(Config.REPLAY_BUFFER_SIZE)

        self.train_step = 0
        self.last_report_monitor_time = 0

    # ── 框架调用入口（Learner 侧）────────────────────────────────────
    def learn(self, list_sample_data):
        """Receive one episode's transitions, push to buffer, then train.

        接收 Actor 发来的一局数据，推入 ReplayBuffer，再执行等量次数的 SAC 更新。
        符合分布式架构：框架在 Learner 侧调用此方法，Actor 侧不直接训练。
        """
        # 1. 将本局数据推入 ReplayBuffer
        self.replay_buffer.push_batch(list_sample_data)

        # 2. buffer 未达到预热量时跳过训练
        if len(self.replay_buffer) < Config.LEARNING_STARTS:
            if self.logger:
                self.logger.info(
                    f"[SAC] warming up buffer: {len(self.replay_buffer)}/{Config.LEARNING_STARTS}"
                )
            return

        # 3. 执行 N 次梯度更新（N = 本局步数，保持 1:1 比例）
        n_updates = len(list_sample_data)
        for _ in range(n_updates):
            batch = self.replay_buffer.sample(self.batch_size)
            self._update(batch)

    # ── SAC 单次梯度更新 ─────────────────────────────────────────────
    def _update(self, batch):
        obs = torch.stack([f.obs for f in batch]).to(self.device)
        legal = torch.stack([f.legal_action for f in batch]).to(self.device)
        act = torch.stack([f.act for f in batch]).to(self.device).long().view(-1)
        rew = torch.stack([f.reward for f in batch]).to(self.device).view(-1, 1)
        next_obs = torch.stack([f.next_obs for f in batch]).to(self.device)
        next_legal = torch.stack([f.next_legal_action for f in batch]).to(self.device)
        done = torch.stack([f.done for f in batch]).to(self.device).view(-1, 1)

        # ── Critic 更新 ───────────────────────────────────────────────
        with torch.no_grad():
            next_probs = self.actor(next_obs, next_legal)               # (B, A)
            next_log_p = torch.log(next_probs.clamp(1e-9))
            q1_t, q2_t = self.critic_target(next_obs)
            min_q_t = torch.min(q1_t, q2_t)
            # 软贝尔曼目标 V(s') = Σ_a π * (Q - α*logπ)
            v_next = (next_probs * (min_q_t - self.alpha * next_log_p)).sum(1, keepdim=True)
            target_q = rew + self.gamma * (1.0 - done) * v_next

        q1_all, q2_all = self.critic(obs)
        q1 = q1_all.gather(1, act.unsqueeze(1))
        q2 = q2_all.gather(1, act.unsqueeze(1))
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.critic_optimizer.step()

        # ── Actor 更新 ────────────────────────────────────────────────
        probs = self.actor(obs, legal)
        log_p = torch.log(probs.clamp(1e-9))
        with torch.no_grad():
            q1_pi, q2_pi = self.critic(obs)
            min_q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (probs * (self.alpha * log_p - min_q_pi)).sum(1).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.actor_optimizer.step()

        # ── α 更新 ────────────────────────────────────────────────────
        with torch.no_grad():
            entropy = -(probs * log_p).sum(1).mean()
        alpha_loss = self.log_alpha * (entropy - self.target_entropy).detach()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().clamp(1e-4, 1.0).item()

        # ── 软更新目标网络 ────────────────────────────────────────────
        for p_o, p_t in zip(self.critic.parameters(), self.critic_target.parameters()):
            p_t.data.mul_(1.0 - self.tau).add_(self.tau * p_o.data)

        self.train_step += 1

        # ── 日志（每60秒）────────────────────────────────────────────
        now = time.time()
        if now - self.last_report_monitor_time >= 60:
            results = {
                "critic_loss": round(critic_loss.item(), 4),
                "actor_loss": round(actor_loss.item(), 4),
                "alpha": round(self.alpha, 4),
                "entropy": round(entropy.item(), 4),
                "buffer_size": len(self.replay_buffer),
                "train_step": self.train_step,
            }
            if self.logger:
                self.logger.info(
                    f"[SAC] step:{self.train_step} "
                    f"critic:{results['critic_loss']} "
                    f"actor:{results['actor_loss']} "
                    f"alpha:{results['alpha']:.4f} "
                    f"H:{results['entropy']:.3f}/{self.target_entropy:.3f} "
                    f"buf:{results['buffer_size']}"
                )
            if self.monitor:
                self.monitor.put_data({os.getpid(): results})
            self.last_report_monitor_time = now

    # ── 模型存储 ─────────────────────────────────────────────────────
    def save_model(self, path, id="1"):
        torch.save(
            {k: v.clone().cpu() for k, v in self.actor.state_dict().items()},
            f"{path}/actor.ckpt-{id}.pkl",
        )
        torch.save(
            {k: v.clone().cpu() for k, v in self.critic.state_dict().items()},
            f"{path}/critic.ckpt-{id}.pkl",
        )
        if self.logger:
            self.logger.info(f"[SAC] saved model id={id} to {path}")

    def load_model(self, path, id="1"):
        try:
            self.actor.load_state_dict(
                torch.load(f"{path}/actor.ckpt-{id}.pkl", map_location=self.device)
            )
            self.critic.load_state_dict(
                torch.load(f"{path}/critic.ckpt-{id}.pkl", map_location=self.device)
            )
            self.critic_target.load_state_dict(self.critic.state_dict())
            if self.logger:
                self.logger.info(f"[SAC] loaded model id={id} from {path}")
        except FileNotFoundError:
            if self.logger:
                self.logger.info(f"[SAC] no checkpoint at {path}, training from scratch")
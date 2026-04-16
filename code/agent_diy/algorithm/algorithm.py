#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Discrete SAC algorithm for Gorge Chase.
峡谷追猎 离散动作 SAC 算法实现。

损失组成：
  critic_loss = MSE( Q_i(s,a),  r + γ*(1-done)* Σ_a' π(a'|s') * [min(Q1',Q2')(s',a') - α*log π(a'|s')] )
  actor_loss  = Σ_a π(a|s) * [ α*log π(a|s) - min(Q1,Q2)(s,a) ]
  alpha_loss  = -α * (H[π] - H_target)   (auto-alpha)

参考：Christodoulou (2019) "Soft Actor-Critic for Discrete Action Settings"
"""

import os
import time
import copy

import numpy as np
import torch
import torch.nn.functional as F

from agent_diy.conf.conf import Config
from agent_diy.model.model import Actor, Critic


class Algorithm:
    def __init__(self, actor, critic, device=None, logger=None, monitor=None):
        self.device = device
        self.actor = actor           # Actor 网络
        self.critic = critic         # 在线双Q网络
        self.logger = logger
        self.monitor = monitor

        # 目标Q网络（软更新，不参与梯度）
        self.critic_target = copy.deepcopy(critic)
        for p in self.critic_target.parameters():
            p.requires_grad = False

        self.action_num = Config.ACTION_NUM
        self.gamma = Config.GAMMA
        self.tau = Config.TAU
        self.grad_clip = Config.GRAD_CLIP_RANGE

        # 优化器
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=Config.INIT_LEARNING_RATE_START
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=Config.INIT_LEARNING_RATE_START
        )

        # 自动熵调整 α
        # 目标熵：H_target = ratio * log(|A|)（鼓励接近均匀分布）
        self.target_entropy = Config.TARGET_ENTROPY_RATIO * np.log(self.action_num)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha = self.log_alpha.exp().item()
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=Config.ALPHA_LR
        )

        self.train_step = 0
        self.last_report_monitor_time = 0

    def learn(self, batch):
        """One SAC update step on a sampled batch.

        对采样的一批转换执行一次SAC更新（Critic + Actor + Alpha）。
        """
        # ── 解包批次 ─────────────────────────────────────────────────
        obs = torch.stack([f.obs for f in batch]).to(self.device)
        legal_action = torch.stack([f.legal_action for f in batch]).to(self.device)
        act = torch.stack([f.act for f in batch]).to(self.device).long().view(-1)
        reward = torch.stack([f.reward for f in batch]).to(self.device).view(-1, 1)
        next_obs = torch.stack([f.next_obs for f in batch]).to(self.device)
        next_legal = torch.stack([f.next_legal_action for f in batch]).to(self.device)
        done = torch.stack([f.done for f in batch]).to(self.device).view(-1, 1)

        # ── 1. Critic 损失 ────────────────────────────────────────────
        with torch.no_grad():
            # 下一状态的策略概率分布
            next_probs = self.actor(next_obs, next_legal)          # (B, A)
            next_log_probs = torch.log(next_probs.clamp(1e-9))    # (B, A)

            # 目标Q值：用目标网络
            q1_next, q2_next = self.critic_target(next_obs)        # (B, A)
            min_q_next = torch.min(q1_next, q2_next)               # (B, A)

            # 软贝尔曼目标：期望 over 下一状态所有动作
            # V(s') = Σ_a π(a|s') * [Q(s',a) - α*logπ(a|s')]
            v_next = (next_probs * (min_q_next - self.alpha * next_log_probs)).sum(dim=1, keepdim=True)
            target_q = reward + self.gamma * (1.0 - done) * v_next  # (B, 1)

        q1_all, q2_all = self.critic(obs)                          # (B, A)
        q1 = q1_all.gather(1, act.unsqueeze(1))                   # (B, 1) 取执行动作的Q值
        q2 = q2_all.gather(1, act.unsqueeze(1))

        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.critic_optimizer.step()

        # ── 2. Actor 损失 ─────────────────────────────────────────────
        probs = self.actor(obs, legal_action)                      # (B, A)
        log_probs = torch.log(probs.clamp(1e-9))                  # (B, A)

        with torch.no_grad():
            q1_pi, q2_pi = self.critic(obs)
            min_q_pi = torch.min(q1_pi, q2_pi)                    # (B, A)

        # actor_loss = E_π[ α*logπ - Q ] (期望over所有动作)
        actor_loss = (probs * (self.alpha * log_probs - min_q_pi)).sum(dim=1).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.actor_optimizer.step()

        # ── 3. 自动 α 更新 ────────────────────────────────────────────
        # H[π] = -Σ_a π(a|s) * logπ(a|s)（当前策略的熵）
        with torch.no_grad():
            entropy = -(probs * log_probs).sum(dim=1).mean()

        alpha_loss = self.log_alpha * (entropy - self.target_entropy).detach()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().clamp(1e-4, 1.0).item()

        # ── 4. 软更新目标Q网络 ────────────────────────────────────────
        self._soft_update(self.critic, self.critic_target)

        self.train_step += 1

        # ── 监控日志（每60秒一次）────────────────────────────────────
        now = time.time()
        if now - self.last_report_monitor_time >= 60:
            results = {
                "critic_loss": round(critic_loss.item(), 4),
                "actor_loss": round(actor_loss.item(), 4),
                "alpha": round(self.alpha, 4),
                "entropy": round(entropy.item(), 4),
                "target_entropy": round(self.target_entropy, 4),
            }
            self.logger.info(
                f"[SAC train] step:{self.train_step} "
                f"critic_loss:{results['critic_loss']} "
                f"actor_loss:{results['actor_loss']} "
                f"alpha:{results['alpha']} "
                f"entropy:{results['entropy']:.3f}/{results['target_entropy']:.3f}"
            )
            if self.monitor:
                self.monitor.put_data({os.getpid(): results})
            self.last_report_monitor_time = now

    def _soft_update(self, online, target):
        """Polyak soft update: θ_target ← τ*θ_online + (1-τ)*θ_target."""
        for p_o, p_t in zip(online.parameters(), target.parameters()):
            p_t.data.mul_(1.0 - self.tau)
            p_t.data.add_(self.tau * p_o.data)

    def save_model(self, path, id="1"):
        """Save actor + critic checkpoints."""
        torch.save(
            {k: v.clone().cpu() for k, v in self.actor.state_dict().items()},
            f"{path}/actor.ckpt-{id}.pkl",
        )
        torch.save(
            {k: v.clone().cpu() for k, v in self.critic.state_dict().items()},
            f"{path}/critic.ckpt-{id}.pkl",
        )
        self.logger.info(f"[SAC] saved model to {path} (id={id})")

    def load_model(self, path, id="1"):
        """Load actor + critic checkpoints."""
        actor_path = f"{path}/actor.ckpt-{id}.pkl"
        critic_path = f"{path}/critic.ckpt-{id}.pkl"
        try:
            self.actor.load_state_dict(
                torch.load(actor_path, map_location=self.device)
            )
            self.critic.load_state_dict(
                torch.load(critic_path, map_location=self.device)
            )
            # 同步更新目标网络
            self.critic_target.load_state_dict(self.critic.state_dict())
            self.logger.info(f"[SAC] loaded model from {path} (id={id})")
        except FileNotFoundError:
            self.logger.info(f"[SAC] no checkpoint found at {path}, training from scratch")
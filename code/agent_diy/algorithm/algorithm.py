#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Discrete SAC algorithm for Gorge Chase — distributed-compatible.
峡谷追猎 离散动作 SAC 算法（兼容分布式架构，多通道空间特征版）。

分布式架构适配：
  learn(list_sample_data) 由框架在 Learner 侧调用，接收 Actor 发来的一批样本：
    1. 将样本批量推入内置 ReplayBuffer
    2. 若 buffer 已达到 LEARNING_STARTS，执行固定次数 SAC 梯度更新

损失：
  critic_loss = SmoothL1(Q_i(s,a), r + γ*(1-d)*V(s'))
  actor_loss  = E_π[α*logπ(a|s) - min(Q1,Q2)(s,a)]
  alpha_loss  = -logα * (H[π] - H_target)
"""

import os
import time
import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from agent_diy.conf.conf import Config
from agent_diy.feature.definition import PrioritizedReplayBuffer


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

        # 学习率余弦衰减调度器
        self.actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.actor_optimizer,
            T_max=Config.LR_DECAY_STEPS,
            eta_min=Config.INIT_LEARNING_RATE_END,
        )
        self.critic_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.critic_optimizer,
            T_max=Config.LR_DECAY_STEPS,
            eta_min=Config.INIT_LEARNING_RATE_END,
        )

        # 自动熵调整 α
        # 约定：Config.TARGET_ENTROPY_RATIO 为正数比例，负号在公式里体现
        self.target_entropy = -Config.TARGET_ENTROPY_RATIO * np.log(Config.ACTION_NUM)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha = self.log_alpha.exp().clamp(Config.ALPHA_MIN, Config.ALPHA_MAX).item()
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=Config.ALPHA_LR)

        # ReplayBuffer（PER）
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=Config.REPLAY_BUFFER_SIZE,
            alpha=Config.PER_ALPHA,
            beta_start=Config.PER_BETA_START,
            beta_frames=Config.PER_BETA_FRAMES,
        )

        # 混合精度
        self.scaler = GradScaler(enabled=Config.USE_AMP)
        self.use_amp = Config.USE_AMP

        self.train_step = 0
        self.last_report_monitor_time = 0

        # 滑动窗口损失累积
        self._loss_accum = {
            "critic_loss": 0.0,
            "actor_loss": 0.0,
            "alpha_loss": 0.0,
            "entropy": 0.0,
            "count": 0,
        }

    # ── Learner 训练入口 ─────────────────────────────────────────────
    def learn(self, list_sample_data):
        """
        Receive one batch/episode/chunk of transitions, push to buffer, then train.
        接收 Actor 发来的一批样本，推入 ReplayBuffer，再执行固定次数 SAC 更新。
        """
        # 1) 推入回放池
        self.replay_buffer.push_batch(list_sample_data)

        # 2) 预热阶段不训练
        if len(self.replay_buffer) < Config.LEARNING_STARTS:
            if self.logger:
                self.logger.info(
                    f"[SAC] warming up buffer: {len(self.replay_buffer)}/{Config.LEARNING_STARTS}"
                )
            return None

        # 3) 固定次数梯度更新
        n_updates = Config.UPDATES_PER_LEARN

        all_tree_indices = []
        all_td_errors = []
        td_abs_mean_list = []

        for _ in range(n_updates):
            batch, tree_indices, is_weights = self.replay_buffer.sample(self.batch_size)

            if len(batch) == 0:
                continue

            td_errors = self._update(batch, is_weights)

            if td_errors is not None and len(td_errors) > 0:
                td_abs_mean_list.append(float(np.mean(np.abs(td_errors))))

            if len(tree_indices) > 0 and td_errors is not None and len(td_errors) > 0:
                min_len = min(len(tree_indices), len(td_errors))
                all_tree_indices.extend(tree_indices[:min_len])
                all_td_errors.extend(td_errors[:min_len])

        # 4) PER 优先级回写
        if len(all_tree_indices) > 0 and len(all_td_errors) > 0:
            self.replay_buffer.update_priorities(
                all_tree_indices, np.array(all_td_errors)
            )

        # 5) 本轮没有有效更新
        if self._loss_accum["count"] == 0:
            return None

        # 6) 汇总返回
        cnt = self._loss_accum["count"]
        results = {
            "critic_loss": round(self._loss_accum["critic_loss"] / cnt, 4),
            "actor_loss": round(self._loss_accum["actor_loss"] / cnt, 4),
            "alpha_loss": round(self._loss_accum["alpha_loss"] / cnt, 4),
            "entropy": round(self._loss_accum["entropy"] / cnt, 4),
            "total_loss": round(
                (self._loss_accum["critic_loss"] + self._loss_accum["actor_loss"]) / cnt, 4
            ),
            "alpha": round(self.alpha, 4),
            "target_entropy": round(self.target_entropy, 4),
            "buffer_size": len(self.replay_buffer),
            "train_step": self.train_step,
            "beta": round(self.replay_buffer.beta, 4),
            "effective_updates": int(self._loss_accum["count"]),
            "td_abs_mean": round(
                float(np.mean(td_abs_mean_list)) if td_abs_mean_list else 0.0, 4
            ),
        }

        # 返回后重置累积器
        self._loss_accum = {k: 0.0 for k in self._loss_accum}
        return results

    # ── SAC 单次梯度更新 ─────────────────────────────────────────────
    def _update(self, batch, is_weights=None):
        if len(batch) == 0:
            return None

        obs = torch.stack([f.obs for f in batch]).to(self.device)
        legal = torch.stack([f.legal_action for f in batch]).to(self.device)
        act = torch.stack([f.act for f in batch]).to(self.device).long().view(-1)
        rew = torch.stack([f.reward for f in batch]).to(self.device).view(-1, 1)
        next_obs = torch.stack([f.next_obs for f in batch]).to(self.device)
        next_legal = torch.stack([f.next_legal_action for f in batch]).to(self.device)
        done = torch.stack([f.done for f in batch]).to(self.device).view(-1, 1)

        # PER 重要性采样权重
        is_weights_t = None
        if is_weights is not None and len(is_weights) > 0:
            is_weights_t = torch.tensor(
                is_weights, dtype=torch.float32, device=self.device
            ).view(-1, 1)

        # ── Critic 更新 ─────────────────────────────────────────────
        with autocast(enabled=self.use_amp):
            with torch.no_grad():
                next_probs = self.actor(next_obs, next_legal)         # [B, A]
                next_log_p = torch.log(next_probs.clamp(1e-9))        # [B, A]

                q1_t, q2_t = self.critic_target(next_obs, next_legal)
                min_q_t = torch.min(q1_t, q2_t)

                # V(s') = Σ_a π(a|s') * (Q(s',a) - α log π(a|s'))
                v_next = (next_probs * (min_q_t - self.alpha * next_log_p)).sum(
                    dim=1, keepdim=True
                )
                target_q = rew + self.gamma * (1.0 - done) * v_next

            q1_all, q2_all = self.critic(obs, legal)
            q1 = q1_all.gather(1, act.unsqueeze(1))
            q2 = q2_all.gather(1, act.unsqueeze(1))

            td_error_1 = q1 - target_q
            td_error_2 = q2 - target_q

            critic_loss_elem = (
                F.smooth_l1_loss(q1, target_q, reduction="none") +
                F.smooth_l1_loss(q2, target_q, reduction="none")
            )

            if is_weights_t is not None:
                is_weights_norm = is_weights_t / (is_weights_t.max() + 1e-8)
                critic_loss = (critic_loss_elem * is_weights_norm).mean()
            else:
                critic_loss = critic_loss_elem.mean()

        self.critic_optimizer.zero_grad()
        self.scaler.scale(critic_loss).backward()
        self.scaler.unscale_(self.critic_optimizer)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.scaler.step(self.critic_optimizer)

        # ── Actor 更新 ──────────────────────────────────────────────
        with autocast(enabled=self.use_amp):
            probs = self.actor(obs, legal)
            log_p = torch.log(probs.clamp(1e-9))

            with torch.no_grad():
                q1_pi, q2_pi = self.critic(obs, legal)
                min_q_pi = torch.min(q1_pi, q2_pi)

            actor_loss = (probs * (self.alpha * log_p - min_q_pi)).sum(dim=1).mean()

        self.actor_optimizer.zero_grad()
        self.scaler.scale(actor_loss).backward()
        self.scaler.unscale_(self.actor_optimizer)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.scaler.step(self.actor_optimizer)

        # ── α 更新 ─────────────────────────────────────────────────
        with autocast(enabled=self.use_amp):
            with torch.no_grad():
                entropy = -(probs * log_p).sum(dim=1).mean()

            alpha_loss = -self.log_alpha * (entropy - self.target_entropy).detach()

        self.alpha_optimizer.zero_grad()
        self.scaler.scale(alpha_loss).backward()
        self.scaler.step(self.alpha_optimizer)

        self.alpha = self.log_alpha.exp().clamp(
            Config.ALPHA_MIN, Config.ALPHA_MAX
        ).item()

        # 更新 AMP scaler
        self.scaler.update()

        # ── 软更新目标网络 ───────────────────────────────────────────
        for p_o, p_t in zip(self.critic.parameters(), self.critic_target.parameters()):
            p_t.data.mul_(1.0 - self.tau).add_(self.tau * p_o.data)

        # ── 学习率调度 ───────────────────────────────────────────────
        if self.train_step < Config.LR_DECAY_STEPS:
            self.actor_scheduler.step()
            self.critic_scheduler.step()

        self.train_step += 1

        # ── 累积统计 ────────────────────────────────────────────────
        self._loss_accum["critic_loss"] += critic_loss.item()
        self._loss_accum["actor_loss"] += actor_loss.item()
        self._loss_accum["alpha_loss"] += alpha_loss.item()
        self._loss_accum["entropy"] += entropy.item()
        self._loss_accum["count"] += 1

        # 用于 PER 更新优先级：两个 Q 的绝对 TD 误差均值
        avg_td_error = (
            (td_error_1.abs() + td_error_2.abs()) / 2.0
        ).detach().cpu().numpy().flatten()

        # ── 周期日志 ────────────────────────────────────────────────
        now = time.time()
        if now - self.last_report_monitor_time >= 60:
            cnt = max(self._loss_accum["count"], 1)
            results = {
                "critic_loss": round(self._loss_accum["critic_loss"] / cnt, 4),
                "actor_loss": round(self._loss_accum["actor_loss"] / cnt, 4),
                "alpha_loss": round(self._loss_accum["alpha_loss"] / cnt, 4),
                "entropy": round(self._loss_accum["entropy"] / cnt, 4),
                "total_loss": round(
                    (self._loss_accum["critic_loss"] + self._loss_accum["actor_loss"]) / cnt, 4
                ),
                "alpha": round(self.alpha, 4),
                "target_entropy": round(self.target_entropy, 4),
                "buffer_size": len(self.replay_buffer),
                "train_step": self.train_step,
                "beta": round(self.replay_buffer.beta, 4),
            }

            if self.logger:
                self.logger.info(
                    f"[SAC] step:{self.train_step} "
                    f"total_loss:{results['total_loss']} "
                    f"critic_loss:{results['critic_loss']} "
                    f"actor_loss:{results['actor_loss']} "
                    f"entropy:{results['entropy']:.3f}/{results['target_entropy']:.3f} "
                    f"alpha:{results['alpha']:.4f} "
                    f"beta:{results['beta']:.4f} "
                    f"buf:{results['buffer_size']}"
                )

            if self.monitor:
                self.monitor.put_data({os.getpid(): results})

            self._loss_accum = {k: 0.0 for k in self._loss_accum}
            self.last_report_monitor_time = now

        return avg_td_error

    # ── 模型存储（完整训练状态，支持断点续训）──────────────────────
    def save_model(self, path, id="1"):
        torch.save(
            {k: v.clone().cpu() for k, v in self.actor.state_dict().items()},
            f"{path}/actor.ckpt-{id}.pkl",
        )
        torch.save(
            {k: v.clone().cpu() for k, v in self.critic.state_dict().items()},
            f"{path}/critic.ckpt-{id}.pkl",
        )
        torch.save(
            {k: v.clone().cpu() for k, v in self.critic_target.state_dict().items()},
            f"{path}/critic_target.ckpt-{id}.pkl",
        )
        torch.save(
            {
                "actor_opt": self.actor_optimizer.state_dict(),
                "critic_opt": self.critic_optimizer.state_dict(),
                "alpha_opt": self.alpha_optimizer.state_dict(),
                "log_alpha": self.log_alpha.item(),
                "train_step": self.train_step,
            },
            f"{path}/train_state.ckpt-{id}.pkl",
        )

        if self.logger:
            self.logger.info(
                f"[SAC] saved model id={id} to {path} (train_step={self.train_step})"
            )

    def load_model(self, path, id="1"):
        try:
            self.actor.load_state_dict(
                torch.load(f"{path}/actor.ckpt-{id}.pkl", map_location=self.device)
            )
        except FileNotFoundError:
            if self.logger:
                self.logger.info("[SAC] no actor checkpoint, training from scratch")
            return

        try:
            self.critic.load_state_dict(
                torch.load(f"{path}/critic.ckpt-{id}.pkl", map_location=self.device)
            )
        except FileNotFoundError:
            pass

        try:
            self.critic_target.load_state_dict(
                torch.load(f"{path}/critic_target.ckpt-{id}.pkl", map_location=self.device)
            )
        except FileNotFoundError:
            self.critic_target.load_state_dict(self.critic.state_dict())

        try:
            state = torch.load(
                f"{path}/train_state.ckpt-{id}.pkl", map_location=self.device
            )
            self.actor_optimizer.load_state_dict(state["actor_opt"])
            self.critic_optimizer.load_state_dict(state["critic_opt"])
            self.alpha_optimizer.load_state_dict(state["alpha_opt"])

            with torch.no_grad():
                self.log_alpha.fill_(state["log_alpha"])

            self.alpha = self.log_alpha.exp().clamp(
                Config.ALPHA_MIN, Config.ALPHA_MAX
            ).item()
            self.train_step = state.get("train_step", 0)

            if self.logger:
                self.logger.info(
                    f"[SAC] loaded model id={id} from {path} "
                    f"(train_step={self.train_step}, alpha={self.alpha:.4f})"
                )
        except FileNotFoundError:
            self.critic_target.load_state_dict(self.critic.state_dict())
            if self.logger:
                self.logger.info(
                    "[SAC] loaded weights only (no train_state), optimizer reset"
                )
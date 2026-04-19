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

        # 学习率余弦衰减调度器（从 LR_START 衰减到 LR_END）
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
        self.target_entropy = Config.TARGET_ENTROPY_RATIO * np.log(Config.ACTION_NUM)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha = self.log_alpha.exp().item()
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=Config.ALPHA_LR)

        # ── Learner 侧内置 PrioritizedReplayBuffer ───────────────────
        # PER参数：α=0.6控制优先级程度，β从0.4逐渐增至1.0消除偏差
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=Config.REPLAY_BUFFER_SIZE,
            alpha=Config.PER_ALPHA,
            beta_start=Config.PER_BETA_START,
            beta_frames=Config.PER_BETA_FRAMES,
        )

        # ── 混合精度训练（AMP）────────────────────────────────────────
        self.scaler = GradScaler(enabled=Config.USE_AMP)
        self.use_amp = Config.USE_AMP

        self.train_step = 0
        self.last_report_monitor_time = 0

        # 滑动窗口损失累积（用于每分钟上报平均值，避免瞬时抖动）
        self._loss_accum = {
            "critic_loss": 0.0,
            "actor_loss": 0.0,
            "alpha_loss": 0.0,
            "entropy": 0.0,
            "count": 0,
        }

    # ── 框架调用入口（Learner 侧）────────────────────────────────────
    def learn(self, list_sample_data):
        """Receive one episode's transitions, push to buffer, then train.

        接收 Actor 发来的一局数据，推入 ReplayBuffer，再执行等量次数的 SAC 更新。
        符合分布式架构：框架在 Learner 侧调用此方法，Actor 侧不直接训练。
        返回损失字典（对齐 PPO 字段格式），供框架实时展示。
        """
        # 1. 将本局数据推入 ReplayBuffer
        self.replay_buffer.push_batch(list_sample_data)

        # 2. buffer 未达到预热量时跳过训练
        if len(self.replay_buffer) < Config.LEARNING_STARTS:
            if self.logger:
                self.logger.info(
                    f"[SAC] warming up buffer: {len(self.replay_buffer)}/{Config.LEARNING_STARTS}"
                )
            return None

        # 3. 执行固定次数梯度更新（控制UTD比，降低过快收敛风险）
        n_updates = Config.UPDATES_PER_LEARN
        
        # PER: 收集所有批次的时间步索引和TD误差
        all_tree_indices = []
        all_td_errors = []
        
        for _ in range(n_updates):
            # PER采样：返回样本、时间步索引、IS权重
            batch, tree_indices, is_weights = self.replay_buffer.sample(self.batch_size)
            
            # 检查采样是否成功
            if len(batch) == 0:
                continue
            
            # 存储IS权重用于损失计算
            if len(is_weights) > 0:
                self._is_weights = torch.tensor(is_weights, dtype=torch.float32, device=self.device).view(-1, 1)
            else:
                self._is_weights = None
            
            # 执行梯度更新并获取TD误差
            td_errors = self._update(batch)
            
            # 收集索引和TD误差用于批量更新优先级
            if len(tree_indices) > 0 and td_errors is not None and len(td_errors) > 0:
                # 确保长度匹配
                min_len = min(len(tree_indices), len(td_errors))
                all_tree_indices.extend(tree_indices[:min_len])
                all_td_errors.extend(td_errors[:min_len])
        
        # PER: 批量更新优先级
        if len(all_tree_indices) > 0 and len(all_td_errors) > 0:
            self.replay_buffer.update_priorities(all_tree_indices, np.array(all_td_errors))

        # 4. 返回本局周期平均损失（实时上报给框架，对齐PPO字段名）
        cnt = max(self._loss_accum["count"], 1)
        results = {
            # 对齐PPO字段：value_loss / policy_loss / entropy_loss / total_loss
            "value_loss":   round(self._loss_accum["critic_loss"] / cnt, 4),
            "policy_loss":  round(self._loss_accum["actor_loss"]  / cnt, 4),
            "entropy_loss": round(self._loss_accum["entropy"]      / cnt, 4),
            "total_loss":   round(
                (self._loss_accum["critic_loss"] + self._loss_accum["actor_loss"]) / cnt, 4
            ),
            # SAC 专有字段
            "alpha_loss":   round(self._loss_accum["alpha_loss"]  / cnt, 4),
            "alpha":        round(self.alpha, 4),
            "target_entropy": round(self.target_entropy, 4),
            "buffer_size":  len(self.replay_buffer),
            "train_step":   self.train_step,
            "beta":         round(self.replay_buffer.beta, 4),  # PER β值
        }
        # 返回后重置累积器，避免历史数据污染下一周期上报
        self._loss_accum = {k: 0.0 for k in self._loss_accum}
        return results

    # ── SAC 单次梯度更新 ─────────────────────────────────────────────
    def _update(self, batch):
        # 边界检查：如果batch为空，返回None
        if len(batch) == 0:
            return None
            
        obs = torch.stack([f.obs for f in batch]).to(self.device)
        legal = torch.stack([f.legal_action for f in batch]).to(self.device)
        act = torch.stack([f.act for f in batch]).to(self.device).long().view(-1)
        rew = torch.stack([f.reward for f in batch]).to(self.device).view(-1, 1)
        next_obs = torch.stack([f.next_obs for f in batch]).to(self.device)
        next_legal = torch.stack([f.next_legal_action for f in batch]).to(self.device)
        done = torch.stack([f.done for f in batch]).to(self.device).view(-1, 1)

        # ── Critic 更新（混合精度）──────────────────────────────────────
        with autocast(enabled=self.use_amp):
            with torch.no_grad():
                next_probs = self.actor(next_obs, next_legal)               # (B, A)
                next_log_p = torch.log(next_probs.clamp(1e-9))
                q1_t, q2_t = self.critic_target(next_obs, next_legal)
                min_q_t = torch.min(q1_t, q2_t)
                # 软贝尔曼目标 V(s') = Σ_a π * (Q - α*logπ)
                v_next = (next_probs * (min_q_t - self.alpha * next_log_p)).sum(1, keepdim=True)
                target_q = rew + self.gamma * (1.0 - done) * v_next

            q1_all, q2_all = self.critic(obs, legal)
            q1 = q1_all.gather(1, act.unsqueeze(1))
            q2 = q2_all.gather(1, act.unsqueeze(1))
            
            # PER: 返回未归约的TD误差用于优先级更新
            td_error_1 = q1 - target_q
            td_error_2 = q2 - target_q
            critic_loss = F.mse_loss(q1, target_q, reduction='none') + F.mse_loss(q2, target_q, reduction='none')
            
            # 如果是PER采样，应用重要性采样权重
            if hasattr(self, '_is_weights') and self._is_weights is not None:
                critic_loss = (critic_loss * self._is_weights).mean()
            else:
                critic_loss = critic_loss.mean()

        self.critic_optimizer.zero_grad()
        self.scaler.scale(critic_loss).backward()
        self.scaler.unscale_(self.critic_optimizer)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.scaler.step(self.critic_optimizer)

        # ── Actor 更新（混合精度）───────────────────────────────────────
        with autocast(enabled=self.use_amp):
            probs = self.actor(obs, legal)
            log_p = torch.log(probs.clamp(1e-9))
            with torch.no_grad():
                q1_pi, q2_pi = self.critic(obs, legal)
                min_q_pi = torch.min(q1_pi, q2_pi)
            actor_loss = (probs * (self.alpha * log_p - min_q_pi)).sum(1).mean()

        self.actor_optimizer.zero_grad()
        self.scaler.scale(actor_loss).backward()
        self.scaler.unscale_(self.actor_optimizer)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.scaler.step(self.actor_optimizer)

        # ── α 更新（混合精度）───────────────────────────────────────────
        with autocast(enabled=self.use_amp):
            with torch.no_grad():
                entropy = -(probs * log_p).sum(1).mean()
            alpha_loss = self.log_alpha * (entropy - self.target_entropy).detach()

        self.alpha_optimizer.zero_grad()
        self.scaler.scale(alpha_loss).backward()
        self.scaler.step(self.alpha_optimizer)
        self.alpha = self.log_alpha.exp().clamp(Config.ALPHA_MIN, Config.ALPHA_MAX).item()

        # 更新梯度缩放器
        self.scaler.update()

        # ── 软更新目标网络 ────────────────────────────────────────────
        for p_o, p_t in zip(self.critic.parameters(), self.critic_target.parameters()):
            p_t.data.mul_(1.0 - self.tau).add_(self.tau * p_o.data)

        # ── 学习率调度步进 ────────────────────────────────────────────
        if self.train_step < Config.LR_DECAY_STEPS:
            self.actor_scheduler.step()
            self.critic_scheduler.step()

        self.train_step += 1

        # ── 累积损失（用于周期平均上报）─────────────────────────────
        self._loss_accum["critic_loss"] += critic_loss.item()
        self._loss_accum["actor_loss"]  += actor_loss.item()
        self._loss_accum["alpha_loss"]  += alpha_loss.item()
        self._loss_accum["entropy"]     += entropy.item()
        self._loss_accum["count"]       += 1

        # ── PER: 返回TD误差用于优先级更新 ──────────────────────────────
        # 使用两个Q网络TD误差的平均值
        avg_td_error = ((td_error_1.abs() + td_error_2.abs()) / 2).detach().cpu().numpy().flatten()
        
        # ── 日志（每60秒上报周期均值，字段名对齐PPO）──────────────────
        now = time.time()
        if now - self.last_report_monitor_time >= 60:
            cnt = max(self._loss_accum["count"], 1)
            results = {
                # 对齐PPO字段名，框架/看板可统一展示
                "total_loss":   round(
                    (self._loss_accum["critic_loss"] + self._loss_accum["actor_loss"]) / cnt, 4
                ),
                "value_loss":   round(self._loss_accum["critic_loss"] / cnt, 4),
                "policy_loss":  round(self._loss_accum["actor_loss"]  / cnt, 4),
                "entropy_loss": round(self._loss_accum["entropy"]     / cnt, 4),
                # SAC 专有字段
                "alpha_loss":   round(self._loss_accum["alpha_loss"]  / cnt, 4),
                "alpha":        round(self.alpha, 4),
                "target_entropy": round(self.target_entropy, 4),
                "buffer_size":  len(self.replay_buffer),
                "train_step":   self.train_step,
                "beta":         round(self.replay_buffer.beta, 4),  # PER β值
            }
            if self.logger:
                self.logger.info(
                    f"[SAC] step:{self.train_step} "
                    f"total_loss:{results['total_loss']} "
                    f"value_loss:{results['value_loss']} "
                    f"policy_loss:{results['policy_loss']} "
                    f"entropy_loss:{results['entropy_loss']:.3f}/{results['target_entropy']:.3f} "
                    f"alpha:{results['alpha']:.4f} "
                    f"beta:{results['beta']:.4f} "
                    f"buf:{results['buffer_size']}"
                )
            if self.monitor:
                self.monitor.put_data({os.getpid(): results})
            # 重置累积器
            self._loss_accum = {k: 0.0 for k in self._loss_accum}
            self.last_report_monitor_time = now
        
        return avg_td_error

    # ── 模型存储（完整训练状态，支持断点续训）──────────────────────
    def save_model(self, path, id="1"):
        # 网络权重
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
        # 优化器状态 + log_alpha + train_step（断点续训关键）
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
        # 加载网络权重
        try:
            self.actor.load_state_dict(
                torch.load(f"{path}/actor.ckpt-{id}.pkl", map_location=self.device)
            )
        except FileNotFoundError:
            if self.logger:
                self.logger.info(f"[SAC] no actor checkpoint, training from scratch")
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
            # 目标网络文件不存在时从 critic 复制
            self.critic_target.load_state_dict(self.critic.state_dict())

        # 恢复优化器状态 + log_alpha + train_step（断点续训核心）
        try:
            state = torch.load(
                f"{path}/train_state.ckpt-{id}.pkl", map_location=self.device
            )
            self.actor_optimizer.load_state_dict(state["actor_opt"])
            self.critic_optimizer.load_state_dict(state["critic_opt"])
            self.alpha_optimizer.load_state_dict(state["alpha_opt"])
            # 恢复 log_alpha（可训练参数，需要特殊处理）
            with torch.no_grad():
                self.log_alpha.fill_(state["log_alpha"])
            self.alpha = self.log_alpha.exp().clamp(0.2, 3.0).item()  # 增大下限至0.2，防止熵过早衰减
            self.train_step = state.get("train_step", 0)
            if self.logger:
                self.logger.info(
                    f"[SAC] loaded model id={id} from {path} "
                    f"(train_step={self.train_step}, alpha={self.alpha:.4f})"
                )
        except FileNotFoundError:
            # 旧版checkpoint没有train_state文件，只恢复权重
            self.critic_target.load_state_dict(self.critic.state_dict())
            if self.logger:
                self.logger.info(
                    f"[SAC] loaded weights only (no train_state), optimizer reset"
                )

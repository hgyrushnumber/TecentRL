#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

SAC algorithm implementation for Gorge Chase.
峡谷追猎 SAC（离散动作）算法实现。
"""

import copy
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from agent_diy.conf.conf import Config


class Algorithm:
    def __init__(self, model, optimizer, device=None, logger=None, monitor=None):
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.parameters = [p for pg in self.optimizer.param_groups for p in pg["params"]]
        self.logger = logger
        self.monitor = monitor

        self.label_size = Config.ACTION_NUM
        self.gamma = Config.GAMMA
        self.tau = Config.TAU
        self.alpha = Config.ALPHA
        self.auto_alpha = getattr(Config, "AUTO_ALPHA", False)
        self.target_entropy = getattr(Config, "TARGET_ENTROPY", 2.0)
        self.alpha_min = float(getattr(Config, "ALPHA_MIN", 1e-4))
        self.alpha_max = float(getattr(Config, "ALPHA_MAX", 20.0))
        self.target_q_clip = float(getattr(Config, "TARGET_Q_CLIP", 8.0))
        self.actor_update_interval = int(max(1, getattr(Config, "ACTOR_UPDATE_INTERVAL", 2)))
        self.critic_use_huber = bool(getattr(Config, "CRITIC_USE_HUBER", True))
        self.alpha_loss_value = 0.0
        if self.auto_alpha:
            init_alpha = max(float(Config.ALPHA), 1e-6)
            self.log_alpha = torch.tensor(np.log(init_alpha), device=self.device, requires_grad=True)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=getattr(Config, "ALPHA_LR", 1e-4))
            self.alpha = float(self.log_alpha.exp().clamp(self.alpha_min, self.alpha_max).item())

        self.last_report_monitor_time = 0
        self.train_step = 0
        self.last_actor_loss = 0.0
        self.last_critic_loss = 0.0
        self.last_q_target_mean = 0.0
        self.last_q1_mean = 0.0
        self.last_q2_mean = 0.0
        self.last_q_gap = 0.0
        self.last_entropy = 0.0
        self.last_critic_grad_norm = 0.0
        self.last_actor_grad_norm = 0.0
        self.last_critic_grad_norm_post = 0.0
        self.last_actor_grad_norm_post = 0.0

        self.target_model = copy.deepcopy(self.model).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        for p in self.target_model.parameters():
            p.requires_grad = False

    def learn(self, list_sample_data):
        if not list_sample_data:
            return

        obs = torch.stack([f.obs for f in list_sample_data]).to(self.device)
        legal_action = torch.stack([f.legal_action for f in list_sample_data]).to(self.device)
        act = torch.stack([f.act for f in list_sample_data]).to(self.device).long().view(-1, 1)
        reward = torch.stack([f.reward for f in list_sample_data]).to(self.device).view(-1, 1)
        done = torch.stack([f.done for f in list_sample_data]).to(self.device).view(-1, 1)
        next_obs = torch.stack([f.next_obs for f in list_sample_data]).to(self.device)
        next_legal_action = torch.stack([f.next_legal_action for f in list_sample_data]).to(self.device)

        self.model.set_train_mode()
        self.optimizer.zero_grad()
        critic_loss, critic_info = self._compute_critic_loss(
            obs=obs,
            legal_action=legal_action,
            act=act,
            reward=reward,
            done=done,
            next_obs=next_obs,
            next_legal_action=next_legal_action,
        )
        critic_loss.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters, Config.GRAD_CLIP_RANGE)
        critic_grad_norm_post = self._grad_total_norm(self.parameters)
        self.optimizer.step()

        actor_grad_norm = torch.tensor(0.0, device=self.device)
        actor_loss, actor_entropy = self._compute_actor_loss(obs=obs, legal_action=legal_action)
        if self.train_step % self.actor_update_interval == 0:
            self.optimizer.zero_grad()
            actor_loss.backward()
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters, Config.GRAD_CLIP_RANGE)
            actor_grad_norm_post = self._grad_total_norm(self.parameters)
            self.optimizer.step()
        else:
            actor_grad_norm_post = torch.tensor(0.0, device=self.device)

        self.last_critic_loss = float(critic_loss.item())
        self.last_actor_loss = float(actor_loss.item())
        self.last_q_target_mean = float(critic_info["q_target_mean"].item())
        self.last_q1_mean = float(critic_info["q1_mean"].item())
        self.last_q2_mean = float(critic_info["q2_mean"].item())
        self.last_q_gap = float(critic_info["q_gap"].item())
        self.last_entropy = float(actor_entropy.item())
        self.last_critic_grad_norm = float(critic_grad_norm.item() if hasattr(critic_grad_norm, "item") else critic_grad_norm)
        self.last_actor_grad_norm = float(actor_grad_norm.item() if hasattr(actor_grad_norm, "item") else actor_grad_norm)
        self.last_critic_grad_norm_post = float(
            critic_grad_norm_post.item() if hasattr(critic_grad_norm_post, "item") else critic_grad_norm_post
        )
        self.last_actor_grad_norm_post = float(
            actor_grad_norm_post.item() if hasattr(actor_grad_norm_post, "item") else actor_grad_norm_post
        )

        if self.auto_alpha:
            self.alpha_optimizer.zero_grad()
            # Optimize log(alpha) directly for better numerical stability.
            alpha_loss = (self.log_alpha * (actor_entropy - self.target_entropy).detach()).mean()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            with torch.no_grad():
                self.log_alpha.data.clamp_(np.log(self.alpha_min), np.log(self.alpha_max))
            self.alpha = float(self.log_alpha.exp().clamp(self.alpha_min, self.alpha_max).item())
            self.alpha_loss_value = float(alpha_loss.item())
        self._soft_update_target()
        self.train_step += 1

        now = time.time()
        if now - self.last_report_monitor_time >= 60:
            total_loss = self.last_critic_loss + self.last_actor_loss
            results = {
                "total_loss": round(total_loss, 4),
                "value_loss": round(self.last_critic_loss, 4),
                "policy_loss": round(self.last_actor_loss, 4),
                # NOTE: this is policy entropy value (not a standalone optimized "entropy loss")
                "entropy": round(self.last_entropy, 4),
                "entropy_loss": round(self.last_entropy, 4),  # backward-compatible metric key
                "entropy_gap": round(abs(self.last_entropy - self.target_entropy), 4),
                "reward": round(reward.mean().item(), 4),
                "q_target_mean": round(self.last_q_target_mean, 4),
                "q1_mean": round(self.last_q1_mean, 4),
                "q2_mean": round(self.last_q2_mean, 4),
                "q_gap": round(self.last_q_gap, 4),
                "legal_action_count": round(legal_action.sum(dim=1).float().mean().item(), 4),
                "done_rate": round(done.float().mean().item(), 4),
                "grad_norm": round(max(self.last_critic_grad_norm, self.last_actor_grad_norm), 4),
                "critic_grad_norm": round(self.last_critic_grad_norm, 4),
                "actor_grad_norm": round(self.last_actor_grad_norm, 4),
                "grad_norm_post": round(max(self.last_critic_grad_norm_post, self.last_actor_grad_norm_post), 4),
                "critic_grad_norm_post": round(self.last_critic_grad_norm_post, 4),
                "actor_grad_norm_post": round(self.last_actor_grad_norm_post, 4),
                "alpha": round(self.alpha, 4),
                "alpha_loss": round(self.alpha_loss_value, 4),
                "actor_update_interval": self.actor_update_interval,
            }
            if self.logger:
                self.logger.info(
                    f"[SAC] train_step:{self.train_step} "
                    f"total_loss:{results['total_loss']} "
                    f"value_loss:{results['value_loss']} "
                    f"policy_loss:{results['policy_loss']} "
                    f"entropy:{results['entropy']} "
                    f"entropy_gap:{results['entropy_gap']}"
                )
            if self.monitor:
                self.monitor.put_data({os.getpid(): results})
            self.last_report_monitor_time = now

    def _compute_critic_loss(self, obs, legal_action, act, reward, done, next_obs, next_legal_action):
        logits, q1, q2 = self.model(obs)
        q1_a = q1.gather(1, act)
        q2_a = q2.gather(1, act)

        with torch.no_grad():
            next_logits, next_q1, next_q2 = self.target_model(next_obs)
            next_prob = self._masked_softmax(next_logits, next_legal_action)
            next_log_prob = torch.log(next_prob.clamp_min(1e-8))
            next_min_q = torch.min(next_q1, next_q2)
            next_v = (next_prob * (next_min_q - self.alpha * next_log_prob)).sum(dim=1, keepdim=True)
            q_target = reward + (1.0 - done) * self.gamma * next_v
            q_target = q_target.clamp(-self.target_q_clip, self.target_q_clip)

        if self.critic_use_huber:
            critic_loss = F.smooth_l1_loss(q1_a, q_target) + F.smooth_l1_loss(q2_a, q_target)
        else:
            critic_loss = F.mse_loss(q1_a, q_target) + F.mse_loss(q2_a, q_target)

        return critic_loss, {
            "q_target_mean": q_target.mean(),
            "q1_mean": q1_a.mean(),
            "q2_mean": q2_a.mean(),
            "q_gap": torch.abs(q1_a - q2_a).mean(),
        }

    def _compute_actor_loss(self, obs, legal_action):
        logits_pi, q1_pi, q2_pi = self.model(obs)
        prob = self._masked_softmax(logits_pi, legal_action)
        log_prob = torch.log(prob.clamp_min(1e-8))
        min_q = torch.min(q1_pi, q2_pi)

        actor_loss = (prob * (self.alpha * log_prob - min_q)).sum(dim=1).mean()
        entropy = -(prob * log_prob).sum(dim=1).mean()
        return actor_loss, entropy

    def _masked_softmax(self, logits, legal_action):
        legal_action = legal_action.float()
        masked_logits = logits.masked_fill(legal_action <= 0, -1e9)
        prob = torch.softmax(masked_logits, dim=1)

        invalid_mask = (legal_action.sum(dim=1, keepdim=True) <= 0).float()
        if invalid_mask.any():
            uniform_prob = torch.full_like(prob, 1.0 / prob.size(1))
            prob = invalid_mask * uniform_prob + (1.0 - invalid_mask) * prob
        return prob

    def _soft_update_target(self):
        with torch.no_grad():
            for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def _grad_total_norm(self, parameters):
        grads = [p.grad.detach() for p in parameters if p.grad is not None]
        if not grads:
            return torch.tensor(0.0, device=self.device)
        norms = [torch.norm(g, p=2) for g in grads]
        return torch.norm(torch.stack(norms), p=2)

#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

SAC algorithm implementation for Gorge Chase.
峡谷追猎 SAC（离散动作）算法实现。

保留内容：
1. 使用完整 target_model。
2. Actor / Critic 独立 optimizer。
3. Critic target 使用 target_model.critic(next_obs)。
4. Actor loss 使用离散 SAC 全动作期望。
5. Actor 更新时 critic 不反传，避免污染 critic 梯度。
6. alpha 自动调节。
7. 保留基础监控项，便于判断 SAC 是否稳定。

已删除：
1. reward_safety_clip：reward 只在 preprocessor 里处理。
2. soft_q_clip：不在 algorithm 层强行压缩 q_target。
3. adaptive_target_entropy：使用固定 TARGET_ENTROPY。
4. actor_q_baseline：恢复标准离散 SAC actor loss。
"""

import copy
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from agent_diy.conf.conf import Config


class Algorithm:
    def __init__(self, model, optimizer=None, device=None, logger=None, monitor=None):
        """
        Args:
            model: agent_diy.model.model.Model
            optimizer: 保留参数兼容旧框架，这里不再使用外部传入的 optimizer。
            device: torch device
            logger: logger
            monitor: monitor proxy
        """
        self.device = device
        self.model = model.to(self.device)
        self.logger = logger
        self.monitor = monitor

        # ------------------------------------------------------------------
        # 1) Target model
        # ------------------------------------------------------------------
        self.target_model = copy.deepcopy(self.model).to(self.device)
        self.target_model.eval()
        for p in self.target_model.parameters():
            p.requires_grad = False

        # ------------------------------------------------------------------
        # 2) Parameter groups
        # ------------------------------------------------------------------
        self.actor_parameters = self._get_actor_parameters()
        self.critic_parameters = self._get_critic_parameters()

        # ------------------------------------------------------------------
        # 3) Optimizers
        # ------------------------------------------------------------------
        actor_lr = float(getattr(Config, "ACTOR_LR", getattr(Config, "INIT_LEARNING_RATE_START", 1e-4)))
        critic_lr = float(getattr(Config, "CRITIC_LR", getattr(Config, "INIT_LEARNING_RATE_START", 1e-4)))
        alpha_lr = float(getattr(Config, "ALPHA_LR", 1e-5))

        self.actor_optimizer = torch.optim.Adam(
            params=self.actor_parameters,
            lr=actor_lr,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        self.critic_optimizer = torch.optim.Adam(
            params=self.critic_parameters,
            lr=critic_lr,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        # ------------------------------------------------------------------
        # 4) SAC hyperparameters
        # ------------------------------------------------------------------
        self.action_num = int(Config.ACTION_NUM)
        self.gamma = float(getattr(Config, "GAMMA", 0.98))
        self.tau = float(getattr(Config, "TAU", 0.005))

        self.auto_alpha = bool(getattr(Config, "AUTO_ALPHA", True))
        self.alpha_min = float(getattr(Config, "ALPHA_MIN", 0.01))
        self.alpha_max = float(getattr(Config, "ALPHA_MAX", 2.0))

        # 固定目标熵，避免动态目标熵干扰实验判断
        self.target_entropy = float(getattr(Config, "TARGET_ENTROPY", 1.0))

        init_alpha = float(getattr(Config, "ALPHA", 0.2))
        init_alpha = float(np.clip(init_alpha, self.alpha_min, self.alpha_max))

        if self.auto_alpha:
            self.log_alpha = torch.tensor(
                np.log(max(init_alpha, 1e-8)),
                dtype=torch.float32,
                device=self.device,
                requires_grad=True,
            )
            self.alpha_optimizer = torch.optim.Adam(
                [self.log_alpha],
                lr=alpha_lr,
                betas=(0.9, 0.999),
                eps=1e-8,
            )
            self.alpha = float(self.log_alpha.exp().clamp(self.alpha_min, self.alpha_max).item())
        else:
            self.log_alpha = None
            self.alpha_optimizer = None
            self.alpha = init_alpha

        self.actor_update_interval = int(max(1, getattr(Config, "ACTOR_UPDATE_INTERVAL", 1)))
        self.critic_use_huber = bool(getattr(Config, "CRITIC_USE_HUBER", True))
        self.grad_clip_range = float(getattr(Config, "GRAD_CLIP_RANGE", 2.0))

        # 可选 target Q hard clip，默认关闭。
        # 注意：不使用 soft Q clip，避免隐藏真实 Q target 趋势。
        self.use_target_q_clip = bool(getattr(Config, "USE_TARGET_Q_CLIP", False))
        self.target_q_clip = float(getattr(Config, "TARGET_Q_CLIP", 30.0))

        # ------------------------------------------------------------------
        # 5) Runtime stats
        # ------------------------------------------------------------------
        self.train_step = 0
        self.last_report_monitor_time = 0.0    

        self.last_actor_loss = 0.0
        self.last_critic_loss = 0.0

        self.last_q_target_mean = 0.0
        self.last_q_target_abs_mean = 0.0
        self.last_q_target_max = 0.0
        self.last_q_target_min = 0.0

        self.last_q1_mean = 0.0
        self.last_q2_mean = 0.0
        self.last_q_gap = 0.0

        self.last_entropy = 0.0
        self.last_entropy_gap = 0.0

        self.last_critic_grad_norm = 0.0
        self.last_actor_grad_norm = 0.0
        self.last_critic_grad_norm_post = 0.0
        self.last_actor_grad_norm_post = 0.0

    # ======================================================================
    # Public train entry
    # ======================================================================
    def learn(self, list_sample_data):
        if not list_sample_data:
            return

        # ------------------------------------------------------------------
        # 1) Batch data
        # ------------------------------------------------------------------
        obs = torch.stack([f.obs for f in list_sample_data]).to(self.device).float()
        legal_action = torch.stack([f.legal_action for f in list_sample_data]).to(self.device).float()
        act = torch.stack([f.act for f in list_sample_data]).to(self.device).long().view(-1, 1)
        reward = torch.stack([f.reward for f in list_sample_data]).to(self.device).float().view(-1, 1)
        done = torch.stack([f.done for f in list_sample_data]).to(self.device).float().view(-1, 1)
        next_obs = torch.stack([f.next_obs for f in list_sample_data]).to(self.device).float()
        next_legal_action = torch.stack([f.next_legal_action for f in list_sample_data]).to(self.device).float()

        self.model.set_train_mode()

        # ------------------------------------------------------------------
        # 2) Critic update
        # ------------------------------------------------------------------
        self.critic_optimizer.zero_grad(set_to_none=True)

        critic_loss, critic_info = self._compute_critic_loss(
            obs=obs,
            act=act,
            reward=reward,
            done=done,
            next_obs=next_obs,
            next_legal_action=next_legal_action,
        )

        critic_loss.backward()

        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.critic_parameters,
            self.grad_clip_range,
        )
        critic_grad_norm_post = self._grad_total_norm(self.critic_parameters)

        self.critic_optimizer.step()

        # ------------------------------------------------------------------
        # 3) Actor update
        # ------------------------------------------------------------------
        actor_loss, actor_entropy, actor_info = self._compute_actor_loss(
            obs=obs,
            legal_action=legal_action,
        )

        actor_grad_norm = torch.tensor(0.0, device=self.device)
        actor_grad_norm_post = torch.tensor(0.0, device=self.device)

        should_update_actor = self.train_step % self.actor_update_interval == 0

        if should_update_actor:
            self.actor_optimizer.zero_grad(set_to_none=True)

            actor_loss.backward()

            actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.actor_parameters,
                self.grad_clip_range,
            )
            actor_grad_norm_post = self._grad_total_norm(self.actor_parameters)

            self.actor_optimizer.step()

        # ------------------------------------------------------------------
        # 4) Alpha update
        # ------------------------------------------------------------------
        if self.auto_alpha:
            self.alpha_optimizer.zero_grad(set_to_none=True)

            target_entropy = torch.tensor(
                self.target_entropy,
                device=self.device,
                dtype=torch.float32,
            )

            # 当 entropy < target_entropy 时：
            #   actor_entropy - target_entropy < 0
            #   alpha_loss 对 log_alpha 的梯度为负
            #   optimizer.step() 后 log_alpha 增大
            #   alpha 上升，鼓励更高熵
            #
            # 当 entropy > target_entropy 时：
            #   actor_entropy - target_entropy > 0
            #   alpha 下降，减少熵奖励权重
            alpha_loss = self.log_alpha * (actor_entropy.detach() - target_entropy)

            alpha_loss.backward()
            self.alpha_optimizer.step()

            with torch.no_grad():
                self.log_alpha.data.clamp_(
                    min=np.log(self.alpha_min),
                    max=np.log(self.alpha_max),
                )

            self.alpha = float(self.log_alpha.exp().clamp(self.alpha_min, self.alpha_max).item())

        # ------------------------------------------------------------------
        # 5) Soft update target model
        # ------------------------------------------------------------------
        self._soft_update_target()

        # ------------------------------------------------------------------
        # 6) Save stats
        # ------------------------------------------------------------------
        self.last_critic_loss = float(critic_loss.item())
        self.last_actor_loss = float(actor_loss.item())
      
        self.last_q_target_mean = float(critic_info["q_target_mean"].item())
        self.last_q_target_abs_mean = float(critic_info["q_target_abs_mean"].item())
        self.last_q_target_max = float(critic_info["q_target_max"].item())
        self.last_q_target_min = float(critic_info["q_target_min"].item())

        self.last_q1_mean = float(critic_info["q1_mean"].item())
        self.last_q2_mean = float(critic_info["q2_mean"].item())
        self.last_q_gap = float(critic_info["q_gap"].item())

        self.last_entropy = float(actor_entropy.item())
        self.last_entropy_gap = abs(self.last_entropy - self.target_entropy)

        self.last_critic_grad_norm = self._to_float(critic_grad_norm)
        self.last_actor_grad_norm = self._to_float(actor_grad_norm)
        self.last_critic_grad_norm_post = self._to_float(critic_grad_norm_post)
        self.last_actor_grad_norm_post = self._to_float(actor_grad_norm_post)

        # ------------------------------------------------------------------
        # 7) Monitor report
        # ------------------------------------------------------------------
        self._report_monitor_if_needed(
            reward=reward,
            done=done,
            actor_info=actor_info,
        )

        self.train_step += 1

    # ======================================================================
    # Losses
    # ======================================================================
    def _compute_critic_loss(
        self,
        obs,      
        act,
        reward,
        done,
        next_obs,
        next_legal_action,
    ):
        """
        Critic loss:
            Q_target = r + gamma * (1 - done) *
                       E_a'[ Q_target(s',a') - alpha * log pi(a'|s') ]

        当前 Q：online critic
        next Q：target_model.critic
        next policy：online actor
        """
        q1, q2 = self._critic(obs)

        act = act.clamp(min=0, max=self.action_num - 1)

        q1_a = q1.gather(1, act)
        q2_a = q2.gather(1, act)

        with torch.no_grad():
            next_logits = self._actor(next_obs)
            next_prob = self._masked_softmax(next_logits, next_legal_action)
            next_log_prob = torch.log(next_prob.clamp_min(1e-8))

            next_q1, next_q2 = self._target_critic(next_obs)
            next_min_q = torch.min(next_q1, next_q2)

            next_v = (
                next_prob * (next_min_q - self.alpha * next_log_prob)
            ).sum(dim=1, keepdim=True)

            q_target = reward + (1.0 - done) * self.gamma * next_v

            if self.use_target_q_clip:
                q_target = q_target.clamp(
                    min=-self.target_q_clip,
                    max=self.target_q_clip,
                )

        if self.critic_use_huber:
            q1_loss = F.smooth_l1_loss(q1_a, q_target)
            q2_loss = F.smooth_l1_loss(q2_a, q_target)
        else:
            q1_loss = F.mse_loss(q1_a, q_target)
            q2_loss = F.mse_loss(q2_a, q_target)

        critic_loss = q1_loss + q2_loss

        critic_info = {
            "q_target_mean": q_target.mean(),
            "q_target_abs_mean": q_target.abs().mean(),
            "q_target_max": q_target.max(),
            "q_target_min": q_target.min(),
            "q1_mean": q1_a.mean(),
            "q2_mean": q2_a.mean(),
            "q_gap": torch.abs(q1_a - q2_a).mean(),
        }

        return critic_loss, critic_info

    def _compute_actor_loss(self, obs, legal_action):
        """
        Discrete SAC actor loss:
            J_pi = E_s [ sum_a pi(a|s) * ( alpha * log pi(a|s) - Q(s,a) ) ]

        这里使用全动作期望，不使用单动作采样。
        """
        logits = self._actor(obs)

        prob = self._masked_softmax(logits, legal_action)
        log_prob = torch.log(prob.clamp_min(1e-8))

        # Actor 更新时，Critic 只作为动作评价器，不更新 Critic 参数
        with torch.no_grad():
            q1, q2 = self._critic(obs)
            min_q = torch.min(q1, q2)

        actor_loss = (
            prob * (self.alpha * log_prob - min_q)
        ).sum(dim=1).mean()

        entropy = -(
            prob * log_prob
        ).sum(dim=1).mean()

        max_action_prob = prob.max(dim=1)[0].mean()

        actor_info = {
            "max_action_prob": max_action_prob.detach(),
        }

        return actor_loss, entropy, actor_info

    # ======================================================================
    # Mask / model wrappers
    # ======================================================================
    def _masked_softmax(self, logits, legal_action):
        """
        Masked softmax over legal actions.

        Args:
            logits: [B, A]
            legal_action: [B, A], 1 for legal, 0 for illegal

        Returns:
            prob: [B, A]
        """
        legal_action = legal_action.float()

        if legal_action.dim() == 1:
            legal_action = legal_action.view(-1, self.action_num)

        valid_count = legal_action.sum(dim=1, keepdim=True)
        no_valid_mask = valid_count <= 0

        safe_legal_action = legal_action.clone()
        if no_valid_mask.any():
            safe_legal_action[no_valid_mask.squeeze(1)] = 1.0

        masked_logits = logits.masked_fill(safe_legal_action <= 0, -1e9)

        # 减去 max，提升 softmax 数值稳定性
        masked_logits = masked_logits - masked_logits.max(dim=1, keepdim=True)[0]

        prob = torch.softmax(masked_logits, dim=1)

        # 非法动作概率强制归零后重新归一化，降低数值误差
        prob = prob * safe_legal_action
        prob_sum = prob.sum(dim=1, keepdim=True).clamp_min(1e-8)
        prob = prob / prob_sum

        return prob

    def _actor(self, obs):
        if hasattr(self.model, "actor"):
            return self.model.actor(obs)

        logits, _, _ = self.model(obs)
        return logits

    def _critic(self, obs):
        if hasattr(self.model, "critic"):
            return self.model.critic(obs)

        _, q1, q2 = self.model(obs)
        return q1, q2

    def _target_critic(self, obs):
        if hasattr(self.target_model, "critic"):
            return self.target_model.critic(obs)

        _, q1, q2 = self.target_model(obs)
        return q1, q2

    # ======================================================================
    # Update / utils
    # ======================================================================
    def _soft_update_target(self):
        """
        Soft update complete target model.
        """
        with torch.no_grad():
            for target_param, param in zip(
                self.target_model.parameters(),
                self.model.parameters(),
            ):
                target_param.data.copy_(
                    target_param.data * (1.0 - self.tau)
                    + param.data * self.tau
                )

    def _grad_total_norm(self, parameters):
        grads = []
        for p in parameters:
            if p.grad is not None:
                grads.append(p.grad.detach())

        if not grads:
            return torch.tensor(0.0, device=self.device)

        norms = [torch.norm(g, p=2) for g in grads]
        return torch.norm(torch.stack(norms), p=2)

    def _to_float(self, value):
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu().item())
        return float(value)

    def _get_actor_parameters(self):
        if hasattr(self.model, "actor_encoder") and hasattr(self.model, "actor_trunk") and hasattr(self.model, "actor_head"):
            return (
                list(self.model.actor_encoder.parameters())
                + list(self.model.actor_trunk.parameters())
                + list(self.model.actor_head.parameters())
            )

        if hasattr(self.model, "backbone") and hasattr(self.model, "actor_head"):
            return (
                list(self.model.backbone.parameters())
                + list(self.model.actor_head.parameters())
            )

        return list(self.model.parameters())

    def _get_critic_parameters(self):
        if (
            hasattr(self.model, "critic_encoder")
            and hasattr(self.model, "critic_trunk")
            and hasattr(self.model, "q1_head")
            and hasattr(self.model, "q2_head")
        ):
            return (
                list(self.model.critic_encoder.parameters())
                + list(self.model.critic_trunk.parameters())
                + list(self.model.q1_head.parameters())
                + list(self.model.q2_head.parameters())
            )

        if hasattr(self.model, "q1_head") and hasattr(self.model, "q2_head"):
            return (
                list(self.model.q1_head.parameters())
                + list(self.model.q2_head.parameters())
            )

        return list(self.model.parameters())

    # ======================================================================
    # Monitor
    # ======================================================================
    def _report_monitor_if_needed(
        self,
        reward,
        done,
        actor_info,
    ):
        now = time.time()
        if now - self.last_report_monitor_time < 60:
            return

        done_rate = done.float().mean().item()
        reward_mean = reward.mean().item()
        reward_abs_mean = reward.abs().mean().item()

        results = {
            # Critic / Actor losses
            "value_loss": round(self.last_critic_loss, 4),
            "policy_loss": round(self.last_actor_loss, 4),

            # Entropy / temperature
            "entropy": round(self.last_entropy, 4),
            "entropy_gap": round(self.last_entropy_gap, 4),
            "target_entropy": round(self.target_entropy, 4),
            "alpha": round(self.alpha, 4),

            # Reward scale
            "reward": round(reward_mean, 4),
            "reward_abs_mean": round(reward_abs_mean, 4),

            # Q diagnostics
            "q_target_mean": round(self.last_q_target_mean, 4),
            "q_target_abs_mean": round(self.last_q_target_abs_mean, 4),
            "q_target_max": round(self.last_q_target_max, 4),
            "q_target_min": round(self.last_q_target_min, 4),
            "q1_mean": round(self.last_q1_mean, 4),
            "q2_mean": round(self.last_q2_mean, 4),
            "q_gap": round(self.last_q_gap, 4),

            # Gradient diagnostics
            "critic_grad_norm": round(self.last_critic_grad_norm, 4),
            "critic_grad_norm_post": round(self.last_critic_grad_norm_post, 4),
            "actor_grad_norm": round(self.last_actor_grad_norm, 4),
            "actor_grad_norm_post": round(self.last_actor_grad_norm_post, 4),

            # Policy concentration
            "max_action_prob": round(self._to_float(actor_info["max_action_prob"]), 4),

            # Episode / training state
            "done_rate": round(done_rate, 4),
            "train_step": self.train_step,
        }

        # if self.logger:
            # self.logger.info(
            #     f"[SAC] train_step:{self.train_step} "
            #     f"value_loss:{results['value_loss']} "
            #     f"policy_loss:{results['policy_loss']} "
            #     f"entropy:{results['entropy']} "
            #     f"alpha:{results['alpha']} "
            #     f"q_target_mean:{results['q_target_mean']} "
            #     f"q_gap:{results['q_gap']} "
            #     f"critic_grad_norm:{results['critic_grad_norm']}"
            # )

        if self.monitor:
            self.monitor.put_data({os.getpid(): results})

        self.last_report_monitor_time = now
    def sync_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        for p in self.target_model.parameters():
            p.requires_grad = False
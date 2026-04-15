#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Enhanced PPO algorithm for Gorge Chase DIY Agent.
峡谷追猎 DIY Agent 增强版 PPO 算法。

改进要点：
  1. 优势函数归一化（NORMALIZE_ADVANTAGE）
  2. 熵系数线性退火（从 BETA_START 衰减到 BETA_END）
  3. 学习率线性调度（LR warm-decay）
  4. 保留 PPO Clip + Clipped Value Loss 标准实现
"""

import os
import time

import torch
from agent_diy.conf.conf import Config


class Algorithm:
    def __init__(self, model, optimizer, scheduler=None, device=None, logger=None, monitor=None):
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler  # 可选：外部传入 LR scheduler
        self.parameters = [p for pg in self.optimizer.param_groups for p in pg["params"]]
        self.logger = logger
        self.monitor = monitor

        self.label_size = Config.ACTION_NUM
        self.value_num = Config.VALUE_NUM
        self.clip_param = Config.CLIP_PARAM
        self.vf_coef = Config.VF_COEF

        # 熵系数退火
        self.var_beta = Config.BETA_START
        self.beta_start = Config.BETA_START
        self.beta_end = Config.BETA_END
        self.beta_decay_steps = Config.BETA_DECAY_STEPS

        self.normalize_advantage = Config.NORMALIZE_ADVANTAGE

        self.train_step = 0
        self.last_report_monitor_time = 0

    def _anneal_beta(self):
        """Linearly decay entropy coefficient. / 线性退火熵系数"""
        ratio = min(1.0, self.train_step / self.beta_decay_steps)
        self.var_beta = self.beta_start + ratio * (self.beta_end - self.beta_start)

    def learn(self, list_sample_data):
        """Training entry: enhanced PPO update on a batch of SampleData.

        训练入口：对一批 SampleData 执行增强版 PPO 更新。
        """
        obs = torch.stack([f.obs for f in list_sample_data]).to(self.device)
        legal_action = torch.stack([f.legal_action for f in list_sample_data]).to(self.device)
        act = torch.stack([f.act for f in list_sample_data]).to(self.device).view(-1, 1)
        old_prob = torch.stack([f.prob for f in list_sample_data]).to(self.device)
        reward = torch.stack([f.reward for f in list_sample_data]).to(self.device)
        advantage = torch.stack([f.advantage for f in list_sample_data]).to(self.device)
        old_value = torch.stack([f.value for f in list_sample_data]).to(self.device)
        reward_sum = torch.stack([f.reward_sum for f in list_sample_data]).to(self.device)

        # 优势函数归一化（降低方差，稳定训练）
        if self.normalize_advantage and advantage.numel() > 1:
            adv_mean = advantage.mean()
            adv_std = advantage.std().clamp(min=1e-8)
            advantage = (advantage - adv_mean) / adv_std

        self._anneal_beta()

        self.model.set_train_mode()
        self.optimizer.zero_grad()

        logits, value_pred = self.model(obs)

        total_loss, info_list = self._compute_loss(
            logits=logits,
            value_pred=value_pred,
            legal_action=legal_action,
            old_action=act,
            old_prob=old_prob,
            advantage=advantage,
            old_value=old_value,
            reward_sum=reward_sum,
            reward=reward,
        )

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters, Config.GRAD_CLIP_RANGE)
        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        self.train_step += 1

        now = time.time()
        if now - self.last_report_monitor_time >= 60:
            results = {
                "total_loss": round(total_loss.item(), 4),
                "value_loss": round(info_list[0].item(), 4),
                "policy_loss": round(info_list[1].item(), 4),
                "entropy_loss": round(info_list[2].item(), 4),
                "beta": round(self.var_beta, 6),
                "reward": round(reward.mean().item(), 4),
            }
            if self.logger:
                self.logger.info(
                    f"[train] total_loss:{results['total_loss']} "
                    f"policy_loss:{results['policy_loss']} "
                    f"value_loss:{results['value_loss']} "
                    f"entropy:{results['entropy_loss']} "
                    f"beta:{results['beta']}"
                )
            if self.monitor:
                self.monitor.put_data({os.getpid(): results})
            self.last_report_monitor_time = now

    def _compute_loss(
        self,
        logits,
        value_pred,
        legal_action,
        old_action,
        old_prob,
        advantage,
        old_value,
        reward_sum,
        reward,
    ):
        """Compute PPO loss: policy + clipped value + entropy.

        计算 PPO 损失：策略损失 + 裁剪价值损失 + 熵正则化。
        """
        # 合法动作掩码 softmax
        prob_dist = self._masked_softmax(logits, legal_action)

        # 策略损失（PPO Clip）
        one_hot = torch.nn.functional.one_hot(
            old_action[:, 0].long(), self.label_size
        ).float()
        new_prob = (one_hot * prob_dist).sum(1, keepdim=True)
        old_action_prob = (one_hot * old_prob).sum(1, keepdim=True).clamp(1e-9)
        ratio = new_prob / old_action_prob

        adv = advantage.view(-1, 1)
        policy_loss1 = -ratio * adv
        policy_loss2 = -ratio.clamp(1 - self.clip_param, 1 + self.clip_param) * adv
        policy_loss = torch.maximum(policy_loss1, policy_loss2).mean()

        # 价值损失（Clipped Value Loss）
        vp = value_pred
        ov = old_value
        tdret = reward_sum
        value_clip = ov + (vp - ov).clamp(-self.clip_param, self.clip_param)
        value_loss = (
            0.5
            * torch.maximum(
                torch.square(tdret - vp),
                torch.square(tdret - value_clip),
            ).mean()
        )

        # 熵损失（鼓励探索）
        entropy_loss = (-prob_dist * torch.log(prob_dist.clamp(1e-9, 1))).sum(1).mean()

        # 总损失
        total_loss = self.vf_coef * value_loss + policy_loss - self.var_beta * entropy_loss

        return total_loss, [value_loss, policy_loss, entropy_loss]

    def _masked_softmax(self, logits, legal_action):
        """Softmax with legal action masking.

        合法动作掩码下的 softmax。
        """
        label_max, _ = torch.max(logits * legal_action, dim=1, keepdim=True)
        label = logits - label_max
        label = label * legal_action
        label = label + 1e5 * (legal_action - 1)
        return torch.nn.functional.softmax(label, dim=1)
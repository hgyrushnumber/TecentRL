#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Monitor panel configuration builder for Gorge Chase.
峡谷追猎监控面板配置构建器。
"""

from kaiwudrl.common.monitor.monitor_config_builder import MonitorConfigBuilder


def build_monitor():
    """
    Create monitoring panel configurations for custom indicators.
    """
    monitor = MonitorConfigBuilder()

    config_dict = (
        monitor.title("峡谷追猎")
        .add_group(
            group_name="算法指标",
            group_name_en="algorithm",
        )

        # ------------------------------------------------------------------
        # Reward
        # ------------------------------------------------------------------
        .add_panel(
            name="奖励",
            name_en="reward",
            type="line",
        )
        .add_metric(
            metrics_name="reward",
            expr="avg(reward{})",
        )
        .end_panel()

        # ------------------------------------------------------------------
        # Loss
        # ------------------------------------------------------------------
        .add_panel(
            name="价值损失",
            name_en="value_loss",
            type="line",
        )
        .add_metric(
            metrics_name="value_loss",
            expr="avg(value_loss{})",
        )
        .end_panel()
        .add_panel(
            name="策略损失",
            name_en="policy_loss",
            type="line",
        )
        .add_metric(
            metrics_name="policy_loss",
            expr="avg(policy_loss{})",
        )
        .end_panel()

        # ------------------------------------------------------------------
        # Entropy / Alpha
        # ------------------------------------------------------------------
        .add_panel(
            name="策略熵",
            name_en="entropy",
            type="line",
        )
        .add_metric(
            metrics_name="entropy",
            expr="avg(entropy{})",
        )
        .end_panel()
        .add_panel(
            name="熵偏差",
            name_en="entropy_gap",
            type="line",
        )
        .add_metric(
            metrics_name="entropy_gap",
            expr="avg(entropy_gap{})",
        )
        .end_panel()
        .add_panel(
            name="温度系数",
            name_en="alpha",
            type="line",
        )
        .add_metric(
            metrics_name="alpha",
            expr="avg(alpha{})",
        )
        .end_panel()

        # ------------------------------------------------------------------
        # Q diagnostics
        # ------------------------------------------------------------------
        .add_panel(
            name="Q目标均值",
            name_en="q_target_mean",
            type="line",
        )
        .add_metric(
            metrics_name="q_target_mean",
            expr="avg(q_target_mean{})",
        )
        .end_panel()
        .add_panel(
            name="双Q差值",
            name_en="q_gap",
            type="line",
        )
        .add_metric(
            metrics_name="q_gap",
            expr="avg(q_gap{})",
        )
        .end_panel()

        # ------------------------------------------------------------------
        # Gradient diagnostics
        # ------------------------------------------------------------------
        .add_panel(
            name="Critic梯度范数",
            name_en="critic_grad_norm",
            type="line",
        )
        .add_metric(
            metrics_name="critic_grad_norm",
            expr="avg(critic_grad_norm{})",
        )
        .end_panel()
        .add_panel(
            name="Critic梯度范数裁剪后",
            name_en="critic_grad_norm_post",
            type="line",
        )
        .add_metric(
            metrics_name="critic_grad_norm_post",
            expr="avg(critic_grad_norm_post{})",
        )
        .end_panel()
        .add_panel(
            name="Actor梯度范数",
            name_en="actor_grad_norm",
            type="line",
        )
        .add_metric(
            metrics_name="actor_grad_norm",
            expr="avg(actor_grad_norm{})",
        )
        .end_panel()
        .add_panel(
            name="Actor梯度范数裁剪后",
            name_en="actor_grad_norm_post",
            type="line",
        )
        .add_metric(
            metrics_name="actor_grad_norm_post",
            expr="avg(actor_grad_norm_post{})",
        )
        .end_panel()

        # ------------------------------------------------------------------
        # Reward components
        # ------------------------------------------------------------------
        .add_panel(
            name="危险惩罚分项",
            name_en="comp_danger_penalty",
            type="line",
        )
        .add_metric(
            metrics_name="comp_danger_penalty",
            expr="avg(comp_danger_penalty{})",
        )
        .end_panel()
        .add_panel(
            name="宝箱得分分项",
            name_en="comp_treasure_score",
            type="line",
        )
        .add_metric(
            metrics_name="comp_treasure_score",
            expr="avg(comp_treasure_score{})",
        )
        .end_panel()
        .add_panel(
            name="宝箱接近分项",
            name_en="comp_treasure_approach",
            type="line",
        )
        .add_metric(
            metrics_name="comp_treasure_approach",
            expr="avg(comp_treasure_approach{})",
        )
        .end_panel()
        .add_panel(
            name="距离塑形分项",
            name_en="comp_dist_shaping",
            type="line",
        )
        .add_metric(
            metrics_name="comp_dist_shaping",
            expr="avg(comp_dist_shaping{})",
        )
        .end_panel()

        .end_group()
        .build()
    )

    return config_dict
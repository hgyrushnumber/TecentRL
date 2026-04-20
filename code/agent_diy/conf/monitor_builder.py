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
    This function is used to create monitoring panel configurations for custom indicators.
    该函数用于创建自定义指标的监控面板配置。
    """
    monitor = MonitorConfigBuilder()

    config_dict = (
        monitor.title("峡谷追猎")
        .add_group(
            group_name="算法指标",
            group_name_en="algorithm",
        )

        # 回报
        .add_panel(
            name="累积回报",
            name_en="reward",
            type="line",
        )
        .add_metric(
            metrics_name="reward",
            expr="avg(reward{})",
        )
        .end_panel()

        # 总损失
        .add_panel(
            name="总损失",
            name_en="total_loss",
            type="line",
        )
        .add_metric(
            metrics_name="total_loss",
            expr="avg(total_loss{})",
        )
        .end_panel()

        # Critic 损失
        .add_panel(
            name="Critic损失",
            name_en="critic_loss",
            type="line",
        )
        .add_metric(
            metrics_name="critic_loss",
            expr="avg(critic_loss{})",
        )
        .end_panel()

        # Actor 损失
        .add_panel(
            name="Actor损失",
            name_en="actor_loss",
            type="line",
        )
        .add_metric(
            metrics_name="actor_loss",
            expr="avg(actor_loss{})",
        )
        .end_panel()

        # 熵
        .add_panel(
            name="策略熵",
            name_en="entropy",
            type="line",
        )
        .add_metric(
            metrics_name="entropy",
            expr="avg(entropy{})",
        )
        .add_metric(
            metrics_name="target_entropy",
            expr="avg(target_entropy{})",
        )
        .end_panel()

        # Alpha 损失
        .add_panel(
            name="Alpha损失",
            name_en="alpha_loss",
            type="line",
        )
        .add_metric(
            metrics_name="alpha_loss",
            expr="avg(alpha_loss{})",
        )
        .end_panel()

        # Alpha
        .add_panel(
            name="Alpha值",
            name_en="alpha",
            type="line",
        )
        .add_metric(
            metrics_name="alpha",
            expr="avg(alpha{})",
        )
        .end_panel()

        # TD误差
        .add_panel(
            name="TD绝对误差均值",
            name_en="td_abs_mean",
            type="line",
        )
        .add_metric(
            metrics_name="td_abs_mean",
            expr="avg(td_abs_mean{})",
        )
        .end_panel()

        # Buffer 大小
        .add_panel(
            name="回放池大小",
            name_en="buffer_size",
            type="line",
        )
        .add_metric(
            metrics_name="buffer_size",
            expr="avg(buffer_size{})",
        )
        .end_panel()

        # PER beta
        .add_panel(
            name="PER Beta",
            name_en="beta",
            type="line",
        )
        .add_metric(
            metrics_name="beta",
            expr="avg(beta{})",
        )
        .end_panel()

        # 训练步数
        .add_panel(
            name="训练步数",
            name_en="train_step",
            type="line",
        )
        .add_metric(
            metrics_name="train_step",
            expr="max(train_step{})",
        )
        .end_panel()

        .end_group()
        .build()
    )
    return config_dict
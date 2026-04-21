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
    # This function is used to create monitoring panel configurations for custom indicators.
    # 该函数用于创建自定义指标的监控面板配置。
    """
    monitor = MonitorConfigBuilder()

    config_dict = (
        monitor.title("峡谷追猎")
        .add_group(
            group_name="算法指标",
            group_name_en="algorithm",
        )
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
        .add_panel(
            name="熵损失",
            name_en="entropy_loss",
            type="line",
        )
        .add_metric(
            metrics_name="entropy_loss",
            expr="avg(entropy_loss{})",
        )
        .end_panel()
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
        .add_panel(
            name="梯度范数",
            name_en="grad_norm",
            type="line",
        )
        .add_metric(
            metrics_name="grad_norm",
            expr="avg(grad_norm{})",
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
        .add_panel(
            name="温度损失",
            name_en="alpha_loss",
            type="line",
        )
        .add_metric(
            metrics_name="alpha_loss",
            expr="avg(alpha_loss{})",
        )
        .end_panel()
        .add_panel(
            name="合法动作数",
            name_en="legal_action_count",
            type="line",
        )
        .add_metric(
            metrics_name="legal_action_count",
            expr="avg(legal_action_count{})",
        )
        .end_panel()
        .add_panel(
            name="终局奖励",
            name_en="final_reward",
            type="line",
        )
        .add_metric(
            metrics_name="final_reward",
            expr="avg(final_reward{})",
        )
        .end_panel()
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
        .end_group()
        .build()
    )
    return config_dict

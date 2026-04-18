#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Data definitions and ReplayBuffer for Gorge Chase SAC.
峡谷追猎 SAC 数据类定义与经验回放池。

分布式架构说明：
  - SampleData 由 Actor 进程构造，经框架序列化发送给 Learner
  - ReplayBuffer 由 Algorithm（Learner侧）内置维护
  - workflow 只负责收集数据并 yield，不做任何训练
"""

import numpy as np
from collections import deque
from common_python.utils.common_func import create_cls
from agent_diy.conf.conf import Config


# ObsData / ActData 保持不变
ObsData = create_cls("ObsData", feature=None, legal_action=None)
ActData = create_cls("ActData", action=None, d_action=None, prob=None, value=None)

# SampleData：在标准字段基础上增加 SAC 所需的 next_obs / next_legal_action
# 框架通过字段维度做序列化，int 表示向量维度，None 表示标量
SampleData = create_cls(
    "SampleData",
    obs=Config.DIM_OF_OBSERVATION,        # 当前状态
    legal_action=Config.ACTION_NUM,        # 当前合法动作掩码
    act=1,                                 # 执行的动作
    reward=Config.VALUE_NUM,               # 即时奖励
    done=1,                                # 终止标志
    next_obs=Config.DIM_OF_OBSERVATION,    # 下一状态（SAC 所需）
    next_legal_action=Config.ACTION_NUM,   # 下一状态合法掩码（SAC 所需）
    # 以下字段框架/PPO会用到，SAC置0保持兼容
    reward_sum=Config.VALUE_NUM,
    value=Config.VALUE_NUM,
    next_value=Config.VALUE_NUM,
    advantage=Config.VALUE_NUM,
    prob=Config.ACTION_NUM,
)


class ReplayBuffer:
    """Circular replay buffer for SAC Learner side.

    SAC 经验回放池，由 Algorithm（Learner侧）内置维护。
    Actor 每局产生的 SampleData 列表通过 learn() 批量推入。
    """

    def __init__(self, capacity):
        self._buf = deque(maxlen=capacity)

    def push_batch(self, transitions):
        """Push a list of SampleData from one episode."""
        for t in transitions:
            self._buf.append(t)

    def sample(self, batch_size):
        """Randomly sample a batch."""
        idx = np.random.randint(0, len(self._buf), size=batch_size)
        return [self._buf[i] for i in idx]

    def __len__(self):
        return len(self._buf)


class PrioritizedReplayBuffer:
    """优先经验回放池（Prioritized Experience Replay, PER）。

    基于TD误差的优先级采样，提高高价值样本的训练效率。
    使用SumTree数据结构实现O(log n)采样复杂度。

    关键特性：
      - 优先级 p_i = |TD_error| + ε，ε为小常数防止零优先级
      - 采样概率 P(i) = p_i^α / Σ p_j^α，α控制优先级程度
      - 重要性采样权重 w_i = (1/N * 1/P(i))^β，β逐渐增至1消除偏差
    """

    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000, epsilon=1e-6):
        """
        Args:
            capacity: 缓冲区容量（必须为2的幂次，用于SumTree）
            alpha: 优先级指数（0=均匀采样，1=完全优先级采样）
            beta_start: IS权重初始值（逐渐增至1）
            beta_frames: β增长至1的帧数
            epsilon: 优先级下限，防止零优先级
        """
        # 确保容量为2的幂次
        self.capacity = 1
        while self.capacity < capacity:
            self.capacity *= 2

        self.alpha = alpha
        self.beta = beta_start
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.epsilon = epsilon

        # SumTree存储优先级和累积和
        self.tree = np.zeros(2 * self.capacity - 1, dtype=np.float64)
        # 数据存储（循环缓冲）
        self.data = [None] * self.capacity
        self.size = 0
        self.write_idx = 0

        # 当前最大优先级（新样本初始优先级）
        self.max_priority = 1.0

        # 帧计数器（用于β增长）
        self.frame_count = 0

    def _propagate(self, tree_idx, change):
        """向上传播优先级变化到根节点。"""
        parent = (tree_idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, tree_idx, value):
        """从根节点向下检索，找到value对应的叶子节点。"""
        left = 2 * tree_idx + 1
        right = left + 1

        if left >= len(self.tree):
            return tree_idx

        if value <= self.tree[left]:
            return self._retrieve(left, value)
        else:
            return self._retrieve(right, value - self.tree[left])

    def push_batch(self, transitions):
        """批量推入样本（新样本使用当前最大优先级）。"""
        for t in transitions:
            # 叶子节点索引
            tree_idx = self.write_idx + self.capacity - 1

            # 存储数据
            self.data[self.write_idx] = t

            # 设置优先级（新样本使用最大优先级，保证高TD误差样本被优先采样）
            priority = self.max_priority ** self.alpha
            self.tree[tree_idx] = priority
            self._propagate(tree_idx, priority - self.tree[tree_idx])

            # 更新索引和大小
            self.write_idx = (self.write_idx + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """优先级采样，返回(样本, 索引, IS权重)。"""
        if self.size == 0:
            return [], [], np.array([])

        # 更新β（逐渐增至1，消除重要性采样偏差）
        self.frame_count += 1
        self.beta = min(1.0, self.beta_start + (1.0 - self.beta_start) * 
                        (self.frame_count / self.beta_frames))

        # 采样索引和IS权重
        indices = []
        priorities = []
        total_priority = self.tree[0]
        
        # 边界保护：如果总优先级为0，退化为均匀采样
        if total_priority <= 0:
            total_priority = 1.0

        # 分段采样（保证均匀覆盖优先级区间）
        segment = total_priority / batch_size
        for i in range(batch_size):
            low = i * segment
            high = (i + 1) * segment
            value = np.random.uniform(low, high)

            # 检索叶子节点
            tree_idx = self._retrieve(0, value)
            data_idx = tree_idx - self.capacity + 1

            # 边界检查：确保索引有效
            if data_idx < 0 or data_idx >= self.size or self.data[data_idx] is None:
                # 回退到随机采样
                if self.size > 0:
                    data_idx = np.random.randint(0, self.size)
                    tree_idx = data_idx + self.capacity - 1
                else:
                    continue

            indices.append(tree_idx)
            # 安全访问tree数组
            if tree_idx < len(self.tree):
                priorities.append(self.tree[tree_idx])
            else:
                priorities.append(1.0)

        # 如果没有成功采样任何样本，返回空
        if len(indices) == 0:
            return [], [], np.array([])

        # 计算重要性采样权重 w_i = (N * P(i))^(-β)
        priorities = np.array(priorities, dtype=np.float64)
        # 防止除零
        sample_probs = priorities / max(total_priority, 1e-8)
        # 防止数值问题
        is_weights = np.power(np.maximum(self.size * sample_probs, 1e-8), -self.beta)
        # 归一化权重（最大权重为1）
        if len(is_weights) > 0 and is_weights.max() > 0:
            is_weights = is_weights / is_weights.max()
        else:
            is_weights = np.ones_like(is_weights)

        # 提取样本
        samples = []
        valid_indices = []
        valid_weights = []
        for i, tree_idx in enumerate(indices):
            data_idx = tree_idx - self.capacity + 1
            if 0 <= data_idx < len(self.data) and self.data[data_idx] is not None:
                samples.append(self.data[data_idx])
                valid_indices.append(tree_idx)
                valid_weights.append(is_weights[i])

        # 如果样本数不足，补充随机样本
        while len(samples) < batch_size and self.size > 0:
            idx = np.random.randint(0, self.size)
            if self.data[idx] is not None:
                samples.append(self.data[idx])
                valid_indices.append(idx + self.capacity - 1)
                valid_weights.append(1.0)

        return samples, valid_indices, np.array(valid_weights, dtype=np.float32)

    def update_priorities(self, tree_indices, td_errors):
        """根据TD误差更新样本优先级。"""
        if len(tree_indices) == 0 or len(td_errors) == 0:
            return
            
        td_errors = np.abs(td_errors) + self.epsilon
        new_priorities = np.power(td_errors, self.alpha)

        for tree_idx, priority in zip(tree_indices, new_priorities):
            # 边界检查：确保tree_idx有效
            if 0 <= tree_idx < len(self.tree):
                change = priority - self.tree[tree_idx]
                self.tree[tree_idx] = priority
                self._propagate(tree_idx, change)
                # 更新最大优先级
                self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return self.size


def sample_process(list_sample_data):
    """SAC 不需要 GAE，直接返回原列表（兼容框架调用）。"""
    return list_sample_data
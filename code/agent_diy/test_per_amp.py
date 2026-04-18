#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
测试优先经验回放（PER）和混合精度训练（AMP）功能
"""

import numpy as np
import torch
from agent_diy.feature.definition import SampleData, PrioritizedReplayBuffer
from agent_diy.conf.conf import Config

def test_prioritized_replay_buffer():
    """测试PER缓冲区的基本功能"""
    print("=" * 60)
    print("测试优先经验回放缓冲区（PER）")
    print("=" * 60)
    
    # 创建PER缓冲区
    buffer = PrioritizedReplayBuffer(
        capacity=1000,
        alpha=0.6,
        beta_start=0.4,
        beta_frames=10000
    )
    
    # 创建模拟样本
    print("\n1. 推入100个模拟样本...")
    for i in range(100):
        sample = SampleData(
            obs=np.random.randn(Config.DIM_OF_OBSERVATION).astype(np.float32),
            legal_action=np.random.rand(Config.ACTION_NUM).astype(np.float32),
            act=np.array([i % 10], dtype=np.float32),
            reward=np.array([np.random.rand()], dtype=np.float32),
            done=np.array([0.0], dtype=np.float32),
            next_obs=np.random.randn(Config.DIM_OF_OBSERVATION).astype(np.float32),
            next_legal_action=np.random.rand(Config.ACTION_NUM).astype(np.float32),
            reward_sum=np.zeros(1, dtype=np.float32),
            value=np.zeros(1, dtype=np.float32),
            next_value=np.zeros(1, dtype=np.float32),
            advantage=np.zeros(1, dtype=np.float32),
            prob=np.random.rand(Config.ACTION_NUM).astype(np.float32),
        )
        buffer.push_batch([sample])
    
    print(f"   缓冲区大小: {len(buffer)}")
    assert len(buffer) == 100, "缓冲区大小应为100"
    print("   ✓ 推入成功")
    
    # 测试采样
    print("\n2. 测试优先级采样...")
    samples, indices, is_weights = buffer.sample(32)
    print(f"   采样数量: {len(samples)}")
    print(f"   索引数量: {len(indices)}")
    print(f"   IS权重数量: {len(is_weights)}")
    print(f"   IS权重范围: [{is_weights.min():.4f}, {is_weights.max():.4f}]")
    print(f"   当前beta值: {buffer.beta:.4f}")
    assert len(samples) == 32, "采样数量应为32"
    assert len(indices) == 32, "索引数量应为32"
    assert len(is_weights) == 32, "IS权重数量应为32"
    print("   ✓ 采样成功")
    
    # 测试优先级更新
    print("\n3. 测试优先级更新...")
    td_errors = np.random.rand(32) * 2  # 模拟TD误差
    buffer.update_priorities(indices, td_errors)
    print(f"   更新了{len(indices)}个样本的优先级")
    print(f"   最大优先级: {buffer.max_priority:.4f}")
    print("   ✓ 优先级更新成功")
    
    # 测试β增长
    print("\n4. 测试β自动增长...")
    initial_beta = buffer.beta
    for _ in range(100):
        buffer.sample(32)
    final_beta = buffer.beta
    print(f"   初始beta: {initial_beta:.4f}")
    print(f"   采样100次后beta: {final_beta:.4f}")
    assert final_beta > initial_beta, "beta应该增长"
    print("   ✓ beta自动增长成功")
    
    print("\n" + "=" * 60)
    print("✓ PER缓冲区测试通过！")
    print("=" * 60)


def test_amp():
    """测试混合精度训练"""
    print("\n" + "=" * 60)
    print("测试混合精度训练（AMP）")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    
    if device.type == "cpu":
        print("警告: CPU环境下AMP无效，但仍可运行")
    
    # 创建简单的模型
    model = torch.nn.Linear(10, 5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler(enabled=Config.USE_AMP)
    
    # 模拟训练步骤
    print("\n1. 测试混合精度前向传播...")
    for i in range(5):
        x = torch.randn(32, 10).to(device)
        y = torch.randn(32, 5).to(device)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=Config.USE_AMP):
            output = model(x)
            loss = torch.nn.functional.mse_loss(output, y)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        if i == 0:
            print(f"   损失值: {loss.item():.4f}")
            print(f"   梯度缩放器状态: scale={scaler.get_scale():.1f}")
    
    print("   ✓ 混合精度训练成功")
    
    print("\n" + "=" * 60)
    print("✓ AMP测试通过！")
    print("=" * 60)


if __name__ == "__main__":
    try:
        test_prioritized_replay_buffer()
        test_amp()
        print("\n" + "=" * 60)
        print("✓✓✓ 所有测试通过！PER和AMP功能正常 ✓✓✓")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
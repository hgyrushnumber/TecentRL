# 峡谷追猎 SAC 算法实现

## 项目简介

本项目实现了基于 SAC（Soft Actor-Critic）算法的峡谷追猎游戏 AI 智能体。智能体需要在充满怪物的峡谷中躲避追击并尽可能收集宝箱。

## 算法架构

### 核心组件

1. **模型架构** (`model/model.py`)
   - 共享 MLP 骨干网络 + 残差连接 + LayerNorm
   - Actor：输出合法动作上的概率分布 π(a|s)
   - Critic：双 Q 网络，同时输出所有动作 Q 值（取 min 减少过估计）
   - 输入维度：102 维特征向量
   - 隐藏层：512 → 256（含残差连接）
   - 动作空间：10（8 方向移动 + 闪现 + 天赋技能）

2. **算法实现** (`algorithm/algorithm.py`)
   - 离散动作 SAC，内置优先经验回放（PER）
   - 自动熵调整（Auto-α）平衡探索与利用
   - 混合精度训练（AMP）加速
   - Learner 侧内置 ReplayBuffer，与 Actor 解耦

3. **特征工程** (`feature/preprocessor.py`)
   - 英雄特征（6D）：位置、技能冷却、buff 状态
   - 怪物特征（11D × 2）：位置、速度、距离、相对速度、预测距离、碰撞风险、方向角
   - 宝箱特征（6D）：位置、距离、数量、方向向量
   - Buff 特征（5D）：位置、距离、方向向量
   - 局部地图特征（49D）：7×7 障碍网格
   - 合法动作掩码（10D）
   - 进度特征（4D）：步数、最近怪物距离、碰撞风险、后半段标志

4. **奖励设计** (`feature/preprocessor.py`)
   - 宝箱收集：每个 +50（主要信号）
   - 危险惩罚：距离 < 0.25 时指数惩罚，最高 -5.0
   - 方向引导：向宝箱靠近时 +0.4×delta（早期探索引导）
   - 存活奖励：每步 +0.01（鼓励持续探索）
   - 终局奖励：被抓 -50 / 存活至终 +20 + 宝箱数×10

---

## SAC 核心损失概念

### 1. Critic 损失（value_loss）

**含义**：衡量 Q 网络预测值与贝尔曼目标值之间的误差。

```
critic_loss = MSE(Q(s,a), r + γ * V(s'))
其中 V(s') = Σ_a π(a|s') * (min(Q1,Q2)(s',a) - α * logπ(a|s'))
```

Q 网络在学习「在状态 s 下执行动作 a，未来能获得多少累积奖励」。损失越小说明价值估计越准确。

**健康信号**：训练初期较大，随训练进行应稳定下降。持续震荡说明学习率过高。

---

### 2. Actor 损失（policy_loss）

**含义**：衡量策略的优化方向，让 Actor 倾向于选择 Q 值高且熵也高的动作。

```
actor_loss = E_π[α * logπ(a|s) - min(Q1,Q2)(s,a)]
```

- `- min(Q1,Q2)` 部分：推动策略选择高价值动作
- `α * logπ` 部分：同时保持策略多样性（熵正则化）
- α 控制两者的平衡点

**健康信号**：通常为负值（Q 值大于熵项），随训练缓慢下降。若突然变大说明 Critic 还未收敛。

---

### 3. 熵损失（alpha_loss / entropy_loss）

**含义**：自动调节温度系数 α 的损失，用于维持策略熵在目标水平。

```
alpha_loss = α * (H[π] - H_target)
其中 H_target = TARGET_ENTROPY_RATIO * ln(动作数)
```

- 当前策略熵 H[π] > 目标熵 H_target → alpha_loss > 0 → α 减小 → 减少探索
- 当前策略熵 H[π] < 目标熵 H_target → alpha_loss < 0 → α 增大 → 增加探索

**通俗理解**：α 是自动调节的「探索开关」，训练初期策略随机（熵高），α 自动收小；策略过度收敛（熵低），α 自动调大强制探索。

**健康信号**：在 0 附近波动为正常。持续偏正说明策略太随机尚未收敛。

---

### 4. 熵（Entropy）

```
H[π] = -Σ_a π(a|s) * logπ(a|s)
```

| 熵的值 | 含义 |
|--------|------|
| 高（接近 ln(10) ≈ 2.3） | 策略接近均匀随机，充分探索 |
| 低（接近 0） | 策略高度确定，接近贪心 |

当前配置 `TARGET_ENTROPY_RATIO = 0.7`，目标熵约为 `0.7 × ln(10) ≈ 1.61`。

---

### 5. 训练健康信号总览

| 指标 | 正常状态 | 异常信号及原因 |
|------|----------|----------------|
| value_loss | 逐步下降并收敛 | 持续震荡/不降 → 学习率过高 |
| policy_loss | 负值，缓慢下降 | 突然变大 → Critic 未收敛 |
| alpha_loss | 在 0 附近波动 | 持续偏正 → 策略过于随机 |
| alpha | 0.1 ~ 0.5 | 接近 1.0 → 策略太随机未收敛 |
| buffer_size | 持续增长至满 | 增长停滞 → Actor 采样异常 |

---

## 断点续训机制分析

### 结论：当前实现已支持利用上一轮模型

#### Actor 侧（每局开始时）

在 `workflow/train_workflow.py` 的 `run_episodes()` 中，**每局开始都会拉取最新模型**：

```python
# train_workflow.py, EpisodeRunner.run_episodes()
self.agent.reset(env_obs)
self.agent.load_model(id="latest")   # ← 每局拉取一次最新 checkpoint
```

Actor 始终使用 Learner 最新训练好的策略进行采样，无需重启即可利用上一轮权重。

#### Learner 侧（进程启动时）

在 `agent.py` 的 `__init__` 末尾调用 `_try_resume()`，自动加载上次的完整训练状态：

```python
# agent.py, Agent.__init__()
self._try_resume()   # ← Learner 启动时自动恢复

# _try_resume() 加载内容：
# - Actor 网络权重
# - Critic / CriticTarget 网络权重
# - Actor/Critic/Alpha 三个优化器状态
# - log_alpha（温度系数）
# - train_step（训练步数计数）
```

#### 模型保存策略（Workflow 侧）

```
每5分钟 → 保存带版本号 checkpoint（model.ckpt-{n}.pkl）
        → 同时覆盖写 latest（model.ckpt-latest.pkl）
        → 清理超过5个的旧版本
```

#### 注意事项

- 修改网络维度（如 `hidden_dim`、`mid_dim`）后，旧 checkpoint 与新结构不兼容，需**清空旧模型从头训练**
- `model_file_path` 由框架注入到 `BaseAgent`，若框架未注入则 `_try_resume` 静默跳过
- Actor 侧仅加载 Actor 权重；Learner 侧同时加载 Critic、优化器等完整状态

---

## 配置参数（conf/conf.py）

| 参数 | 当前值 | 说明 |
|------|--------|------|
| GAMMA | 0.99 | 折扣因子 |
| INIT_LEARNING_RATE_START | 3e-4 | 初始学习率 |
| INIT_LEARNING_RATE_END | 1e-5 | 学习率衰减终点 |
| LR_DECAY_STEPS | 500,000 | 学习率衰减步数 |
| TARGET_ENTROPY_RATIO | 0.7 | 目标熵比例（×ln(10)） |
| TAU | 0.005 | 目标网络软更新系数 |
| REPLAY_BUFFER_SIZE | 50,000 | 经验回放池容量 |
| BATCH_SIZE | 512 | 训练批大小 |
| LEARNING_STARTS | 2,000 | 开始训练的预热步数 |
| PER_ALPHA | 0.6 | PER 优先级指数 |
| PER_BETA_START | 0.4 | PER IS 权重初始值 |

---

## 许可证

Copyright © 1998 - 2026 Tencent. All Rights Reserved.
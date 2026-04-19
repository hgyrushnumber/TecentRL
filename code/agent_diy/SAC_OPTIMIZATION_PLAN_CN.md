# SAC 训练“收敛快但效果差”优化方案（峡谷追猎）

> 目标：把当前策略从“早收敛到次优生存策略”升级为“稳健生存 + 主动拿宝箱 + 强泛化”。

## 1) 先定位：为什么会“收敛很快但分数不好”

结合当前实现，核心问题通常不是“学不会”，而是“太快学会了错误目标”。

### 1.1 动作空间不完整（只做了 10 动作，缺少 8 向闪现）
- 当前 `ACTION_NUM = 10`，是 `8移动 + 1闪现 + 1天赋`，并非环境原生 `16` 动作。  
- 这会导致策略无法学习“方向性闪现逃生/抢宝箱”，上限被硬性限制。

### 1.2 奖励存在“局部最优吸引子”
- 当前每步生存奖励较强（缩放后每步 +0.1），1000 步累计 +100。  
- 宝箱奖励是每个 +50（缩放后），但拿宝箱需要承担接近怪物的风险。  
- SAC 容易学成“保守绕圈活久一点”，而不是“机会主义地拿箱子”。

### 1.3 离策略更新强度偏大，容易把早期偏差策略快速固化
- 每回合按 `n_updates ≈ episode_len` 做更新，且上限 32。  
- 在地图/出生分布还没充分覆盖时，Q 函数会对少量模式过拟合，策略迅速坍缩到次优。

### 1.4 探索目标偏低，策略过早变“确定性”
- `target_entropy = 0.5 * log(|A|)`，对离散动作偏保守。  
- 对“躲避+规划+时机技能”这种组合决策任务，前期需要更高熵避免早收敛。

### 1.5 特征中仍有可改进点
- 未显式提供“怪物到达时间（ETA）/闭塞度/可逃逸通道数”等关键风险先验。  
- 局部地图 7×7 足够做短时避障，但还不足以支撑中程路径规划（抢箱前置决策）。

---

## 2) 优先级最高的改动（建议按顺序落地）

## P0：补全动作空间到 16（必须做）
1. Actor/Critic 输出改为 16 维。  
2. 合法动作掩码直接使用环境 `legal_act[0:16]`。  
3. 若当前框架仍需要“技能聚合动作”，改为“方向化闪现动作”。

**预期收益**：显著提升极限逃生与远距抢箱能力，是最大单点增益。

## P0：重构奖励为“阶段化 + 事件驱动”
推荐结构（先给可执行系数）：

```text
r = w_survive(t) + w_treasure * I(collect)
    + w_risk * Δsafety
    + w_flash * I(flash_good) - w_bad_flash * I(flash_waste)
    - w_dead * I(done_by_monster)
```

建议初始值：
- `w_survive(t)`: 早期 0.01，中后期 0.02（不要过大）
- `w_treasure`: 2.0 ~ 4.0（按 reward 归一化后调）
- `w_risk`: 0.05（用势函数差分，见下）
- `w_flash`: +0.3（仅当“闪现后最小怪物距离显著增加”）
- `w_bad_flash`: 0.2（冷却中无收益、或原地闪现）
- `w_dead`: 2.0 ~ 5.0（终局强惩罚）

### 势函数塑形（避免 reward hacking）
使用 PBRS（Potential-Based Reward Shaping）：

```text
F(s,s') = gamma * Phi(s') - Phi(s)
Phi(s) = a * d_min_monster_norm + b * progress_to_best_treasure
```

这样能给稠密学习信号，同时不改变最优策略集合。

## P0：降低离策略更新比（UTD ratio）
- 把 `n_updates` 改成固定小值（如 4~8），不要跟 episode 长度强绑定。  
- 或者控制 `update_to_data_ratio <= 0.25`。  
- 回放池建议增大到 `200k~500k`，减缓近期样本主导。

---

## P1：探索与稳定性调参（SAC 专项）

1. **目标熵提高**  
   - 从 `0.5*logA` 提到 `0.8~1.0*logA`（前期）。  
   - 可做 schedule：前 30% 训练高熵，后期线性下降。

2. **alpha 下限放松**  
   - 当前 `alpha >= 0.2` 可能限制后期收敛形态。  
   - 建议 `clamp(0.03, 1.0)`，并监控策略熵。

3. **双重 Q 正则**  
   - 增加 target smoothing / clipped double Q 外的 conservative penalty（轻量 CQL 风格）防止高估。

4. **批大小回调**  
   - `BATCH_SIZE=1024` 可改 `256/512`，提高梯度噪声，减少“过平滑早收敛”。

---

## P1：特征工程升级（围绕“会规划”而不是“只反应”）

## 关键新增特征（建议新增 20~40 维）
1. **威胁时序**：
   - `eta_m1, eta_m2`（怪物到达 hero 的估计步数）
   - `eta_delta = eta_hero_to_safe - min(eta_m)`

2. **局部拓扑风险**：
   - 可逃逸方向数量（8邻域可达方向计数）
   - 前方 K 步通路长度（每个动作方向）
   - 死胡同深度 / 回环性

3. **目标价值特征**（宝箱 & buff）：
   - 最近 2~3 个宝箱的 `(dist, bearing, threat_adjusted_value)`
   - buff 剩余刷新时间（若观测可得）

4. **技能决策特征**：
   - 闪现可用倒计时
   - “闪现后安全增益”估计（8 个方向各一个）

## 表示学习建议
- 7×7 地图块改为 11×11 并过一个小 CNN，优于纯 flatten MLP。  
- 增加 4~8 步帧堆叠或 GRU（怪物追击是强时序过程）。

---

## P2：训练任务编排（Curriculum + Domain Randomization）

## 课程学习三阶段
1. **阶段A（生存）**：`treasure_count=2~4`，`monster_interval` 大，先学逃生。  
2. **阶段B（平衡）**：逐步恢复到默认参数。  
3. **阶段C（对抗）**：`monster_interval` 随机、`monster_speedup` 随机，强化鲁棒性。

## 环境随机化
- `map_random=true`（训练必须开启）。
- 随机化：怪物2出现时间、加速时刻、buff刷新时间。  
- 防止记地图脚本化策略，提升隐藏地图泛化。

---

## 3) 监控指标（没有这些就很难诊断）

除现有 loss 外，建议至少新增：
1. `episode_len`、`treasure_collected`、`score`（拆分 step_score/treasure_score）
2. `flash_use_count`、`good_flash_rate`
3. `policy_entropy`、`alpha`
4. `q_target_mean`、`q1_q2_gap`
5. `near_death_frames_ratio`（min_dist < 阈值的帧占比）
6. `stuck_ratio`（连续位置不变帧占比）

---

## 4) 推荐的最小可执行改造包（1 周版本）

## Day 1~2
- 动作空间改 16 + legal_action 全量接入。
- n_updates 改固定 8；batch 改 512；buffer 改 200k。

## Day 3~4
- 奖励改 PBRS：降低生存常数，强化“拿箱事件 + 终局 + 风险差分”。
- 加入 bad_flash/good_flash 事件奖励。

## Day 5~6
- 增加 ETA / 通道数 / 死胡同深度 特征。
- 目标熵从 0.5logA 提升到 0.9logA（前期），后期衰减到 0.6logA。

## Day 7
- 多地图评估（1~10）+ ablation：
  - baseline
  - +16动作
  - +奖励重构
  - +特征

---

## 5) 你现在这套代码的“直接修改建议”清单

1. `conf.py`
- `ACTION_NUM: 10 -> 16`
- `TARGET_ENTROPY_RATIO: 0.5 -> 0.8`（先）
- `BATCH_SIZE: 1024 -> 512`
- `REPLAY_BUFFER_SIZE: 50_000 -> 200_000`

2. `algorithm.py`
- `n_updates` 固定为 8（或与采样步数弱相关但上限更低）
- alpha clamp 改为更宽容区间（例如 `[0.03, 1.0]`）

3. `preprocessor.py`
- 奖励替换成“低常数生存 + 宝箱事件主奖励 + 终局惩罚 + PBRS风险差分”
- 增加闪现质量奖励（good/bad flash）
- 增加 ETA/可逃逸方向等先验特征

4. `train_env_conf.toml`
- `map_random = true`
- `monster_interval = -1`
- `monster_speedup = -1`

---

## 6) 一句话总结

你当前的 SAC 之所以“收敛快但效果差”，本质是：**动作表达受限 + 奖励导向偏保守 + 更新过强导致次优策略被快速固化**。优先做 **16动作补全、奖励重构、UTD降速、熵目标上调**，通常就能把曲线从“早收敛低分”拉到“稍慢收敛高分”。

# v0.4 — Plastic Self（v0.3 的补课）

> **为什么有这一版**：v0.3 测「我预测我将怎么预测」，诚实的答案是否定的——
> 二阶目标与一阶预测**冗余**（`incrR2` = 0.003），因为我在 t 时刻的下一次预测几乎是我当前预测的函数
> （`R²(p_t → p_{t+1}) = 0.989`）。那一次得出的设计规则是：
>
> 要让递归非冗余，必须存在一个量同时满足：**(a) 参与我未来的计算；(b) 不影响我此刻的输出；(c) 不可观测**。

v0.4 用**在线可塑性的隐藏调制器** `m_t` 满足这三条：它只进入**快权重更新规则**，
从不进入编码器 / latent / 预测通路。

```
slow path   p_slow = W_slow · z_t
fast path   p_t    = p_slow + F_t φ(x_t)                 （每步被重写的联想记忆）
update      F_t    = decay·F_{t-1} + lr·g_t·tanh(err_{t-1})·φ(x_{t-1})ᵀ
                    g_t = 1 + KAPPA·m_t                   ← 隐藏调制器只出现在这里
```

- **(a)** `m_t` 决定**下一次**记忆更新的增益 → 塑造 `p_{t+1}`；
- **(b)** `p_t` 在该更新生效**之前**就算完了 → `m_t` 不可能从我此刻的预测里读出来；
- **(c)** `m_t` 不在观测里。

因此预测「我下一步会怎么预测」**必须**估计**我自己的学习状态** —— 而它只能从**我自己的误差史**得到。
这就是「可塑性自我」的操作化含义：不是我的状态、不是我的历史，而是**我改变自己能力的那个参数**。

协议与判定标准见 **`../PROTOCOL_V04.md`**；实测见 **`../experiment_results.md` §17–§19**。

## 跑起来

```bash
python experiments/v04/train_v04.py --iters 800 --seeds 1,2,3            # 三个世界 × 三个 agent
python experiments/v04/train_v04.py --modes plastic,static --only Plastic --deterministic
```

## 目录

| 路径 | 内容 |
|---|---|
| `environment/plastic_world.py` | 世界：状态持久 + 动作增益每 ~10 步切换（让在线适应有用武之地）；`plastic` / `static`（m≡0）/ `visible`（m 进观测） |
| `agent/plastic_agent.py` | `Plastic`（慢权重 + 快权重 + 二阶头）/ `NoFast`（无可塑性 = 因果对照）/ `NoSelfPred`（去二阶头） |
| `agent/plastic_metrics.py` | 指标：`worldErr`、`selfPred` vs `trivial`、`selfR2fut`、`incrR2`（v0.3 失败的那个数）、**`mInfo`（latent → 我自己的可塑性）**、各自的 NULL |

## 结果一句话

**v0.3 的具体失败被修复了**：`incrR2` 0.003 → **0.072**（24 倍），二阶头首次赢过平凡基线（0.019 vs 0.168），
而且系统确确实实在估计**自己的可塑性**（`mInfo` 0.081，而拆掉快通路后是 **0.004**、可观测时是 0.993、时间打乱的 NULL 是 0.005）。
**没被证明的是「划算」**：可塑性在这里是世界预测的净成本（`NoFast` 0.392 < `Plastic` 0.529），
且世界自己的隐藏 regime 也贡献了同量级的非冗余信息（`static` 下 `incrR2` 仍有 0.050）——
只有 `plastic − static` 的 **+0.022** 能归因于自我。隔离实验留给 v0.6，方案已写进协议。

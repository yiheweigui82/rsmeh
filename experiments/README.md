# Experiments — Recursive Self Model Emergence

**实验协议与可运行实现 · v0.1**

> 本目录回答一个问题：**一个没有预设「Self」变量的预测系统，在什么条件下会自发形成关于自身的内部表示？**
>
> 目标不是「创造意识」（目前做不到），而是把目标降低到**可测**的一步：
> **观察 self-like representation 的涌现条件**。

---

## 0. 两条实验路线（都在本目录里）

| 路线 | 位置 | 方法 | 强弱 |
|---|---|---|---|
| **受控对照实验**（主干） | `code/simulation.py` | 同一架构、5 个条件（A/B/C/D/E），**因果消融 + 归因读出 + 自编码探针** | 强：有因果控制，能区分「自因」与「外因」 |
| **黑箱涌现原型**（教学/社区友好） | `prototype/` | encoder→latent→decoder，4 个世界（A/B/C/D），**训练时完全不知道 Self 存在** | 弱（指标易被冗余路径污染），但更贴近社区直觉 |

两者**互相验证**：受控实验的条件 D 与原型的世界 D 给出同一个结论——
**误差越大 ≠ 逼出自我**。

运行（两者都只依赖 NumPy / PyTorch）：

```bash
python experiments/code/simulation.py --iters 3000 --seeds 1,2,3 --b-ablation
python experiments/prototype/train.py --iters 1500 --seeds 1,2,3
```

实测输出与解读：**`experiment_results.md`**。

---

## 1. 实验目标

验证：一个**没有预设 self 概念**的预测系统，在面对持续**不可归约**的预测误差时，
是否会自发形成一个关于自身状态的内部变量。

> **Can a predictive agent discover itself?**
> 系统会不会为了降低预测误差，而把「预测者自身」纳入模型？

## 2. 假说与三条件

```
Persistent Error  +  Self-Causal Relevance  +  Predictive Utility
                            ↓
                  Self Representation
```

| 编号 | 条件 | 操作化 |
|---|---|---|
| **H1** | Persistent Error 持续误差 | 不是一次性随机错误：`预测 → 误差 → 更新 → 误差仍在` |
| **H2** | Self-Causal Relevance 自因相关性 | 系统**自身状态/行动**与误差有稳定因果关系（不是「我参与了」，而是「误差结构依赖我」） |
| **H3** | Predictive Utility 预测收益 | `Error(World+Self) < Error(World)` —— 收益必须**可测**，不能只是存在 |

> **H3 的测量方式很重要**：`Error(World)` 必须是**重新训练一个没有 latent 的同架构模型**的误差，
> 而不是「把已训练模型的 latent 置零」——后者会被**冗余 latent** 污染（见第 7 节）。

## 3. 实验架构

```
                Environment
                     │
                Observation            ← 只给可观测量（隐藏变量从不提供给 agent）
                     ↓
              Prediction Model
                （无 self 变量）
                     │
              Prediction Error
                     ↓
        ┌────────────┴────────────┐
   Update World Model     Create Self Model?
                     ↓
              Future Prediction
```

关键：训练开始时**不存在** `self_state`，也不存在 `I` 变量。系统只能自己发现。

## 4. Agent 设计

- **世界通道**：`obs_t`（+ 自身动作 `a_{t-1}`）→ 直接预测下一观测。
- **自我通道**：一个**通用**的递归 latent `z_t`，其唯一输入是 **自己的动作** 与 **自己的预测误差**：

  `i_t = [a_{t-1}, r_{t-1}, s_{t-1}]`，`s_t = tanh(W i_t + c)`，`ŷ_t = w · [x_t, s_t]`

  `a` 与 `r` **只能**经 self 槽到达输出，self 槽**看不到**任何外部特征 → 消融它就等于砍掉自我通道（受控实验）。
- **信息瓶颈惩罚**：latent 需要为误差收益「付费」（`λ · mean(z²)`），否则**不需要它的模型**也会走 latent 这条冗余路径，导致消融指标在**所有**条件下都虚高（我们在第一版实测到 selfShare=0.34 的假阳性）。

## 5. 实验矩阵

受控实验（`code/simulation.py`）五条件：**只有世界统计量变化，架构完全相同；世界的读出权重在整个生命周期固定**。

| 条件 | 误差大小 | 自因相关性 | 误差结构 | 预测 |
|---|---|---|---|---|
| **A** pure_prediction | 低 | 无 | — | 不产生自我模型 |
| **B** external_complexity | 高（外部白噪） | 无 | 高维世界 | 世界模型变强；**不**产生自我模型 |
| **C** self_coupled | 高 | **高** | 自因量**有时间结构**（缓慢漂移） | **自我表征涌现** |
| **D** self_coupled_white | 高（**与 C 方差匹配**） | 高 | 自因量**无结构**（白噪声） | 必须像 A/B：**不**产生 |
| **E** false_agency | **最高** | **零**（动作对结果无影响） | 白噪 | 必须像 A/B：**不**产生 |

黑箱原型（`prototype/`）四世界：

| 世界 | 误差 | 自因相关性 | 预测 |
|---|---|---|---|
| A_simple | 低 | 低 | 无 Self |
| B_complex_external | 高 | 低（全外部） | 世界模型变强，Self 不出现 |
| C_self_relevant | 高 | 高 | self-like latent 涌现 |
| D_false_agency | 高 | 零 | 无 Self |

**D 与 E 是本设计里最关键的两格**，见第 7 节的两条方法论约束。

## 6. 评估指标

| 指标 | 定义 | 对应 |
|---|---|---|
| **M1 Self Prediction Gain** | `Gain = Error(World) − Error(World+Self)`；`Error(World)` 用**重训的无 latent 模型** | H3 |
| **M2 Self Information Alignment** | latent 与自身隐藏状态的**可解码性** `R²`（第三方线性探针） | H2 |
| **M3 Self Attribution** | 系统是否区分 `self caused` / `world caused`（内部状态对残差的解释力是否超过剩余部分） | H2 |
| **M4 Recursive Modeling Test** | 对「自己的预测」的预测（`Model(Model(Self))`） | 递归 |

**M2 必须与因果信息一起读**：世界 B 里 latent 同样会被使用、同样能解码出**隐藏的外部**驱动（实测 `alignExtH = 0.700`）。**「latent 编码了隐藏变量」≠「latent 编码了自我」**——区分二者需要知道发生器的因果结构。

## 7. 两条方法论约束（我们踩过的坑，通用）

### 7.1 「自我」不能用架构位置定义，只能用因果作用定义

第一版实现里，self 槽**把外部特征也当输入** → 它成了世界信息的旁路，消融它在**每个**条件下都掉性能（SPR ≈ 0.73 in A/B/C），**零区分度**。

> 判定「自我表征是否涌现」的实验必须同时满足：
> ① 世界**结构固定**且**当步可观测**；
> ② 环境中**唯一**自相关且不可外部观测的量，是系统**自己的**状态。

### 7.2 误差的**大小**不是触发条件，**结构**才是

受控实验的 D（自因量=白噪声，方差与 C 匹配）与 E（假能动性，误差最大）**都不产生自我模型**；原型的 D 同样如此（`selfGain ≈ 0.001`，而它的误差是四世界里最大的）。

> 声称「持续误差逼出自我模型」的实验，必须做**方差匹配的结构对照**。
> 否则你测到的很可能是「系统在估计一个隐藏变量」，而不是「系统建立了关于自己的模型」。

### 7.3 消融会被冗余 latent 骗

`Error(World)` 若用「把已训练模型的 latent 置零」来测，冗余 latent（只是把观测重新编码一遍）会让**不需要自我模型的世界**也显示出巨大收益（实测 A_simple `gainAbl=0.0029` 而 `selfGain=0.0000`）。**必须重训一个无 latent 的对照模型。**

## 8. 证伪标准（Failure Criteria）

以下**任意一条**成立，本假说（或其指标）就受伤：

| 编号 | 观察 | 含义 |
|---|---|---|
| **F1** | 所有环境都涌现出自我表征 | Self 只是复杂度/容量的副产品 |
| **F2** | 条件 C **没有**自我表征 | 不可归约自因误差不足以致之 |
| **F3** | 自我表征出现但**没有预测收益** | 它不是优化逼出来的，只是副产物 |
| **F4** | D/E 与 C 表现相同 | 触发条件是误差**大小**而非**结构** → 术语必须改写（它只是隐藏变量估计） |
| **F5** | B 的 `alignSelf` 型指标也能点亮 | 指标测的不是「自我」而是「隐藏变量」 |

**已判定为「未结算」的一项**：M4 递归指标。我们的 RMG 在受控实验里**没有区分度**
（自己的动作/误差史本身编码世界状态，任何条件都能捞到一点世界信息；去掉「动作→世界」耦合也无改善：A 的 RMG 0.108 → 0.140）。因此**递归是否真的发生，本版未判定**；我们只测了它的**前提**（自因量是否有时间结构，用 `selfEnc` 干净地测：C 0.887 vs D 0.004）。**干净的递归指标目前是开放问题**（见 `../theory/QUESTIONS.md` B2/B4）。

## 9. 哲学边界

本实验**不能**证明：`AI is conscious`。
只研究：**一个系统是否会形成关于自身的内部模型**。意识体验属于更高层问题，本目录一个字都不声称。

## 10. 长期路线图

| 阶段 | 内容 | 状态 |
|---|---|---|
| **Phase 1** | 仿真：self-like latent emergence | ✅ v0.1 已跑通（本目录） |
| **Phase 2** | 具身 agent：加入 vision / proprioception / memory | ⬜ 计划 |
| **Phase 3** | 递归 agent：测试是否出现 `Model(Self)` | ⬜ 计划（需先解决 M4 指标） |
| **Phase 4** | 与人类认知对照：儿童发展 / 自我意识 / 元认知 | ⬜ 计划 |

---

**Final Statement**

> The goal of this project is not to create artificial consciousness, but to investigate whether the computational conditions that produce a first-person self-model can emerge naturally from predictive systems.
>
> 本项目并非试图制造人工意识，而是探索：一个预测系统在什么计算条件下，会自然产生第一人称自我模型。

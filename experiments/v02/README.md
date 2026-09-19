# v0.2 — Temporal Self Model Emergence

> **问题**：v0.1 证明了「latent 编码隐藏变量」**不等于**编码「自我」。v0.2 沿**时间轴**再问一次：
> **是否存在一个跨时间保持连续的表征？而这个连续性属于「自我」，还是只属于「持久性」？**
>
> 假说（Temporal Self Continuity）：当系统必须解释**自己的过去如何约束自己的未来**时，
> 一个稳定的自我表征可能作为**跨时间的信息压缩结构**出现：`past → me → future`。

完整协议与判定标准见 **`../PROTOCOL_V02.md`**；实测输出与解读见 **`../experiment_results.md`**。

---

## 跑起来

```bash
pip install -r experiments/v02/requirements.txt     # torch（CPU 够用）+ numpy
python experiments/v02/train_v02.py --iters 1200 --seeds 1,2,3                      # 主表
python experiments/v02/train_v02.py --only T1_self_persistent --identity-loss naive # 身份损失消融
python experiments/v02/analysis/identity_test.py --world T1_self_persistent         # 身份连续性分析
python experiments/v02/analysis/counterfactual_test.py                              # 反事实分析
```

## 目录

| 路径 | 内容 |
|---|---|
| `environment/temporal_self_world.py` | 三个世界 T1 / T2 / T3（统计量相同，只有因果位置不同） |
| `agent/memory_agent.py` | 三个 agent：`Memory`（持久 latent）/ `NoPersist`（无持续性对照）/ `NoLatent`（无 latent 基线） |
| `agent/temporal_metrics.py` | 指标：分层预测误差、**反退化**的连续性指标对（drift + separability + info）、反事实误差 |
| `train_v02.py` | 主入口（含 `--identity-loss {none,naive,contrastive}` 消融） |
| `analysis/identity_test.py` | 身份连续性分析；**把「死的 latent」与训练出的 latent 并列打分**，证明朴素身份指标会奖励常量解 |
| `analysis/counterfactual_test.py` | 反事实分析；用真实隐藏变量的分支差（T1 非零 / T3 恒为 0）标出「我的动作是否驱动我的状态」 |

## 三个世界（统计量相同，只有一个因果差异）

| 世界 | 隐藏变量 | 动力学 | **我的动作**能驱动它吗 |
|---|---|---|---|
| **T1** `self_persistent` | **agent 自己的状态** | AR(0.92) 漂移 + 噪声 | **能**（增益 0.25） |
| **T2** `self_white` | agent 自己的状态 | 白噪声（**方差匹配**） | 能，但**无时间结构** |
| **T3** `external_persistent` | **属于世界的隐藏驱动** | AR(0.92) 漂移 + 噪声 | **不能**（增益 0） |

**T1 vs T3 是本版的核心对照**：两者统计量、不可观测性、对结果的影响**完全相同**，
只有因果位置不同——我的动作是否驱动它。若所有「自我」指标在两者上同时点亮，
那它们测的是**持久性**，不是**自我**。

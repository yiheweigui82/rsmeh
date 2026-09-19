# v0.3 — Recursive Prediction

> **问题**：v0.1 问「系统是否发现自身状态」，v0.2 问「连续性属于自我还是持久性」。
> **v0.3 第一次问：系统是否开始把「自己的预测过程」当成预测对象？**
> 即 `M(W, S)` → `M(W, S, M)`。
>
> 硬约束：**递归不许写进架构**。agent 里没有任何叫 `self` / `self_model` / `I` / `recursive`
> 的变量；递归若真出现，必须由**任务**逼出来，再由**外部探针**找出来。

协议与判定标准见 **`../PROTOCOL_V03.md`**；实测输出见 **`../experiment_results.md`** §10–§12。

## 跑起来

```bash
pip install -r experiments/v02/requirements.txt   # torch + numpy 已覆盖
python experiments/v03/train_v03.py --iters 900 --seeds 1,2,3
python experiments/v03/train_v03.py --only R1_prediction_loop --deterministic
python experiments/v03/analysis/recursion_test.py --world R1_prediction_loop
python experiments/v03/analysis/recursion_test.py --world R2_exogenous_control
```

## 目录

| 路径 | 内容 |
|---|---|
| `environment/recursive_prediction_world.py` | 三个世界 R1（预测反馈闭环，隐藏漂移增益）/ R2（等方差**外生**对照）/ R3（可见常数响应）；世界**逐步**用 agent 自己的预测推进 |
| `agent/recursive_agent.py` | `Rec`（递归 latent + 二阶头）/ `NoSelfPred`（去掉二阶头重训 = 因果对照）/ `NoRec`（无持续性 = 连续性对照） |
| `agent/recursive_metrics.py` | 指标：`worldErr`、`vacuousR2`（草稿的恒亮指标，故意留着）、`selfPred` vs `trivial`、`selfR2fut`、`incrR2`、`gainInfo` |
| `train_v03.py` | 主入口：世界与 agent **逐步交织**（这才是因果闭环），BPTT 覆盖整段 episode |
| `analysis/recursion_test.py` | 修好的递归测试：未来预测探针 + **NULL 对照**（时间打乱的 latent）+ `loop check`（世界对我的预测的响应是否可学） |

## 世界（形式相同，只差一个因果差异）

```
x_{t+1} = 0.85·x_t + 0.20·a_t + 0.15·gain_t·tanh(pred_{t-1}) + 0.05·ξ
```

| 世界 | 响应项 | 我的预测能推动世界吗 | 增益 |
|---|---|---|---|
| **R1** `prediction_loop` | `0.15·gain_t·tanh(pred_{t-1})` | **能** | **隐藏、缓慢漂移** |
| **R2** `exogenous_control` | `0.15·gain_t·tanh(ξ)` | **不能**（外生） | 常数 |
| **R3** `visible_response` | `0.15·gain_t·tanh(pred_{t-1})` | 能 | 常数、可见 |

**R2 是本版最关键对照**：同样幅度、同样函数形式的额外驱动，但**不是由我造成的**。
没有它，「涌现出递归结构」无法排除「世界本身有个自相关扰动」。

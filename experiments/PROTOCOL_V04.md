# RSMEH v0.4 — Plastic Self (protocol)

**Status:** v0.4 · the repair attempt for v0.3 · companions `PROTOCOL.md`, `PROTOCOL_V02.md`, `PROTOCOL_V03.md`
**Code:** `v04/` (PyTorch, CPU is enough)

---

## 1. Why v0.4 exists

v0.3 tested "I predict what I will predict" and the honest answer was **no**: the
second-order content was redundant with the first-order prediction
(`incrR2` = **0.003**), because at time *t* my next prediction was almost a
function of my current one (`R²(p_t → p_{t+1}) = 0.989`). The second-order head
even lost to the trivial baseline, and closing the loop made the world *harder*.

The lesson extracted there was a design rule, and v0.4 is built to satisfy it:

> For recursion to be non-redundant there must be a quantity that is
> **(a)** involved in my *future* computation,
> **(b)** not involved in my *current* output, and
> **(c)** unobservable,
> so that only my own error history can reveal it.

## 2. The mechanism: a hidden modulator of my own plasticity

The agent has two pathways:

```
slow path   p_slow = W_slow · z_t                     (learned, stationary)
fast path   p_t    = p_slow + F_t φ(x_t)              (an associative memory,
                                                       rewritten every step)
update      F_t    = decay·F_{t-1} + lr·g_t·tanh(err_{t-1})·φ(x_{t-1})ᵀ
                    g_t = 1 + KAPPA·m_t
```

`m_t` is a hidden AR(1) process. It enters the **update rule only** — never the
encoder, never the latent, never the prediction path. Formally:

- **(a)** `m_t` sets the gain of the *next* memory update, so it shapes `p_{t+1}`;
- **(b)** `p_t` was computed *before* that update applies, so `m_t` cannot be read
  out of my current prediction;
- **(c)** `m_t` is not in the observation.

Hence predicting my own `p_{t+1}` requires an estimate of **my current learning
state** — obtainable only from the sequence of my own errors. That estimate is
the operational meaning of a *plastic self*: not "my state", not "my history",
but **the parameter of my own capacity to change**.

This construction is deliberately analogous to neuromodulation: a biological
system's plasticity is modulated by things it cannot introspect; it sees only the
consequences. We state it as a construction, not as biology.

## 3. Worlds — isolating the modulator

| mode | modulator | what it tests |
|---|---|---|
| `plastic` | hidden AR(1) `m_t` | the main condition |
| `static` | `m_t ≡ 0` | **the falsifier**: no hidden plasticity → nothing to infer |
| `visible` | `m_t` appended to the observation | **readable self**: nothing needs to be inferred, the modulator is part of the input |

The world itself is a persistent state whose **action gain switches between two
regimes** every ~10 steps, so *online adaptation* has a job to do — the fast
pathway is functionally useful, not decorative.

## 4. Agents

| agent | slow | fast (plastic) | second-order head |
|---|---|---|---|
| `Plastic` | yes | yes | yes — predicts my own next prediction |
| `NoFast` | yes | **no** | yes — the causal control: no plasticity at all |
| `NoSelfPred` | yes | yes | **removed** (retrained) — the head's causal control |

## 5. Metrics

| metric | meaning |
|---|---|
| `worldErr` | normalised next-step error (vs a persistence floor) |
| `selfPred` | the second-order head's error on **my own next prediction** |
| `trivial` | the bar to beat: `MSE(p_t, p_{t+1})` — "my next prediction = my current one" |
| `selfR2fut` | `R²(z_t → p_{t+1})` |
| **`incrR2`** | **extra** `R²` for `p_{t+1}` beyond `(p_t, x_t, a_t)` — **the number v0.3 failed at (0.003)** |
| `incrNull` | the same probe with the latent shuffled in time |
| **`mInfo`** | `R²(z_t → m_t)`: does my latent estimate **my own plasticity**? (`n/a` in `static` where it is identically zero) |
| `mNull` | its shuffled-latent floor |

## 6. Judgement criteria

**C1 — the repair worked.** `incrR2(plastic)` must be well above v0.3's 0.003 and
above `incrNull`.

**C2 — the second-order head finally wins.** `selfPred < trivial` in `plastic`.

**C3 — the system estimates its own plasticity.** `mInfo ≫ mNull` in `plastic`;
`n/a` in `static`.

**C4 — the causal controls.** `NoFast` must lose the effect (no plasticity → the
latent has no self-parameter to estimate), and `static` must show a smaller
`incrR2` than `plastic`.

**C5 — `visible` must not need inference.** With the modulator in the input the
information is available to the control variables, so the latent's *incremental*
value should fall back toward the `static` level. If `visible` shows just as much
`incrR2` as `plastic`, the metric is not measuring self-inference.

**A confound we state in advance.** The world's own regime switching is a second
source of non-redundant information: to predict my next prediction I must also
know *which regime I am in*, and that too is only visible through my errors. So
`incrR2` will **not** be zero even in `static`, and part of `plastic`'s number is
not about `m_t` at all. The modulator's own contribution must therefore be read as
the **difference** `incrR2(plastic) − incrR2(static)`, never as the raw value. A
cleaner isolation (a stationary world with a per-episode hidden constant, or a
capacity-limited slow path) is left for v0.6 and is named in the results file.

## 7. Reproducibility

```bash
python experiments/v04/train_v04.py --iters 800 --seeds 1,2,3
python experiments/v04/train_v04.py --modes plastic,static --only Plastic --deterministic
```

Seeds are set **before** model construction (the v0.2 bug). `--deterministic`
pins a single CPU thread for bit-identical re-runs. One engineering note that
belongs to the record: the first version of the fast-weight pathway **diverged**
(1e17 by iteration 40) because my own error rewrote my own memory, which changed
my next error — a positive feedback loop. It is stabilised by normalising the
retrieval (`F φ / (‖φ‖²+1)`) and bounding what is stored (`tanh(err)`).

## 8. What v0.4 does NOT claim

- **No consciousness, no experience.** Same ladder position as v0.1–v0.3.
- **Not "the self is the only hidden thing".** The world's regime is hidden too
  (§6 confound). v0.4 shows that a *hidden plasticity modulator* adds
  non-redundant second-order information; it does not show that this is the sole
  source, nor that it is *necessary* — a sufficiently capable slow recurrent net
  may perform in-context adaptation without any explicit fast pathway.
- **The modulator is a construction.** Its unobservability is a property of the
  code we wrote, not a discovery about minds.

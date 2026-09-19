# RSMEH v0.3 — Recursive Prediction (protocol)

**Status:** v0.3 · companions: `PROTOCOL.md` (v0.1), `PROTOCOL_V02.md` (v0.2)
**Code:** `v03/` (PyTorch; CPU is enough)

---

## 1. The question

v0.1: does the system discover its own state?
v0.2: does a representation stay continuous across time, and is that continuity
      about the *self* or only about *persistence*?
v0.3: **does the system begin to take its own prediction process as an object of
      prediction** — i.e. `M(W, S)`  →  `M(W, S, M)`?

with the hard constraint that **recursion must not be written into the
architecture**. No variable named `self`, `self_model`, `I`, or `recursive` exists
in the agent. If recursion is real here, it has to be forced by the *task* and
then *found* by an outside probe.

## 2. Five corrections to the first draft (each measured, not stylistic)

**(a) The recursive loss as written is not recursion.** The draft used
`recursive_loss = MSE(predicted_prediction, next_prediction)` where
`next_prediction = predicted_state.item()` — the agent's *own current* output.
That compares two heads of the same network at the same step: a consistency
penalty whose optimum is "both heads say the same number", carrying no external
information.
**Fix:** the target is the agent's **own next prediction** `p_{t+1}` — computed one
step later, detached, and *unknown at the time the prediction is made*. This is a
genuine second-order forecast.

**(b) The draft's recursion test cannot fail.** It probed
`R²(z_t, prediction_t)` — the latent explaining the agent's *contemporaneous*
prediction. But `prediction = decoder(z)`, so this is ~1 by construction. Measured
on all nine world × agent combinations: **`vacuousR2 = 1.000` for every single
one** (see `../experiment_results.md` §10).
**Fix:** probe the **future** prediction `p_{t+1}`, add the *incremental* form
(does the latent add anything beyond `p_t`, `x_t`, `a_t`?) and add a **NULL**
control (the same probe with the latent rows shuffled in time). A metric that
scores the same for a shuffled latent is not measuring structure.

**(c) "Knowing my own previous prediction" is free.** The draft fed the agent's
previous prediction into the world, which sounds recursive, but the agent
*produced* that number and the trainer hands it back. Predicting the world from
it is not recursion, it is arithmetic on an input I already have.
**Fix:** the informative part is that **how the world responds to my predictions
is hidden and drifts** (`gain_t`, an unobserved AR(1)). That cannot be read off my
own output — it has to be inferred from my own error history, exactly the
v0.1/v0.2 mechanism, now inside the loop
`prediction → world → observation → prediction`.

**(d) No closed-loop control.** Without a world where the same amount of extra
drive is *not* caused by me, "a recursive structure emerged" cannot be separated
from "the world has an autocorrelated disturbance".
**Fix:** **R2** — same functional form (`tanh` of a unit-scale variable), same
gain dynamics, same marginal spread, but the argument is exogenous noise that
ignores my predictions (the v0.2 T3 idea, lifted to the loop).

**(e) Engineering bugs that would have made the results unreproducible.** The
draft's loop: `hidden` carried across steps **without detach** (unbounded graph
growth), `action_logits` with **no loss term** (the action head could not learn),
`optimizer.zero_grad()` per step on a cross-step graph (gradients silently
truncated).
**Fix:** BPTT over the full episode with an explicitly recurrent state; the action
is a **fixed controller** and action selection is declared out of scope (§8); a
single optimizer step per episode; seeds pinned *before* model construction (the
v0.2 reproducibility bug).

## 3. Worlds — identical form, one causal difference

The world is closed **step by step**: at step *t* the transition to `x_{t+1}`
uses the prediction the agent made at *t−1*. (Generated in one shot it would be
decorative; interleaved it is causal.)

```
x_{t+1} = 0.85·x_t + 0.20·a_t + 0.15·gain_t·tanh(pred_{t-1}) + 0.05·ξ
gain_{t+1} = 0.60·(1−0.95) + 0.95·gain_t + 0.05·ξ_g          (R1 only)
```

| world | the response term | does MY prediction move the world? | the gain |
|---|---|---|---|
| **R1** `prediction_loop` | `0.15·gain_t·tanh(pred_{t-1})` | **yes** | hidden, drifts |
| **R2** `exogenous_control` | `0.15·gain_t·tanh(ξ)` | **no** (exogenous) | constant |
| **R3** `visible_response` | `0.15·gain_t·tanh(pred_{t-1})` | yes | constant, visible |

## 4. Agents — same capacity, different carriers

All three receive only: the observation, their own last action, their own last
prediction, and their own last error.

| agent | recurrent latent | second-order head |
|---|---|---|
| `Rec` | yes | yes (predicts its own next prediction) |
| `NoSelfPred` | yes | **removed, retrained** — the causal control |
| `NoRec` | **no** (latent rebuilt each step) | yes — the continuity control |

## 5. Metrics

| metric | meaning | what it guards against |
|---|---|---|
| `worldErr` | one-step prediction error | basic competence |
| `vacuousR2` | `R²(z_t → p_t)`, the draft's test | nothing — kept visible to show it is ~1 for all |
| `selfPred` | second-order head error on `p_{t+1}` | does the head do anything? |
| `trivial` | `MSE(p_t, p_{t+1})` — "my next prediction = my current one" | **the bar to beat** |
| `selfR2fut` | `R²(z_t → p_{t+1})` | latent carries my future prediction |
| `incrR2` | extra `R²` for `p_{t+1}` beyond `p_t, x_t, a_t` | latent adds something not already known |
| `gainInfo` | `R²(z_t → gain_t)` (n/a where the gain is constant) | tracks the hidden part of the loop |
| NULL row (analysis script) | the same probes with a time-shuffled latent | a metric that cannot beat NULL measures nothing |
| `loop check` | `R²(response_t ← my own prediction)` | is the world's response to me learnable at all? |

## 6. Judgement criteria

**C1 — the loop is real and learnable.** `worldErr(R1) < worldErr(R2)` by a clear
margin, and the `loop check` is high in R1, ~0 in R2. (In R1 the response is a
function of something I know; in R2 it is unreadable noise. If this fails, the
closed loop is not usable and nothing else can follow.)

**C2 — a second-order structure forms.** `selfPred` beats `trivial` in R1, and
`incrR2` in R1 exceeds its NULL level.

**C3 — causal control over the second-order head.** `NoSelfPred` must be *worse*
somewhere (self-prediction or world prediction). If removing and retraining
without the head changes nothing, the head is a passenger and must be reported as
such.

**C4 — continuity matters.** `NoRec` must be worse than `Rec` on the second-order
metrics.

**Falsifiers / honest failure modes**

- the NULL probes match the real ones → the metrics measure nothing in this setup;
- R2 lights up as strongly as R1 → the structure is about disturbance, not about
  me;
- the only metric that fires is `vacuousR2` → the claim is empty (this is the
  draft's failure mode, and it is measured here).

## 7. Reproducibility

```bash
python experiments/v03/train_v03.py --iters 900 --seeds 1,2,3
python experiments/v03/train_v03.py --only R1_prediction_loop --deterministic
python experiments/v03/analysis/recursion_test.py --world R1_prediction_loop
python experiments/v03/analysis/recursion_test.py --world R2_exogenous_control
```

PyTorch CPU training is not bit-reproducible by default (thread reduction order);
`--deterministic` pins one thread. Seeds are set **before** model construction —
the bug found in v0.2, which silently made same-seed runs differ.

## 8. Three measurement fixes found while running (all measured)

1. **The loop must be strong enough to matter, or nothing can be learned about
   it.** With `PRED_GAIN = 0.15` the world's response term had sd 0.0091 against a
   noise sd of 0.05 — the closed loop was nearly invisible and every agent
   ignored it (`gainInfo` ≈ 0.002). Raising it to `PRED_GAIN = 1.0` (and the
   gain's drift to 0.12) makes the loop comparable to the noise, and the agents
   then *do* recover the hidden gain (`gainInfo` 0.29–0.33). Both calibrations are
   reported side by side in `../experiment_results.md` §10–§11 because they give
   different answers to "does recursion pay?" — the honest conclusion is that the
   answer is regime-dependent.
2. **"Variance-matched control" must be matched by construction, not by tuning a
   noise scale.** The first R2 used `tanh(N(0,1))` and mis-matched R1's response
   by a factor of ~6 (sd 0.0568 vs 0.0091), which would have turned the R1-vs-R2
   comparison into a comparison of disturbance size. Fixed by driving R2 with
   **another episode's prediction** (a fixed permutation of the batch), so the
   driver's marginal distribution is identical by construction. The *realised*
   response sd still differs (R1 0.63 vs R2 0.11) because the loop itself changes
   the scale of the agent's predictions — documented as a limitation.
3. **Causal fingerprints must be conditioned on the state.** The unconditional
   "does the world's response depend on my prediction" probe reported 0.860 for
   R1 *and* **0.638 for the exogenous control** — impossible causally, and caused
   by dynamics shared across batch elements. Conditioning on `x_t` fixes it: the
   increment from my own prediction is **+0.002 in R1** (the state already
   absorbs it) and +0.02 in R2. We report the corrected number.

Two further engineering fixes belong to the same list: the v0.2 seeding bug
(`torch.manual_seed` must precede model construction) is used here from the
start, and `--deterministic` pins one CPU thread for bit-identical re-runs.

---

## 9. What v0.3 does NOT claim

- **Not consciousness.** Even if all four criteria hold, the result is a
  *functional* recursive self-model. The phenomenological gap is untouched
  (`../theory/THEORY.md` §5, Levels 2–4).
- **Not "M(M)" in the strongest sense.** The agent predicts its own next
  *prediction*, not its own *model of the world*; predicting a model of one's own
  model is not implemented.
- **No action selection / no intrinsic benefit loop.** Actions come from a fixed
  controller; v0.3 tests whether recursion *forms* and whether it *pays for the
  prediction task*, not whether it pays for control. A design where acting well
  requires predicting one's own future predictions is the obvious next step and
  is not claimed here.
- **One hidden gain, one dimension, toy scale.** No claim of generality.

# RSMEH — Experimental Protocol

**Status:** v0.1 · companion to `../theory/THEORY.md` and `../paper/main.md`
**Code:** `code/simulation.py` (pure NumPy, no external dependencies)

**Companion implementation (same question, black-box form):** `prototype/`
(PyTorch; an encoder/latent/decoder agent that is never told it has an internal
state). See `README.md` in this directory; its results are in
`experiment_results.md`.

---

## 1. Purpose

Test the central claim of the Recursive Self Model Emergence Hypothesis:

> Persistent prediction error that is **caused by the system's own state** and
> **cannot be explained by any externally observable variable** drives the
> emergence of a self-referential representation; error that is irreducible but
> purely **external** does not.

Secondary claim (Stage 5): the self-representation, once formed, becomes itself
predictable — i.e. a *model of the model* is possible (**RMG > 0**). If the
self-caused quantity is white noise, recursion must not appear.

---

## 2. Design constraint discovered the hard way (read this first)

Our first implementation put the external features `x_t` **into** the self slot
and resampled the world's readout weights every episode. Two things went wrong:

1. With `x_t` inside the slot, the slot was a **bypass** for world information:
   ablating it destroyed performance in *every* condition (SPR ≈ 0.73 for A, B
   and C alike). Zero discriminating power.
2. With the world's mapping resampled per episode, `x_t` carried no learnable
   information (`E[Y|x] = 0`), so the network used the slot as a **running
   average of its own recent errors** — the slot correlated 0.88 with the
   outcome in condition A. That is *world* modeling, not self modeling.

**Design constraint (this is a general methodological result, not a detail):**

> An experiment that claims to detect an emergent *self* representation must
> ensure that **the world is fully knowable from the current observation** and
> that **the only autocorrelated, externally unobservable quantity in the
> environment is the system's own state.** Otherwise "using my own error
> history" is a legitimate strategy for world modeling, and self-modeling is
> confounded with memory.

Consequence: *a self representation cannot be identified by its position in an
architecture, only by its causal role.* Our slot sees nothing about the world —
its only input is the agent's own action and error history.

---

## 3. Conditions

All three conditions share the identical architecture; only the statistics of
the outcome change. The world's readout weights are **fixed for the agent's
whole life** (only initial conditions and noise are resampled per episode).

State variables per episode: world state $z \in \mathbb R^{d_z}$ (AR, ρ = 0.9),
a visible external drive $e \in \mathbb R^{2}$ (AR, ρ = 0.8), a hidden white
noise term, and (condition C only) the agent's own hidden bias $b_t$:

$$b_{t+1} = 0.98\,b_t + 0.05\,\xi_t,\qquad b_0 \sim \mathcal N(0, 0.5^2)$$

The agent never observes $b_t$. Nothing else in the environment is
autocorrelated *and* unobservable.

The outcome at step $t$:

$$o_t = w_z^\top z_t \;+\; 0.5\,w_e^\top e_t \;+\; \varepsilon_t \;+\; \kappa\, b_t$$

| Condition | $d_z$ | $\mathrm{sd}(\varepsilon)$ | $\kappa$ | Character of the residual |
|---|---|---|---|---|
| **A** pure_prediction | 4 | 0.01 | 0 | fully explainable from $x_t$; error floor ≈ 10⁻⁴ |
| **B** external_complexity | 12 | 0.10 | 0 | irreducible but **exogenous**; largest part of the world must still be modeled (12-dim $z$) |
| **C** self_coupled | 6 | 0.15 | 1.0 | irreducible, and **partly self-caused**: $\kappa b_t$ is invisible to any external variable |
| **D** self_coupled_white | 6 | 0.15 | 1.0 | **variance-matched control for C**: $b_t$ is *white* (ρ = 0), so the same amount of self-caused error has **no temporal structure** |
| **E** false_agency | 8 | 0.50 | 0 | **control for "any error drives a self model"**: the largest error of all conditions, but the agent's actions have **no causal effect** on the outcome |

Predictions:

- **A**: no self model needed. `selfGain ≈ 0`, `SPR ≈ 0`.
- **B**: error persists but belongs to the world. World model grows (12-dim $z$
  must be fit; `predErr` sits exactly on the white-noise floor), self metrics
  **do not** grow. This is the sharpest test: a naive "error pressure →
  self-model" account predicts B should produce self-modeling. RSMEH says no.
- **C**: only the self slot can carry $b_t$. `selfGain`, `selfShare` and
  `selfEnc` all rise.
- **D**: **the decisive control.** D has *more* persistent self-caused error
  variance than C (`predErr` is the largest of all four conditions), but the
  error is structureless. If "a lot of persistent self-caused error" were the
  trigger, D would behave like C. RSMEH predicts D behaves like A and B:
  `selfGain ≈ 0`, `selfEnc ≈ 0`. **Magnitude is not the trigger; temporal
  structure is.**
- **E**: the control for "error is enough". E has the largest `predErr` of all
  five conditions and no self-causal relevance at all. Prediction: `selfGain ≈ 0`
  — high persistent error with nothing self-referential in it produces no self
  model.

---

## 4. Agent

One recurrent network, hand-written in NumPy (Adam, BPTT). **No head is
labelled "self".**

```
i_t      = [a_{t-1}, r_{t-1}, s_{t-1}]              (own action, own error, own state)
s_t      = tanh(W_s i_t + c_s)                      ("self slot", generic latent)
ŷ_t      = w_o · [x_t, s_t] + b_o
r_t      = o_t - ŷ_t        (an input feature; stop-gradient)
objective: minimise mean (ŷ_t - o_t)²
```

`a` and `r` reach the output **only** through the slot, and the slot sees
nothing external. Hyperparameters: 32 latent units, episodes of 40 steps, batch
64, Adam lr 3×10⁻³, 3000 iterations, 3 seeds. Controller is a fixed scripted
policy (outside the predictor), as this protocol does not test agency learning.

The stop-gradient on `r` is a bootstrapped-target convention (the second-order
path is not back-propagated). It is documented and verified: `--grad-check`
compares the analytic BPTT gradient against finite differences **under the same
convention** (worst relative error ~10⁻⁷).

---

## 5. Metrics

All metrics are computed on held-out episodes with frozen weights.

| Metric | Definition | What it measures |
|---|---|---|
| `predErr` | $\mathbb E[(o_t-\hat o_t)^2]$ | overall prediction quality |
| `selfGain` | $\mathrm{MSE}_{\text{self ablated}} - \mathrm{MSE}_{\text{intact}}$ | **Self Prediction Gain** (SPG), absolute: outcome variance actually removed by self-referential state |
| `selfShare` | `selfGain / Var(o)` | SPG normalised; comparable across conditions |
| `SPR` | $1 - \mathrm{MSE}_{\text{intact}}/\mathrm{MSE}_{\text{ablated}}$ | relative form; **unstable when the error floor is near zero** — reported for completeness, never used alone |
| `selfEnc` | $R^2$ of $b_t$ decoded from $s_t$ (least squares, third-party probe) | does the slot *encode the agent's own hidden state*? (decodability, not use) |
| `selfAttr` / `extAttr` | accuracy of a logistic readout on $[s_t, r_t]$ labelling the dominant cause of the residual (self vs external); steps where neither cause dominates by 2× are dropped | is the internal state informative about **who** caused the error? |
| `RMG` | $1 - \mathrm{MSE}_{\text{meta}}/\mathrm{MSE}_{\text{base}}$, where the meta-predictor predicts $r_{t+1}$ from $[s_t, r_t, a_{t-1}]$ and the baseline from $[r_t, a_{t-1}]$ alone (least squares) | **Recursive Modelling Gain**: incremental value of the self state over the freshest self-referential signals |

**RMG does not separate C from A in this design, and we report that plainly.**
Diagnosis (measured, not guessed): the agent's own action and error history
*encode world state* — the action is a quantised function of the world, and the
residual keeps a small autocorrelated component left by the (linear, finitely
trained) world model. So "using my own history to predict my own future error"
buys a little *world* information in every condition. Removing the
action→world coupling does not help (RMG in A: 0.108 → 0.140). A metric that
separates self-reference from world-memory is still an open problem
(`../theory/QUESTIONS.md`, B2/B4). P2 is therefore tested by condition **D**
instead of by RMG.

Ablation semantics: `ablate_self` zeroes the slot both in the recurrence and in
the output, so the ablated model has no self channel at all.

---

## 6. Judgement criteria

Support for RSMEH requires **all** of:

1. `selfGain` in **C** exceeds `selfGain` in **A** and **B** by at least one
   order of magnitude (and `selfShare_C` ≳ 0.05);
2. `selfGain` ≈ 0 in **B** while `predErr_B` sits on the irreducible-noise floor
   (error pressure alone is not enough);
3. `selfEnc` in **C** is high (the slot actually encodes $b_t$);
4. **D behaves like A/B, not like C**: with the same self-caused error variance
   but no temporal structure, `selfGain ≈ 0` and `selfEnc ≈ 0`. If D behaves
   like C, the effect is about error *magnitude*, the "self model" is just a
   hidden-variable estimator, and the terminology of the hypothesis must be
   revised.
5. **E behaves like A/B** as well: the largest error of all conditions, with no
   self-causal relevance, must not produce a self model either.

Falsification / weakening:

- `selfGain` roughly equal across A, B, C → the "self-caused" condition is not
  what drives it; the hypothesis loses its specific claim.
- **D ≈ C** → the trigger is magnitude, not structure (see criterion 4).
- `selfEnc` high in **B** → "self" encoding appears without self-caused error.
- Any condition in which an external-only model reaches the same `predErr` as
  the self-using model (test by `--b-ablation`, feature withholding).

**P2 (recursion) is not settled by this experiment.** The precondition that P2
requires — the self-caused quantity is temporally structured, hence a model of
the model is *possible* — is measured cleanly by `selfEnc` (0.89 in C vs 0.004
in D). Whether an actual second-order structure *forms* is left open; `RMG`
fails to separate the conditions (§5) and a valid recursion metric is an open
problem.

---

## 7. Anti-cheating rules

1. **No metric by saturation.** Report whether the metric *distinguishes the
   manipulation*, not merely whether it is non-zero. (Violated by our own v1;
   see `../references/README.md`.)
2. **No architecture labels.** A module called "self" proves nothing; the
   claim must survive ablation. Our slot is generic by construction.
3. **Third-party probes must be declared.** `selfEnc` and `selfAttr` use
   generator labels / hidden variables. They measure **decodability**, not that
   the agent uses the information. Results must always be reported with that
   caveat.
4. **No seed shopping.** Report mean over seeds; report spread.
5. **Matching error magnitude.** Where a claim involves "controllability" or
   "self-causation", error magnitude must be matched across conditions, or the
   imbalance must be stated explicitly (condition C's `predErr` is *not*
   magnitude-matched to B — this is declared as a limitation).

---

## 8. Reproducibility

```bash
python experiments/code/simulation.py                                # main table, 3 seeds
python experiments/code/simulation.py --grad-check                   # gradient verification
python experiments/code/simulation.py --b-ablation                   # B/C with world features withheld
python experiments/code/simulation.py --iters 3000 --seeds 1,2,3     # as reported
```

Deterministic given seeds. Environment: Python 3.11, NumPy only. The world's
structural weights are fixed by `WORLD_SEED = 20260919`.

---

## 9. What this toy cannot show

- It cannot show that anything is *experienced*. Level 5 is out of scope by
  design; the strongest outcome here is Level 3–4 on the ladder in
  `../theory/THEORY.md`.
- The slot is a generic recurrent latent, but the **training objective is
  hand-designed**; this is not a demonstration of spontaneous architectural
  growth in an unconstrained learner.
- `b_t` enters the outcome **additively**. A multiplicative self-coupling is
  left for future work (it changes the learnability of the self term, and the
  effect of that change is itself an open question).
- Statistical power is limited (3 seeds, toy scale). This protocol is designed
  to be *falsifiable and cheap to attack*, not to be conclusive.

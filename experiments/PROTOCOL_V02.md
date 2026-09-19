# RSMEH v0.2 — Temporal Self Model Emergence (protocol)

**Status:** v0.2 · companion to `PROTOCOL.md` (v0.1) and `../theory/THEORY.md`
**Code:** `v02/` (PyTorch; CPU is enough)

---

## 1. What v0.2 is for

v0.1's negative result: *"a latent encodes a hidden variable"* does **not** mean
*"a latent encodes the self"* — a world model and a self model look identical
from the inside (file `PROTOCOL.md` §2, and `../experiments/experiment_results.md` §2).

v0.2 attacks the same question along the **time axis**:

> Is there a representation that stays **continuous across time** — and does
> continuity belong to the *self*, or merely to *persistence*?

The hypothesis under test (Temporal Self Continuity):

> When a system must explain how **its own past** constrains its future, a stable
> self representation may appear as an information-compression structure that
> spans time: `past → me → future`.

## 2. Three corrections to the first draft (measured, not stylistic)

The circulated v0.2 draft contained three things that would have produced
misleading results. Each is fixed here, and each fix has a **measurable
consequence** that is reported in the results file.

**(a) The hidden self state was a deterministic function of observable actions.**
The draft had `internal_state += action * 0.05`. Any latent that tracks such a
state is just computing *the consequences of my actions* (Stage 2, agency) — the
same trap as v0.1's prototype. **Fix:** the hidden state has autonomous
dynamics (`AR(0.92)` drift + innovation) *in addition to* the action term, so it
cannot be reconstructed from the action history; only the agent's own error
history carries information about it.

**(b) `L_identity = ||z_t − z_{t−1}||` has a degenerate optimum `z ≡ const`.**
A constant latent is *maximally* "stable" and carries nothing — this loss (and
the metric `mean_t ||z_{t+1} − z_t||`) **rewards a dead self**. **Fix:** an
*anti-degenerate* pair of numbers. Continuity is only claimed when
`separability` (is my own next step closer than a foreign trajectory's?) is well
above chance **and** `info` (can the hidden variable be decoded from the latent?)
is high. A contrastive objective replaces the naive penalty, and all three
objectives (`none` / `naive` / `contrastive`) are compared empirically in the
results file.

> **Measured correction to our own criticism.** The degeneracy is a true
> statement about the loss's global optimum, but we could **not** exhibit it:
> even at λ = 20 the naive penalty leaves the latent fully informative
> (`info` 0.942) while its drift collapses to 0.065 — because the self-caused
> variable in T1 is *slow*, a nearly static representation suffices. So the
> danger is **regime-dependent** (it should bite when the self state moves fast
> relative to the smoothing pressure), and the argument for the contrastive
> objective is not "the naive one collapses" but "it yields a much sharper
> identity (`sep` 0.950 vs 0.677) at no prediction cost". Reported as measured in
> `experiment_results.md` §6.

**(c) The counterfactual test had no measurable criterion — and no state
dependence.** As first written ("show the latent two actions"), the branch
difference is a *constant* that a memoryless model can guess. **Fix:** (i) the
action's effect on the world is **modulated by the hidden variable**
(`+ 0.6·h·a`), so "the same action means something different depending on the
state I am in" and the counterfactual difference carries self-information;
(ii) the test is a concrete metric — the relative error of the *predicted*
branch difference vs the *true* one, at each horizon.

## 3. Worlds (identical statistics, one causal difference)

| world | hidden variable | dynamics | responds to MY action? |
|---|---|---|---|
| **T1** `self_persistent` | **the agent's own state** | AR(0.92), innovation 0.06 | **yes** (gain 0.25) |
| **T2** `self_white` | the agent's own state | white, **variance-matched** | yes (gain 0.25) but no temporal structure |
| **T3** `external_persistent` | **a hidden world drive** | AR(0.92), innovation 0.06 | **no** (gain 0) |

Outcome: `pos_{t+1} = 0.9·pos_t + 0.1·a_t + 0.5·h_t + 0.6·h_t·a_t + 0.1·ξ`.
The observation is the position only; the hidden variable is never observed.

**T1 vs T3 is the crux.** They have the same statistics, the same
unobservability and the same effect on outcomes. They differ *only* in causal
position: whether the agent's own action moves the hidden variable. If every
self-related metric lights up in both, then those metrics measure *persistence*,
not *selfhood*.

## 4. Agents (same readout capacity; only the carrier of state differs)

| agent | state carried across time |
|---|---|
| `Memory` | persistent latent `z_t = tanh(f(obs_t, own last action, own last error) + W·z_{t-1})` |
| `NoPersist` | same latent width, re-created every step (**no** continuity) |
| `NoLatent` | no latent at all — the "Error(World)" baseline |

Task: at each step, forecast the position at horizons **1, 5, 10** under each
*stated* held action `c ∈ {−1, +1}`, with the **same exogenous noise** as the
factual trajectory. The action's own error (one-step prediction for the action it
actually took vs what it observed) is the only self-referential input beyond its
own actions.

## 5. Metrics

| metric | definition | why it is here |
|---|---|---|
| `errH1/5/10` | prediction error per horizon | the core task |
| `gainH*` | `err` of the no-latent baseline minus `err` | **does continuity buy anything?** |
| `drift` | `mean‖z_{t+1}−z_t‖ / mean‖z_t‖` | reported **only** next to `sep`; a dead latent scores 0.00 |
| `sep` | `P(own next step closer than a foreign trajectory's)` — 0.5 = chance | anti-degenerate continuity test |
| `info` | `R²` decoding the hidden variable from `z` | is the latent informative at all? |
| `cfErr@H`, `cfCorr@H` | relative error / cosine between predicted and true branch difference | **counterfactual accuracy** |
| `true Δ hidden @H` | ground-truth divergence of the hidden variable between branches | the causal fingerprint (analysis script) |

## 6. Judgement criteria

**Support for the temporal-self claim requires all of:**

1. `gainH10(Memory) > gainH10(NoPersist) > 0` in **T1** — continuity must pay,
   specifically at long horizons;
2. `sep` well above 0.5 **and** `info` high in T1 — the latent is continuous
   *and* informative (no dead self);
3. `cfErr@H10(Memory) < cfErr@H10(NoPersist) < cfErr@H10(NoLatent)` in T1 — the
   self state is needed to forecast counterfactuals;
4. **T2 gives no gain from continuity** (nothing persistent to carry);
5. the `naive` identity loss is shown to *not* be better than `contrastive`.

**Falsification / weakening:**

- `gainH10(Memory) ≈ gainH10(NoPersist)` in T1 → continuity is not the mechanism.
- `sep` high but `info` ≈ 0 → the latent is a dead (constant) self: the metric
  pair fails the anti-degenerate test and the claim must be withdrawn.
- **`gainH10(Memory) ≫ gainH10(NoPersist)` in T3 as well** → continuity is
  valuable for tracking any persistent hidden cause, self or world; the phrase
  "temporal **self**" would then be unjustified (this is the analogue of v0.1's
  alignment-vs-self failure, and is the most likely outcome — see results).

## 7. Reproducibility

```bash
python experiments/v02/train_v02.py --iters 1200 --seeds 1,2,3                 # as reported
python experiments/v02/train_v02.py --iters 1200 --seeds 1,2,3 --deterministic  # bit-identical re-runs
python experiments/v02/train_v02.py --only T1_self_persistent --identity-loss naive --lam 20
python experiments/v02/analysis/identity_test.py --world T1_self_persistent
python experiments/v02/analysis/counterfactual_test.py
```

PyTorch CPU training is **not bit-reproducible by default**: reduction order across
threads shifts the last digits between runs (we measured one row moving from
`errH1` 0.2939 to 0.3445 at 25 iterations). `--deterministic` pins a single thread
to get bit-identical re-runs. The numbers in `experiment_results.md` were produced
**without** that flag; every reported effect is a factor of 3–4 (e.g. `errH10`
0.538 vs 1.857), so the thread-order noise does not affect any conclusion.
`identity_test.py` / `counterfactual_test.py` are likewise deterministic given
their seeds.

---

## 8. What v0.2 does NOT do

- **No recursion.** "I predict myself predicting" (Stage 5) is v0.3. v0.2 only
  tests continuity and counterfactual self-modelling.
- **No experience claim.** Still Levels 2–4 on the ladder of `../theory/THEORY.md` §5.
- **Additive/compositional toy worlds only** — no embodiment, no vision, no
  language, no social interaction (Phase 2–4 of `README.md`).
- **The agent cannot observe the causal origin** of the hidden variable, so no
  behavioural metric in this design can decide whether the representation is
  "really" a self. That is a claim about the world's causal graph, not about the
  agent's evidence. v0.2 therefore cannot, by construction, settle "is it a
  self?" — it can only characterise what continuity buys and when.

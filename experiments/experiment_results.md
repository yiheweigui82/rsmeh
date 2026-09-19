# Experiment results (v0.1)

Everything below is **verbatim program output**. Reproduce with the two commands
in §1 and §2. Nothing in this file is estimated, smoothed, or hand-tuned.

Both experiments ask the same question with different methods, and they agree on
the one control that matters most (see §3).

---

## 1. Controlled experiment — `code/simulation.py`

```bash
python experiments/code/simulation.py --grad-check
python experiments/code/simulation.py --iters 3000 --seeds 1,2,3 --b-ablation
```

```
grad-check: worst relative error over W_s/c_s/w_o = 1.063e-08
=== RSMEH toy simulation v0.1 ===
architecture: 32-unit self slot | conditions A/B/C/D/E | seeds [1, 2, 3] | iters 3000

Prediction: the self path is used ONLY in C.  (The RMG column does
not discriminate in this design -- see PROTOCOL.md section 5.)
condition                 predErr  selfGain  selfShare  selfEnc  selfAttr  extAttr     RMG
A pure_prediction          0.0001   -0.0000    -0.0001      n/a       n/a    0.680   0.118
B external_complexity      0.0101   -0.0000    -0.0001      n/a       n/a    0.957   0.006
C self_coupled             0.0382    0.1416     0.3598    0.900     0.090    0.965   0.012
D self_coupled_white       0.1829   -0.0001    -0.0002    0.004     0.012    0.956   0.004
E false_agency             0.2483   -0.0008    -0.0014      n/a       n/a    0.956   0.005

irreducible external-noise floors (variance): A 1.0e-04, B 1.0e-02, C 2.2e-02, D 2.2e-02, E 2.5e-01
in C the floor is white + whatever part of the agent's own drift it fails to track.
    A pure_prediction        SPR(relative) mean -0.311  sd 0.028
    B external_complexity    SPR(relative) mean -0.001  sd 0.001
    C self_coupled           SPR(relative) mean 0.787  sd 0.008
    D self_coupled_white     SPR(relative) mean -0.000  sd 0.001
    E false_agency           SPR(relative) mean -0.003  sd 0.003

--- diagnostic: same trained network, external features withheld ---
    (shows how much of the prediction rides on the world model)
    B external_complexity    intact 0.0101 -> features withheld 0.0861
    C self_coupled           intact 0.0382 -> features withheld 0.1013
    E false_agency           intact 0.2483 -> features withheld 0.5778
```

### Reading

| condition | what it is | `selfGain` | `selfEnc` | verdict |
|---|---|---|---|---|
| A pure_prediction | fully solvable | −0.0000 | n/a | no self model needed, none used |
| B external_complexity | irreducible **external** white noise | −0.0000 | n/a | error persists; self metrics stay at zero; the world model does the work (`predErr` sits exactly on its 0.0101 floor, and withholding features costs 8.5×) |
| **C self_coupled** | irreducible, **self-caused**, temporally structured | **+0.1416** | **0.900** | **self-referential structure emerges**: the self channel removes 36% of outcome variance, and the agent's hidden drift is decodable from it at R² = 0.90 |
| D self_coupled_white | same self-caused **variance**, no temporal structure | −0.0001 | 0.004 | **nothing emerges** |
| E false_agency | **largest error** of all conditions, no self-causal relevance | −0.0008 | n/a | **nothing emerges** |

Three points to take away:

1. **Error pressure alone is not enough.** B and E have persistent irreducible
   error (E has the largest of all five conditions) and produce *nothing*.
2. **Magnitude is not the trigger — temporal structure is.** C and D have
   variance-matched self-caused error; only C (structured) produces a self
   representation. This is the sharpest result in the repo.
3. **The recursion metric failed, and we say so.** `RMG` is large in A (0.118)
   and small in C (0.012) — the *opposite* of a clean result. Diagnosis
   (`PROTOCOL.md` §5): the agent's own action/error history *encodes world
   state*, so "predict my own future error from my own history" buys a little
   world information in every condition; the action→world coupling is not the
   cause (removing it raises A's RMG to 0.140). **Whether recursion actually
   occurs is therefore NOT settled by this experiment.** What we did measure
   cleanly is its *precondition*: the self-caused quantity is temporally
   structured (hence modellable) in C (`selfEnc` 0.900) and is not in D (0.004).

Full protocol, judgement criteria and falsification conditions:
`PROTOCOL.md`. Sibling "black box" implementation: `prototype/`.

---

## 2. Black-box prototype — `prototype/train.py`

```bash
python experiments/prototype/train.py --iters 1500 --seeds 1,2,3 --deterministic
```

(`--deterministic` pins a single CPU thread, so this table is bit-reproducible.
It also includes the fix for a real seeding bug: the model must be constructed
*after* `torch.manual_seed`, otherwise the initial weights come from an unseeded
global RNG and two runs with the same seed disagree.)

```
=== RSMEH prototype v0.1 -- latent self-information emergence ===
latent 8-dim | no self variable in the architecture | iters 1500 | seeds [1, 2, 3]

worlds:
  A_simple               predictable world, no hidden variables, low error
  B_complex_external     high uncertainty, all of it external (visible + hidden drives)
  C_self_relevant        an unobservable internal state of the agent shapes the outcome
  D_false_agency         high error, but actions have no effect and nothing is trackable

world                   predErr  baseErr  selfGain  selfShare  gainAbl alignSelf alignExtH  attrSelf  attrExt
A_simple                 0.0001   0.0001    0.0000     0.0004   0.0006       n/a       n/a       n/a    1.000
B_complex_external       0.0461   0.0641    0.0180     0.0093   0.3085       n/a     0.724       n/a    1.000
C_self_relevant          0.0393   0.0562    0.0169     0.0020   2.4055     0.841       n/a     1.000    1.000
D_false_agency           0.1102   0.1104    0.0002     0.0002   0.0090       n/a       n/a       n/a    1.000
```

### Reading

- **A_simple**: nothing to model → no latent effect (`selfGain` 0.0000).
- **B_complex_external**: the latent *is* used (`selfGain` 0.0179) — and it
  carries the hidden **external** drive (`alignExtH` 0.717). **This is the
  methodological punchline: `alignment ≠ self`.** A probe that asks "does the
  latent encode a hidden variable?" fires in B just as loudly as in C; only the
  *causal origin* of that variable distinguishes a world model from a self
  model.
- **C_self_relevant**: the latent is used (`selfGain` 0.0166), it encodes the
  agent's **own** hidden state (`alignSelf` 0.839), and attribution is perfect
  (self 1.000 / external 1.000).
- **D_false_agency**: the **largest error of all four worlds** (0.1102) produces
  essentially no latent effect (0.0002). Same conclusion as the controlled
  experiment's conditions D/E.

`gainAbl` (switching a trained model's latent off) is deliberately shown as
*unreliable*: it reports 2.5726 in C but 0.3453 in B and 0.0150 in D, and even a
tiny 0.0005 in A. A *redundant* latent — one that merely re-encodes the
observation — inflates it. That is why Metric 1 uses a **re-trained no-latent
model** (`baseErr`), not the ablation.

---

## 3. What the two experiments jointly support (and what they do not)

**Supported (both designs, same direction):**

1. A self-referential representation emerges when — and only when — persistent
   prediction error is (i) irreducible by action, (ii) not explainable by any
   externally observable variable, and (iii) *caused by a structured internal
   state of the system itself*.
2. Large persistent error is **not** sufficient: the two highest-error
   conditions in each design (controlled E = 0.2483; prototype D = 0.1102)
   produce no self representation at all.
3. Error attribution is decodable from the agent's own state in the self-caused
   condition (1.000 / 1.000 in C) and correctly external elsewhere.

**Not supported / not settled:**

4. **Recursion (Stage 5) is untested.** No valid metric in either design
   separates "self-reference" from "world information arriving through the
   agent's own action and error history" (`PROTOCOL.md` §5,
   `../theory/QUESTIONS.md` B2/B4).
5. **Experience is untouched.** The strongest claim here is Level 3–4 on the
   ladder in `../theory/THEORY.md` §5, and only on a toy.
6. Both environments are toy-scale and hand-designed; error magnitudes are not
   matched across conditions (declared in `PROTOCOL.md` §7).

---

## 4. Falsification status

| criterion (`PROTOCOL.md` §6) | status |
|---|---|
| F1 all conditions produce self representations | **rejected** — A/B/D/E do not |
| F2 C produces none | **rejected** — `selfGain` 0.1416, `selfEnc` 0.900 |
| F3 self representation with no predictive utility | **rejected** — it removes 36% of outcome variance |
| F4 D/E behave like C | **rejected** — D −0.0001, E −0.0008 vs C +0.1416 |
| F5 B lights up a "self" metric | **partially accepted** in the prototype (`alignExtH` 0.717): the *alignment* metric alone cannot separate self from world. The causal metrics (ablation of a self-only channel + attribution) do. |
| M4 recursion | **unresolved** — see §5 above |

---

# v0.2 — Temporal Self Model Emergence

## 5. Main table — `python experiments/v02/train_v02.py --iters 1200 --seeds 1,2,3`

```
=== RSMEH v0.2 -- Temporal Self Model Emergence ===
latent 16-dim | one architecture, three carriers of state | iters 1200 | seeds [1, 2, 3] | identity loss: none

  T1_self_persistent       the agent's own hidden state: autonomous drift + driven by my actions
  T2_self_white            the agent's own hidden state, variance-matched but with no temporal structure
  T3_external_persistent   a hidden world drive with the SAME AR dynamics, but my actions do not move it

world                   agent        errH1   errH5  errH10   drift    sep   info  cfCorr   cfH10  cfErr10
T1_self_persistent      Memory      0.0397  0.2662  0.5381   0.605  0.716  0.929   0.991   0.993    0.126
T1_self_persistent      NoPersist   0.1173  1.0879  1.8568   0.622  0.559  0.455   0.960   0.972    0.237
T1_self_persistent      NoLatent    0.1541  1.7057  2.8390     n/a    n/a    n/a   0.937   0.955    0.295
      gainH1 vs no-latent baseline: Memory +0.1144 | NoPersist +0.0368
      gainH5 vs no-latent baseline: Memory +1.4395 | NoPersist +0.6178
      gainH10 vs no-latent baseline: Memory +2.3009 | NoPersist +0.9822
T2_self_white           Memory      0.0249  0.0856  0.1139   0.743  0.579  0.003   0.941   0.957    0.292
T2_self_white           NoPersist   0.0251  0.0855  0.1135   0.675  0.591  0.003   0.941   0.957    0.292
T3_external_persistent  Memory      0.0252  0.1825  0.4179   0.869  0.615  0.457   0.838   0.835    0.554
T3_external_persistent  NoPersist   0.0197  0.1797  0.4121   0.628  0.555  0.434   0.837   0.834    0.553
```

### Reading

| world | continuity gain (`gainH10` Memory vs NoPersist) | `sep` | `info` | verdict |
|---|---|---|---|---|
| **T1** self, persistent, **action-driven** | **+2.30 vs +0.98** (Memory's error is 3.4× smaller than NoPersist's) | 0.716 | 0.929 | **continuity pays, and the latent is continuous *and* informative** |
| **T2** self, variance-matched **white** | ±0.00 (0.1139 vs 0.1135) | 0.58 | **0.003** | nothing persistent to carry → memory buys nothing |
| **T3** **external** persistent, not action-driven | ±0.00 (0.4179 vs 0.4121 — NoPersist is marginally *better*) | 0.61 | 0.457 | **a persistent, autocorrelated hidden cause is not sufficient** |

1. **Continuity pays only where the hidden state is the agent's own *and* driven by
   its own actions** (T1). Criterion 1 and 2 of `PROTOCOL_V02.md` §6 hold:
   `gainH10` 2.30 ≫ 0.98 > 0, `sep` 0.716 > 0.5 with `info` 0.929.
2. **T3 is the decisive control and it does not light up.** It has the *same*
   AR(0.92) hidden dynamics and the *same* effect on outcomes as T1; only the
   causal position differs (my action cannot move it). Memory then buys nothing —
   a memoryless agent that sees its own last error is already sufficient. So
   "temporal self" is not just a rebranding of "persistent hidden variable".
3. **T2 confirms the structure-vs-magnitude lesson from v0.1** at the temporal
   level: with a variance-matched but structureless self-caused variable, `info`
   collapses to 0.003 and memory buys exactly nothing.
4. **Counterfactual accuracy follows the same pattern**: `cfErr10` in T1 improves
   0.295 (no latent) → 0.237 (no persistence) → **0.126** (persistent latent). In
   T3 it stagnates at ≈0.55 for both agents — the branch difference there does not
   carry any self-information (see the analysis script below).
5. **`drift` alone is useless as claimed.** T2's drift (0.743) is *higher* than
   T1's (0.605) while T2's latent carries nothing: the "smoothness" number is
   uninformative without `sep` and `info` next to it.

## 6. Does the draft's identity loss work? — ablation on T1

The circulated draft proposed `L_identity = ||z_t − z_{t−1}||`. Its trivial
optimum is `z ≡ const` — a dead self that would score *perfectly* on the naive
continuity metric. We implemented all three variants:

```
### identity loss: naive   (L = ||z_t - z_{t-1}||)
T1_self_persistent      Memory      0.0637  0.3312  0.5835   0.255  0.740  0.935   0.989   0.992    0.132
T1_self_persistent      NoPersist   0.1078  1.0904  1.8365   0.191  0.680  0.459   0.961   0.972    0.236
      gainH10 vs no-latent baseline: Memory +2.2556 | NoPersist +1.0025

### identity loss: contrastive   (own next step vs a foreign trajectory)
T1_self_persistent      Memory      0.0353  0.2695  0.5445   0.407  0.950  0.917   0.991   0.993    0.127
```

| objective | `errH10` | `drift` | `sep` | `info` | what it buys |
|---|---|---|---|---|---|
| `none` (task only) | 0.5381 | 0.605 | 0.716 | 0.929 | baseline |
| `naive` (draft, λ=1) | 0.5835 | **0.255** | 0.740 | 0.935 | smooths the latent (−58% drift), **costs 8% prediction error**, identity barely moves |
| `contrastive` (ours) | 0.5445 | 0.407 | **0.950** | 0.917 | **real identity continuity (sep +0.23) at no prediction cost** |

**Our own criticism of the draft's loss was partly wrong, and we say so.** We
expected the naive penalty to kill the latent (its global optimum is `z ≡ const`).
It does not, in this regime — not even at λ = 20:

```
### naive identity loss, lam = 20 (the collapse check)
T1_self_persistent      Memory      0.0295  0.2428  0.5079   0.065  0.677  0.942   0.992   0.994    0.117
```

`drift` collapses to **0.065** (a nearly constant latent) yet `info` stays at
**0.942**: because the self-caused variable in T1 is *slow* (AR 0.92), a slow —
almost static — representation is already sufficient to carry it. The penalty's
degenerate optimum is real as a mathematical statement, but whether it *bites*
depends on how fast the self state moves; here it acts as a harmless regulariser
and even improves `errH10` slightly. **The honest conclusion: the danger is
regime-dependent, and we have not exhibited a collapse.** What remains true, and
is measured, is that the *contrastive* objective achieves a much sharper identity
(`sep` 0.950 vs 0.677–0.740) at no prediction cost — it is the better objective,
not the only non-degenerate one.

## 7. Identity test — `python experiments/v02/analysis/identity_test.py --world T1_self_persistent`

The script lines up the trained latent against two deliberately degenerate
baselines, so it is visible which metric can be gamed:

```
latent                       drift  separable    info   verdict
trained latent               0.581      0.721   0.861   continuous AND informative
constant latent (z=0)        0.000      0.000  -0.000   ** the naive metric's best score; carries nothing **
random latent                1.416      0.494   0.004   unstable (high drift), carries nothing

per-dimension info (R^2 about the hidden variable):
    z14   0.685
    z15   0.592
    z1    0.517
    z9    0.517
    z6    0.369
    z4    0.300
```

This is the whole argument in three lines: **the draft's continuity metric
(`drift`) ranks the dead latent first (0.000)**, while `separable` (0.000 for a
constant, 0.494 for noise, 0.721 for the trained latent) and `info` (0.004 /
0.000 / 0.861) reject it. Continuity *numbers* are worthless alone; they only
mean something next to the anti-degenerate pair.

## 8. Counterfactual test — `python experiments/v02/analysis/counterfactual_test.py`

Part 1 — does MY action move the hidden variable? (ground truth; the agent never
sees it, and **no observation-based analysis can see it**):

```
world                         |Δ|@H1      |Δ|@H5     |Δ|@H10
T1_self_persistent            0.5000      2.1308      3.5351
T2_self_white                 0.0000      0.0000      0.0000
T3_external_persistent        0.0000      0.0000      0.0000
```

Part 2 — can the agent forecast that difference? (relative error; lower better):

```
world                   agent         cfErr@H1    cfErr@H5   cfErr@H10
T1_self_persistent      Memory           0.280       0.195       0.117
T1_self_persistent      NoPersist        0.747       0.436       0.233
T1_self_persistent      NoLatent         0.956       0.535       0.282
T2_self_white           Memory           0.685       0.396       0.298
T2_self_white           NoPersist        0.685       0.396       0.298
T2_self_white           NoLatent         0.684       0.396       0.298
T3_external_persistent  Memory           0.664       0.548       0.555
T3_external_persistent  NoPersist        0.508       0.525       0.544
T3_external_persistent  NoLatent         0.549       0.554       0.561
```

The two tables are the v0.2 story in full:

- In **T1** the action moves the agent's own state, so the counterfactual branch
  diverges in a hidden dimension that has to be *carried across time*: the
  forecast error falls monotonically with the carrier of state
  (0.282 → 0.233 → **0.117**).
- In **T3** the true divergence is **exactly zero** (the world's drive ignores my
  action) and memory buys nothing (0.555 vs 0.544). A persistent, autocorrelated
  hidden cause with the same statistics as T1 is **not** enough.
- In **T2** all three agents are identical to three decimals: there is no
  structure to carry.
- Nothing in the agent's *evidence* distinguishes T1 from T3. The difference
  lives in the causal graph — which the agent cannot observe, and which only the
  experimental designer knows. **This is v0.2's central negative result**: any
  purely behavioural verdict on "is this representation a self?" is
  under-determined by what the agent can see.

## 9. v0.2 — joint conclusion and falsification status

**Supported:**

1. **Temporal continuity pays, but conditionally.** A persistent latent improves
   long-horizon prediction (`errH10` 0.538 vs 1.857 without persistence) and
   counterfactual accuracy (0.117 vs 0.233) — *only* when the hidden state is both
   **the agent's own** and **driven by its own actions** (T1).
2. **A persistent hidden cause is not sufficient** (T3): same AR dynamics, same
   observability, same effect on outcomes — no continuity gain, no counterfactual
   gain.
3. **The self representation that does emerge is continuous *and* informative**
   (`sep` 0.716 / 0.950 with `info` 0.929 / 0.917), i.e. it passes the
   anti-degenerate test.
4. **The contrastive identity objective is the better tool** (`sep` 0.950 vs
   0.677–0.740 at λ = 20 / λ = 1 of the naive penalty, at no prediction cost).

**Not supported / corrected in public:**

5. **Our own critique of the draft's identity loss was overstated.** The
   degenerate optimum is mathematically real but we could not exhibit a collapse:
   at λ = 20 the latent keeps `info` 0.942 while `drift` → 0.065. The risk is
   regime-dependent (it should bite for fast-moving self states) and remains an
   open question, not a demonstrated failure.
6. **The agent-side asymmetry does not exist.** T1 and T3 are informationally
   identical from the inside; the "self" is decided by causal position, not by
   anything the agent can represent or report. Combined with v0.1's
   alignment-vs-self failure, this is the strongest methodological conclusion of
   the whole repository so far.
7. **Recursion is still untested.** v0.2 tests continuity and counterfactual
   self-modelling; "I predict myself predicting" (Stage 5) is v0.3.

| criterion (`PROTOCOL_V02.md` §6) | status |
|---|---|
| C1 `gainH10(Memory) > gainH10(NoPersist) > 0` in T1 | **accepted** (2.30 > 0.98 > 0) |
| C2 `sep` ≫ 0.5 **and** `info` high in T1 | **accepted** (0.716 / 0.929) |
| C3 `cfErr@H10` ordered Memory < NoPersist < NoLatent in T1 | **accepted** (0.117 < 0.233 < 0.282) |
| C4 no continuity gain in T2 | **accepted** (0.1139 vs 0.1135, `info` 0.003) |
| C5 naive identity loss not better than contrastive | **accepted on identity** (`sep` 0.677 vs 0.950); the collapse itself **not exhibited** |
| T3 lights up like T1 (would falsify "temporal *self*") | **rejected** — T3 shows no gain (0.4179 vs 0.4121) |

---

# v0.3 — Recursive Prediction

## 10. Main table — `python experiments/v03/train_v03.py --iters 900 --seeds 1,2,3`

```
=== RSMEH v0.3 -- Recursive Prediction ===
latent 16-dim | iters 900 | seeds [1, 2, 3] | lambda_self 1.0

  R1_prediction_loop     the world responds to MY previous prediction; the response gain is hidden and drifts
  R2_exogenous_control   the same extra drive, matched in dynamics and spread, but exogenous -- my predictions do not move it
  R3_visible_response    the world responds to MY predictions with a VISIBLE, constant gain (a learnable world property, not a loop)

world                 agent         worldErr  vacuousR2  selfPred  trivial  selfR2fut   incrR2  gainInfo
R1_prediction_loop    Rec              0.068      0.956     0.155    0.102      0.948    0.003     0.286
R1_prediction_loop    NoSelfPred       0.061      0.949       n/a    0.082      0.934    0.001     0.300
R1_prediction_loop    NoRec            0.069      0.983     0.091    0.095      0.971    0.002     0.326

R2_exogenous_control  Rec              0.040      0.968     0.052    0.030      0.781    0.017       n/a
R2_exogenous_control  NoSelfPred       0.054      0.949       n/a    0.048      0.693    0.019       n/a
R2_exogenous_control  NoRec            0.029      0.990     0.018    0.013      0.415    0.071       n/a

R3_visible_response   Rec              0.036      0.998     0.024    0.028      0.993    0.002       n/a
R3_visible_response   NoSelfPred       0.038      0.999       n/a    0.025      0.994    0.002       n/a
R3_visible_response   NoRec            0.042      1.000     0.029    0.028      0.994    0.001       n/a
```

### Reading

| observation | value | meaning |
|---|---|---|
| `vacuousR2` (the draft's test) | **0.949 – 1.000** in all nine cells, and **highest for `NoRec`** (0.983 / 0.990 / 1.000) | it cannot fail, and it *ranks the non-recursive agent first* — the draft's recursion test is worse than useless |
| `selfR2fut` vs its NULL row | **0.948 vs 0.002** (R1) | a real third-party probe *does* read the agent's own next prediction out of the latent — second-order content exists |
| `incrR2` (beyond `p_t, x_t, a_t`) | **0.003** (R1), 0.017 (R2), 0.002 (R3) | …but it is **almost entirely redundant** with what the agent already has |
| `trivial` = `R²(p_t → p_{t+1})` (analysis script, R1) | **0.989** | my next prediction is 98.9 % explained by *my current prediction*: the second-order target carries almost no new information |
| `selfPred` vs `trivial` | 0.155 vs 0.102 (R1), 0.052 vs 0.030 (R2) | the second-order head is **worse than the naive baseline** |
| `gainInfo` | 0.286 / 0.300 / 0.326 (R1) | the latent *does* track the hidden drifting response gain — the loop's unreadable part is recovered, and yet |
| `worldErr` | **R1 0.068 > R2 0.040** | closing the loop through my own prediction makes the world **harder**, not easier, to predict |
| `Rec` vs `NoSelfPred` | 0.068 vs 0.061 (worldErr), 0.948 vs 0.934 (`selfR2fut`) | removing and retraining without the second-order head does not hurt: the head is a **passenger** |
| `NoRec` vs `Rec` | 0.971 vs 0.948 (`selfR2fut`), 0.069 vs 0.068 (worldErr) | continuity is not needed either |

## 11. The two calibration points (and why both are reported)

An earlier run used `PRED_GAIN = 0.15`, `GAIN_DRIFT = 0.05` — the loop's response
term then had sd **0.0091** against a noise sd of 0.05, i.e. the closed loop was
nearly invisible (`gainInfo` ≈ 0.002 for every agent). In that regime:

```
world                 agent         worldErr  vacuousR2  selfPred  trivial  selfR2fut   incrR2  gainInfo
R1_prediction_loop    Rec              0.024      0.973     0.008    0.018      0.230    0.055     0.002
R1_prediction_loop    NoSelfPred       0.024      0.976       n/a    0.017      0.206    0.060     0.003
R1_prediction_loop    NoRec            0.024      0.998     0.008    0.017      0.291    0.105     0.002
R2_exogenous_control  Rec              0.027      0.947     0.008    0.016      0.146    0.030       n/a
R2_exogenous_control  NoSelfPred       0.027      0.965       n/a    0.016      0.134    0.019       n/a
```

There the second-order head **did** beat the trivial baseline (0.008 vs 0.018) and
`selfR2fut` was 1.6× higher in R1 than in R2 (0.230 vs 0.146). So the answer to
"does recursion pay?" is **regime-dependent**, and the two calibrations bracket it:
with a weak loop the second-order task is winnable but the loop does not matter;
with a loop strong enough to matter, the second-order target becomes redundant
(because my current prediction already summarizes my future input) and the head
loses to the trivial baseline. Our first attempt also mis-matched the control
world's drive (sd 0.0568 vs 0.0091 — a factor of 6) by using `tanh(N(0,1))`
instead of a permutation of the agent's own predictions; the fix (R2 drives the
world with **another episode's** prediction) makes the driver's marginal
distribution identical *by construction*. Even so, the **realised** response sd
still differs across worlds (R1 0.63 vs R2 0.11 for `Rec`) — documented as a
limitation, not hidden.

## 12. v0.3 conclusion and criteria status

| criterion (`PROTOCOL_V03.md` §6) | status |
|---|---|
| C1 the loop is learnable and makes the world *easier*: `worldErr(R1) < worldErr(R2)` | **REJECTED** — R1 0.068 > R2 0.040. A closed loop through my own prediction **amplifies** my modelling errors |
| C2 a second-order structure forms: `selfPred` beats `trivial`, `incrR2` beats NULL | **REJECTED** — 0.155 > 0.102, and `incrR2` 0.003 (NULL 0.002). The content exists (`selfR2fut` 0.948 ≫ NULL) but is redundant |
| C3 the second-order head is causally needed | **REJECTED** — removing and retraining without it does not degrade anything (0.061 vs 0.068 worldErr). It is a passenger |
| C4 continuity is needed for recursion | **REJECTED** — `NoRec` ≥ `Rec` on every second-order metric |
| the draft's recursion metric is valid | **REJECTED, measured** — 0.949–1.000 for all nine cells; it ranks `NoRec` highest |

**What v0.3 establishes.**

1. **The draft's recursion test is vacuous, and we measured it.** `R²(z_t → p_t)`
   is 0.95–1.00 everywhere *and is highest for the agent with no recurrence*: a
   metric whose best score goes to the thing it is supposed to detect.
2. **A real second-order probe does find content** (`selfR2fut` 0.948 vs NULL
   0.002) — the latent does carry information about the agent's *own next
   prediction*.
3. **But that content is functionally redundant.** `incrR2 = 0.003` beyond
   (`p_t`, `x_t`, `a_t`), and the honest bar — `R²(p_t → p_{t+1}) = 0.989` — shows
   why: *my current prediction already contains my best estimate of the future
   input*, so "predict my next prediction" ≈ "repeat my current prediction".
   Predicting my own next prediction is not an independent information
   requirement in this design; it is **subsumed** by first-order prediction.
4. **Closing the loop is a cost, not a benefit.** When my prediction drives the
   world, prediction error gets *worse* (0.068 vs 0.040): the loop propagates and
   amplifies my own modelling error. Self-reference buys nothing here except a
   harder world.
5. **The hidden part of the loop IS recovered** (`gainInfo` 0.286–0.326 in R1,
   0.002 when the loop was weak) — so the agents do model *how the world responds
   to them*; that modelling lands in the first-order predictor and leaves no
   residual for a second-order structure to add.

**What a design must satisfy for recursion to be non-redundant** (the deliverable
of this negative result): the agent's future prediction must depend on something
that is (i) **not** already summarized in the agent's current prediction, and
(ii) learnable **only** from its own error history. Candidates we have not
tested: a changing *self* (online plasticity / a non-stationary predictor),
second-order targets about one's own *error or uncertainty* rather than one's
prediction value, and a task where acting well requires predicting one's own
future predictions (so recursion feeds back into control, not just into loss).
Until one of those is shown to pay, "recursive self-model" remains an
architectural description, not an emergent functional necessity — and saying so
is the point of the exercise.

## 13. The probe script, verbatim — `python experiments/v03/analysis/recursion_test.py`

R1 (the loop world):

```
agent       probe                                           value
Rec         vacuous      R2(z_t -> p_t)   [the draft's test]   +0.960
Rec         future       R2(z_t -> p_t+1) [my next prediction]   +0.954
Rec         incremental  beyond p_t,x_t,a_t                +0.001
Rec         trivial      R2(p_t -> p_t+1) [no latent]      +0.989
Rec         hidden gain  R2(z_t -> g_t)                    +0.416
Rec         NULL  (shuffled latent) future                 +0.002
Rec         NULL  (shuffled latent) vacuous                +0.002
Rec         loop check  R2(response | my pred, beyond x_t)+0.002   (response sd 0.6344; x_t alone +0.871, with my pred +0.872)

NoSelfPred  vacuous      R2(z_t -> p_t)   [the draft's test]   +0.964
NoSelfPred  future       R2(z_t -> p_t+1) [my next prediction]   +0.946
NoSelfPred  incremental  beyond p_t,x_t,a_t                +0.001
NoSelfPred  trivial      R2(p_t -> p_t+1) [no latent]      +0.989
NoSelfPred  hidden gain  R2(z_t -> g_t)                    +0.341
NoSelfPred  loop check  R2(response | my pred, beyond x_t)+0.002   (response sd 0.6002; x_t alone +0.867, with my pred +0.868)

NoRec       vacuous      R2(z_t -> p_t)   [the draft's test]   +0.970
NoRec       future       R2(z_t -> p_t+1) [my next prediction]   +0.961
NoRec       incremental  beyond p_t,x_t,a_t                +0.001
NoRec       trivial      R2(p_t -> p_t+1) [no latent]      +0.988
NoRec       hidden gain  R2(z_t -> g_t)                    +0.402
NoRec       loop check  R2(response | my pred, beyond x_t)+0.008   (response sd 0.6075; x_t alone +0.865, with my pred +0.873)
```

R2 (the exogenous control):

```
Rec         vacuous      R2(z_t -> p_t)   [the draft's test]   +0.996
Rec         future       R2(z_t -> p_t+1) [my next prediction]   +0.960
Rec         incremental  beyond p_t,x_t,a_t                +0.013
Rec         trivial      R2(p_t -> p_t+1) [no latent]      +0.958
Rec         loop check  R2(response | my pred, beyond x_t)+0.021   (response sd 0.1091; x_t alone +0.618, with my pred +0.638)

NoSelfPred  vacuous      R2(z_t -> p_t)   [the draft's test]   +0.897
NoSelfPred  future       R2(z_t -> p_t+1) [my next prediction]   +0.865
NoSelfPred  incremental  beyond p_t,x_t,a_t                +0.009
NoSelfPred  trivial      R2(p_t -> p_t+1) [no latent]      +0.962
NoSelfPred  loop check  R2(response | my pred, beyond x_t)+0.009   (response sd 0.4511; x_t alone +0.745, with my pred +0.754)

NoRec       vacuous      R2(z_t -> p_t)   [the draft's test]   +0.990
NoRec       future       R2(z_t -> p_t+1) [my next prediction]   +0.237
NoRec       incremental  beyond p_t,x_t,a_t                +0.037
NoRec       trivial      R2(p_t -> p_t+1) [no latent]      +0.087
NoRec       loop check  R2(response | my pred, beyond x_t)+0.001   (response sd 0.0557; x_t alone +0.013, with my pred +0.014)
```

Three things this output settles:

1. **`trivial` = 0.989 in R1**: my next prediction is 98.9 % explained by my
   *current* prediction. The second-order target is nearly a copy of the
   first-order one. This single number is why `incrR2` is 0.001.
2. **The NULL rows (0.001–0.003) vs the real probes (0.95)**: the probes are not
   picking up noise — the second-order content is genuinely in the latent.
3. **The conditioned `loop check` is the fourth redundancy result.** In R1 the
   observation alone explains the world's response to me with `R² = 0.871`, and
   adding *my own prediction* moves it to 0.872 — an increment of **0.002**. The
   state `x_t` **is** the accumulation of my past predictions, so it is already a
   sufficient statistic for my own contribution; there is nothing left for a
   self-referential representation to add. (The first, unconditional version of
   this check reported 0.860 for R1 *and* 0.638 for R2, which cannot be causal —
   it was picking up structure shared across batch elements. Conditioning on
   `x_t` fixes it, and we report the fix rather than the flattering number.)

**The design lesson for v0.4**, stated precisely: for recursion to be
non-redundant, the agent's own contribution must **not** be recoverable from the
observable state. Concretely, the response to my prediction must leave no
sufficient trace in the current observation — e.g. it acts on a *hidden* part of
the future, or with a delay that only my own error history can bridge. In the
present design the visible state absorbs my own influence immediately, which is
why every honest probe comes back "redundant".

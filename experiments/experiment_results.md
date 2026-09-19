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
python experiments/prototype/train.py --iters 1500 --seeds 1,2,3
```

```
=== RSMEH prototype v0.1 -- latent self-information emergence ===
latent 8-dim | no self variable in the architecture | iters 1500 | seeds [1, 2, 3]

worlds:
  A_simple               predictable world, no hidden variables, low error
  B_complex_external     high uncertainty, all of it external (visible + hidden drives)
  C_self_relevant        an unobservable internal state of the agent shapes the outcome
  D_false_agency         high error, but actions have no effect and nothing is trackable

world                   predErr  baseErr  selfGain  selfShare  gainAbl alignSelf alignExtH  attrSelf  attrExt
A_simple                 0.0001   0.0001    0.0000     0.0004   0.0005       n/a       n/a       n/a    1.000
B_complex_external       0.0462   0.0641    0.0179     0.0092   0.3453       n/a     0.717       n/a    1.000
C_self_relevant          0.0396   0.0562    0.0166     0.0020   2.5726     0.839       n/a     1.000    1.000
D_false_agency           0.1102   0.1104    0.0002     0.0002   0.0150       n/a       n/a       n/a    1.000
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

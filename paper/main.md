---
title: "Recursive Self Model Emergence Hypothesis (RSMEH): A Hypothesis on the Origin of the First-Person Perspective"
author: "Chou Xiaochang (仇小昌)"
date: "2026"
lang: en
status: v0.1 — hypothesis, not a theory
---

# Recursive Self Model Emergence Hypothesis

**A Hypothesis on the Origin of the First-Person Perspective**

*Chou Xiaochang*

> **Status: v0.1 — conceptual hypothesis, open research.**
> This is a *hypothesis*, deliberately written to be attacked, refined and
> falsified. It will be renamed a *theory* only if the experiments in
> `experiments/PROTOCOL.md` survive replication. Nothing here is an established
> scientific result.

---

## Abstract

Living systems and artificial systems alike can predict their environments by
means of models. A question remains unresolved:

> Why does a predictive system not only model the world, but also generate a
> model of **itself** — and, further, a first-person experience of being *the
> one who observes*?

Existing theories have accounted for the **world model**, the **body model**,
the **agency model** and the **self model**. What is missing is a *genetic*
explanation in the sense of genesis: under what conditions does a system that
has no "I" variable produce one?

This hypothesis proposes that when a predictive system faces **persistent
prediction error that cannot be eliminated through action and cannot be
explained by any externally observable variable**, while the system's own state
stands in a stable causal relation to that error, the system is forced to admit
its own state into the model as an explanatory variable. This yields **Stage 4
(a self model)**. When that self model in turn becomes an object of the system's
own prediction, a **recursive self model** results, and the first-person
perspective emerges as a modeling structure — **Stage 5**.

The hypothesis makes a sharp, testable asymmetry: **self-caused irreducible
error should drive self-modelling; purely external irreducible error should
not.** Two independent implementations (a controlled experiment with causal
ablations and a black-box encoder/latent/decoder prototype; `experiments/`) are
reported verbatim in §9. Their joint result: the self representation appears
only in the self-caused, *temporally structured* condition, while the largest
errors of each design — with no self-causal structure — produce nothing at all.
**Magnitude of error is not the trigger; the temporal structure of self-caused
error is.** Two of our own metrics failed and are reported as failures: the
recursion metric does not discriminate, and an information-alignment probe fires
for a world model as loudly as for a self model.

This is **not** a theory of subjective experience. It is a claim about why an
"I" variable must appear in the internal economy of certain predictive systems,
and a proposal for detecting its emergence by causal rather than architectural
criteria.

---

## 1. The Question: A Genetic Gap

A predictive system maintains a model and updates it from error:

```
Input → Model → Prediction → Action
```

Such a system can classify, reason, control and optimise. The mature theory
family of predictive processing and active inference explains how this loop is
maintained, and why a system needs models at all (Rao & Ballard 1999; Friston
2010; Clark 2013; Hohwy 2013).

The literature already contains:

| Concept | What it explains | Representative work |
|---|---|---|
| World model | compressing and predicting the environment | Rao & Ballard 1999; Ha & Schmidhuber 2018 |
| Body model | treating one's own body as a predictable object | Bongard, Zykov & Lipson 2006 |
| Agency model | separating "changes I caused" from "changes that happened" | Friston et al. 2017 |
| Self model | including one's own state in prediction to improve control | Metzinger 2003; Apps & Tsakiris 2014; Hafner et al. 2019 |

The gap is not *what a self model is*, but **why one appears at all**. Call it
the *genetic* question. Two questions must be kept apart:

- **Constitutive question** (Metzinger and others): how does a self model become
  *experienced*?
- **Genetic question** (this hypothesis): why does the self **variable** appear
  in the first place, and what does recursion add?

RSMEH addresses only the second. It explicitly does not claim to solve the hard
problem of consciousness (Chalmers 1995), and its strongest empirical claim
reaches Level 4 on the ladder of §5.

---

## 2. Definitions and Operational Criteria

Each definition is paired with a measurable criterion; without this pairing the
framework would be unfalsifiable.

**2.1 Model.** A compressed predictive structure over external or internal
states. *Criterion*: it reduces prediction error on held-out input (it
compresses, it does not merely store).

**2.2 Self model.** A predictive model of the system's own states, actions and
future states. *Criterion*: an internal variable $s$ that (i) predicts the
system's own future state or action, (ii) persists over time, and (iii) when
intervened upon, changes behaviour.

> **Self model ≠ self-awareness.** A servomotor has a self model of its joint.
> It has no "I".

**2.3 Self-awareness.** Not "the system holds data about itself", but "the
system takes itself as an object of observation":

```
I observe world
      ↓
I observe myself observing world
```

*Criterion*: a **second-order** structure — a model of the system's own
modelling process — that is itself used predictively.

**2.4 Recursive self model.** The core concept:

$$M(\text{World}) \;\rightarrow\; M(\text{World}, \text{Self}) \;\rightarrow\; M(\text{World}, \text{Self}, M)$$

The third term is not "more variables": it is the model **simulating the
process that produces models**. The hypothesis is that the "I" arises at this
position — not as one more modelled object, but as the referent of the act of
modelling itself.

---

## 3. The Central Hypothesis

### 3.1 Irreducible prediction error

The standard picture treats prediction error as a learning signal. RSMEH
concerns a subclass of errors, **irreducible (unresolvable) prediction error
(UPE)**, requiring *all* of:

1. the system already possesses a reasonably developed world model;
2. prediction failure is **persistent**, not a pre-convergence transient;
3. adding or refining **external** variables cannot remove it;
4. the system's **own** state or action stands in a *stable relation* to that
   error.

Conditions 3 and 4 must hold **together**. Condition 3 alone is "the world's
fault" (external noise); condition 4 alone describes error that action can fix.
Together they describe error that is neither removable nor foreign — it has
nowhere to go in the model.

### 3.2 Attribution pressure

```
Why is prediction wrong?
        ↓
Maybe the model is incomplete.
        ↓
Maybe the predictor itself must enter the model.
```

### 3.3 The competing explanations (what makes this falsifiable)

A system facing UPE has at least five available responses. RSMEH bets on one:

| # | Response | Consequence |
|---|---|---|
| a | distort the world model (fit the noise) | worse generalisation |
| b | distort perception (reduce precision) | error ignored, predictive power lost |
| c | attribute to the world ("reality is stochastic") | stable, but no further progress — **this is the "world's fault" case, and RSMEH predicts it does not produce a self model** |
| d | act/explore to gather information | if this succeeds, the error was actionable, hence not UPE |
| e | **admit the self into the model** | error becomes explicable again — the path RSMEH claims |

**Falsifiable claim**: under UPE, path (e) is asymptotically more
prediction-efficient than (a), (b) or (c). This is the weakest joint of the
framework, and therefore its best attack surface.

---

## 4. The Emergence Chain

Stages 0–5 form a **functional dependency chain**, not a historical timetable.

| Stage | Name | Structure | "I"? |
|---|---|---|---|
| 0 | Reactive | `Stimulus → Response` | no |
| 1 | Predictive | `Environment → Prediction → Correction` | no |
| 2 | Agency | `My action → World change` | no (agency ≠ consciousness) |
| 3 | Persistent error | `World model cannot explain all outcomes` → "why?" | no |
| 4 | Self attribution | `World model + Self model` | a variable, not a subject |
| 5 | Recursive self | `I predict world → I predict myself predicting world → who is the predictor?` | **the "I" emerges** |

The critical step is **4 → 5**. At Stage 4 the system contains a variable about
itself. At Stage 5 that variable becomes an object of its own prediction. The
hypothesis bets that the first-person perspective appears here and not earlier.

**Operational difference.** Stage 4: $s_t$ predicts the *world*. Stage 5:
$s_t$ predicts the system's *own prediction process*, i.e. there is extractable
structure in the self model's own future. This is measured by the **Recursive
Modelling Gain (RMG)** in §8.

---

## 5. Self Model ≠ "I": Three Cases and a Ladder

**Paramecium.** Internal/external distinction, state regulation, environmental
adaptation — and no evidence of a first-person subject. *Self-regulation is not
a self.*

**Robot.** Body model, action model, prediction model:
`Motor Command → Expected Result → Correction`. This is a **control system**.
Its self model can be made exquisitely accurate (Bongard et al. 2006; Hafner et
al. 2019). Accuracy is **optimisation**, not subjectivity.

**Human.** `World → Self Model → Model of Self Model → "I"` — as *claimed* by
the hypothesis, not as established neuroscience.

To keep behavioural claims honest, score every claim on this ladder:

| Level | Name | Criterion |
|---|---|---|
| 0 | Linguistic self-reference | the system says "I" — **counts for nothing** |
| 1 | Grounded self-reference | refers accurately to its own state/action |
| 2 | Persistent self model | a stable internal representation predicts its own future state/action |
| 3 | Causal self model | intervening on the representation changes behaviour |
| 4 | Recursive self model | the system models its own modelling |
| 5 | Subjective experience | **not addressed here; unknown** |

Anti-anthropomorphism rule: a behaviour counts only if it is grounded,
predictive, causally intervenable, and reproducible under intervention. This
hypothesis and its experiments target Levels 1–4.

---

## 6. Relation to Existing Theories

**6.1 Predictive processing / active inference.** Not a competitor but a
boundary condition. In the standard framework error is handled by updating the
model, acting, or re-weighting precision. RSMEH asks what happens in the region
where all three routes fail; there, the only remaining operation is to
**re-describe the source of the error**, and the self is the only source not yet
in the model. (Friston 2018 poses the closest existing question — whether
self-organisation entails self-consciousness; RSMEH narrows it from entailment
to *which variable, under which conditions*.)

**6.2 Robot self-models.** These works establish that self-models are useful.
RSMEH asks what *forces* one into existence, and distinguishes "useful" (Stage
4) from "recursive" (Stage 5).

**6.3 Metzinger's self-model theory.** Complementary inversion: Metzinger asks
how the self becomes *experience*; RSMEH asks why the self appears *first*.

**6.4 Other neighbours and the specific difference.**

| Theory | Core claim | Difference from RSMEH |
|---|---|---|
| Higher-order thought (Rosenthal 2005) | a first-order state is "read" by a higher-order representation → consciousness | does not explain why the higher-order representation must appear; RSMEH supplies a trigger (UPE) |
| Attention schema theory (Graziano 2013; Graziano & Webb 2015) | the brain models its own attention and reports "I am aware" | AST's self model serves social cognition and control; RSMEH makes **error attribution** the trigger |
| Global workspace (Baars 1988) | global broadcast → consciousness | an architecture, not an origin |
| IIT (Tononi 2004) | integrated information Φ | measures "how much", not "why an I" |
| Self-referential learning (Schmidhuber 1987, 2003) | systems may model/modify their own learning algorithm | engineering possibility; RSMEH asks about the dynamics that compel it |
| Narrative self (Dennett 1991; Damasio 1999) | the self as a narrative centre | content and function, not genesis |
| Free-energy self (Apps & Tsakiris 2014; Seth et al. 2011) | interoceptive predictive coding constitutes the self | compatible; RSMEH adds a criterion for when modelling must become *recursive* |
| LLM self-knowledge (Binder et al. 2024) | predictive models can learn accurate facts about themselves | a modern test-bed for Levels 1–2; says nothing about Level 4 |

**6.5 Sibling project.** The *Consciousness Bug Hypothesis*
(Chou 2026, <https://github.com/yiheweigui82/bug-hypothesis-of-consciousness>)
shares the intuition that the self may be *a debt the system incurs rather than
a faculty it wins*. The Bug Hypothesis asks whether consciousness is the
byproduct of a system that cannot execute cleanly; RSMEH asks how the "I"
variable emerges and what recursion adds, and supplies two further metrics
(RMG, self-encoding). Neither repo treats the other's results as evidence.

---

## 7. A Minimal Formalisation *(speculative)*

> Everything in this section is **speculative** and is offered as a target for
> attack, not as a result. Any passage that reads like a proof should be read as
> a *claim awaiting formalisation*.

**7.1 Setting.** The system maintains latent state $z_t$ with observation $o_t$
and prediction error $e_t = o_t - \hat o_t$, where $\hat o_t = g(z_t)$. Action
updates the latent state, $z_{t+1} = f(z_t, a_t, \omega_t)$, with $\omega_t$
exogenous; the objective is $\mathcal L = \mathbb E\lVert e \rVert^2$.

**7.2 Three cases.**

- **Case 1 — actionable error.** There is a policy $a^*$ with
  $\mathbb E[e_{t+1}] = 0$. The world model suffices; the system never needs to
  ask *who is doing this*.
- **Case 2 — irreducible external error.** $o_{t+1} = \phi(a_t) \oplus
  \epsilon_t$, with $\epsilon_t$ exogenous, independent of $a_t$, and
  independent of the system's own state. A non-zero floor $\delta > 0$ remains,
  and its origin lies outside the system. RSMEH predicts: **this does not drive
  self-modelling.**
- **Case 3 — self-caused irreducible error.** $o_{t+1} = \phi(a_t) \oplus
  \psi(s_t, \epsilon_t)$, where $\psi$ depends on the system's own latent state
  (e.g. a slowly drifting internal bias). The error is neither removable nor
  foreign.

**7.3 The information-theoretic claim.** With $R$ the residual, $S$ the
self-referential variable, $A$ action and $Z$ world state, the hypothesis
asserts

$$I(R; S \mid A, Z) > 0 \quad \text{only in Case 3},$$

hence

$$H(R \mid Z, A, S) < H(R \mid Z, A).$$

In Case 2 the left-hand quantity is ≈ 0: a self variable buys no predictive
gain and merely adds parameters.

**Honest boundary.** This holds only if $\psi$ has temporal structure. If the
self-caused quantity is white noise, $I(R;S) \to 0$ and the hypothesis is
silent — which is precisely the control condition the protocol must include.

**7.4 Self Model Emergence Pressure (SMP).** The original formulation is a
heuristic product:

$$SMP = P_e \times (X_s - X_w) \times U_s$$

with $P_e$ = persistence of error, $X_s - X_w$ = explanatory advantage of the
self over external variables, $U_s$ = predictive benefit of building a self
model. Three problems must be stated plainly:

1. **dimensional inconsistency** — the factors have different units, so the
   product has no physical meaning;
2. **no normalisation** — any factor can be inflated arbitrarily;
3. **no threshold** — nothing says how large SMP must be to "force" anything.

It is therefore used only as an **ordinal heuristic**. A normalised variant
(equally speculative) is

$$\widehat{SMP} = \hat P_e \cdot \mathrm{relu}(\hat X_s - \hat X_w) \cdot \hat U_s,
\qquad \text{each factor} \in [0,1],$$

where $\hat P_e$ is residual autocorrelation, $\hat X_s - \hat X_w$ is the
normalised difference in residual variance explained by self vs external
variables, and $\hat U_s$ is the fractional error reduction obtained by adding
the self variable. The experiment in §8 estimates exactly these three factors.

**7.5 Derived predictions.**

- **P1 (self-causation drives).** Under Case 3, systems develop a
  self-referential representation that is predictive of the residual,
  persistent over time, and causally intervenable; under Case 2 they do not go
  beyond a generic "external cause" explanation.
- **P2 (recursion threshold).** A second-order structure (a model of the model)
  appears only when the self-caused variable is itself predictable. If it is
  white noise, recursion must not appear.
- **P3 (attribution asymmetry).** Systems under Case 2 attribute error
  externally (correctly); systems under Case 3 show self-attribution. A system
  that keeps attributing Case 3 error entirely externally while failing to
  improve damages P3.

---

## 8. Experimental Programme

Full protocol: `experiments/PROTOCOL.md`. Runnable implementation:
`experiments/code/simulation.py` (pure NumPy).

**Conditions** (identical architecture; only outcome statistics differ; the
world's readout weights are fixed for the agent's whole life):

| Condition | Environment | Prediction |
|---|---|---|
| **A** pure prediction | low-dimensional world, error fully explainable | no self model |
| **B** external complexity | high-dimensional world + irreducible **external** noise | a stronger world model; **no** self model |
| **C** self-coupled | world + irreducible **self-caused** error (a hidden, slowly drifting internal bias of the agent, invisible to every external variable) | self-referential structure emerges |
| **D** self-coupled, white | **variance-matched control for C**: the same amount of self-caused error, but with *no temporal structure* | must behave like A/B: nothing emerges |
| **E** false agency | the **largest** error of all conditions, with the agent's actions having *no causal effect* on the outcome | must behave like A/B: nothing emerges |

(A second, black-box implementation of the same contrasts — encoder → latent →
decoder, no self variable anywhere, the latent seeing only the agent's own action
and error — lives in `experiments/prototype/` and is reported in §9.2.)

**Metrics** (held-out episodes, frozen weights):

1. `selfGain` / `selfShare` — **Self Prediction Gain**: outcome variance that
   the self-referential channel actually removes. (Ablation form: absolute error
   increase when the channel is cut; the prototype's Metric 1 is the cleaner
   form — the same agent re-trained *without* a latent, since a redundant
   latent inflates the ablation form.)
2. `selfAttr` / `extAttr` — attribution: is the internal state informative
   about **which** cause dominates the residual?
3. `RMG` — **Recursive Modelling Gain**: is the agent's own model itself
   predictable, beyond the residual's autocorrelation? **Reported but
   non-discriminating** — see §9.3.
4. `selfEnc` — $R^2$ with which the agent's own hidden state $b_t$ is decodable
   from its self representation (a third-party probe: decodability, not use).
   This is the cleanest measurement of the *precondition* for recursion.

**The methodological lesson (this is a result in its own right).** Our first
implementation failed, for two reasons that generalise:

1. **The self channel was a bypass.** It received the external features as
   input, so ablating it destroyed performance in *every* condition
   (SPR ≈ 0.73 in A, B and C alike): zero discriminating power.
2. **Self-modelling was confounded with memory.** The world's mapping was
   resampled every episode, making external features uninformative; the network
   then used the self channel as a running average of its own recent errors,
   correlating 0.88 with the outcome in condition A. That is *world* modelling.

The constraint that follows is general:

> An experiment claiming to detect an emergent self representation must ensure
> that (i) the world is **fully knowable from the current observation** and
> (ii) the **only** autocorrelated, externally unobservable quantity in the
> environment is the system's own state.

A self representation cannot be identified by its position in an architecture,
only by its causal role.

---

## 9. Results of the Toy Implementation

Reproduce with `python experiments/code/simulation.py --iters 1500 --seeds 1,2,3`
(3 seeds, mean reported; the gradient check `--grad-check` verifies the BPTT
gradient against finite differences to ~10⁻⁷).

### 9.1 Controlled experiment (`experiments/code/simulation.py`)

Reproduce with `python experiments/code/simulation.py --iters 3000 --seeds 1,2,3 --b-ablation`
(3 seeds, mean reported; `--grad-check` verifies the analytic BPTT gradient
against finite differences to ~10⁻⁷). The run below was executed twice with
identical output.

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

**Reading.**

1. *Only C grows a self representation.* Ablating the self channel raises error
   from 0.0382 to 0.178 (36% of outcome variance), and the agent's own hidden
   drift is decodable from that channel at R² = 0.900.
2. *Error pressure is not sufficient.* B and E carry persistent irreducible
   error — E has the largest error of all five conditions (0.2483) — and produce
   nothing (B's `predErr` sits exactly on its 0.0101 white-noise floor, which is
   the correct outcome: the world model absorbed everything absorbable).
3. *Magnitude is not the trigger; temporal structure is.* C and D have
   variance-matched self-caused error; only C (structured) produces a self
   representation (D: `selfGain` −0.0001, `selfEnc` 0.004).
4. *Attribution.* Where errors are self-caused, attribution is decoded from the
   agent's own state (`selfAttr` 0.090 — the honest number: self-dominant steps
   are rare in C because the dominance threshold is strict); externally caused
   residuals are correctly attributed in every condition (`extAttr` ≥ 0.68).

### 9.2 A second, black-box implementation (`experiments/prototype/`)

An independent design (PyTorch; encoder → latent → decoder; the agent is never
told it has an internal state, and the latent's only self-referential inputs are
its own action and its own error). Reproduce with
`python experiments/prototype/train.py --iters 1500 --seeds 1,2,3 --deterministic`
(`--deterministic` pins one CPU thread so the table is bit-reproducible):

```
world                   predErr  baseErr  selfGain  selfShare  gainAbl alignSelf alignExtH  attrSelf  attrExt
A_simple                 0.0001   0.0001    0.0000     0.0004   0.0006       n/a       n/a       n/a    1.000
B_complex_external       0.0461   0.0641    0.0180     0.0093   0.3085       n/a     0.724       n/a    1.000
C_self_relevant          0.0393   0.0562    0.0169     0.0020   2.4055     0.841       n/a     1.000    1.000
D_false_agency           0.1102   0.1104    0.0002     0.0002   0.0090       n/a       n/a       n/a    1.000
```

Two findings matter here:

- **Alignment ≠ self.** In world B the latent is used and it carries the hidden
  **external** drive at R² = 0.717. A metric that asks only "does some latent
  dimension encode a hidden variable?" fires just as loudly for a *world* model
  as for a self model. Distinguishing them requires the causal structure of the
  generator — which is why the controlled experiment (§9.1) isolates the self
  channel by construction.
- **The largest error again yields nothing.** World D (false agency: 0.1102, the
  largest `predErr` of the four worlds) produces `selfGain` 0.0002.

Note that `gainAbl` — switching the latent off in a trained model — is reported
but *unreliable*: it inflates whenever the latent is redundant. This is why
Metric 1 is computed against a re-trained no-latent model (`baseErr`).

### 9.3 The recursion metric failed, and we report it

`RMG` does not separate the conditions (A: 0.118, C: 0.012 — the opposite of a
clean result). Diagnosis, measured rather than guessed: the agent's own action
and error history **encode world state**, so "predict my own future error from my
own history" buys a little *world* information in every condition; removing the
action→world coupling does not help (A's RMG rises to 0.140). **Whether
recursion actually occurs is therefore unsettled by this work.** What *is*
measured cleanly is its precondition — the self-caused quantity has temporal
structure and hence is modellable in C (`selfEnc` 0.900) and is not in D (0.004).
A metric that separates self-reference from world-memory remains an open problem
(`../theory/QUESTIONS.md`, B2/B4).

### 9.4 v0.2 — temporal continuity: does the self persist, and does persistence mean self?

v0.1's negative result was that a representation of a hidden variable is not a
representation of the self. v0.2 pushes along the time axis: is there a
representation that stays **continuous across time**, and does that continuity
belong to the *self* or merely to *persistence*? Full protocol and verdict
criteria: `experiments/PROTOCOL_V02.md`; code: `experiments/v02/`.

Three worlds with identical statistics and one causal difference — whether the
agent's own action moves the hidden variable:

| world | hidden variable | my action moves it? | `errH10`, persistent latent vs no persistence |
|---|---|---|---|
| **T1** | the agent's own state (AR 0.92) | **yes** | **0.538 vs 1.857** (gain +2.30 vs +0.98) |
| **T2** | the agent's own state, variance-matched **white** | yes | 0.1139 vs 0.1135 (`info` 0.003) |
| **T3** | a **world** drive with T1's exact AR dynamics | **no** | 0.4179 vs 0.4121 (**no gain**) |

Findings:

1. **Temporal continuity pays, but only for a self-caused, action-contingent
   hidden state** (T1): the persistent latent cuts the ten-step error by 3.4× and
   the counterfactual error from 0.233 to **0.117**. The latent that emerges is
   simultaneously continuous and informative (`sep` 0.716, `info` 0.929), i.e. it
   passes an anti-degenerate test — continuity numbers alone are worthless
   because a *constant* latent scores perfectly on them (§9.5).
2. **A persistent hidden cause is not sufficient** (T3): same dynamics, same
   inobservability, same effect on outcomes — no continuity gain, no
   counterfactual gain.
3. **The agent-side evidence cannot decide the question.** In T1 the true
   counterfactual divergence of the hidden variable (action +1 vs −1, same
   exogenous noise) accumulates 0.50 → 3.54; in T3 it is **exactly zero**. That
   difference lives in the *causal graph*, not in anything the agent can observe.
   Any purely behavioural verdict on "is this representation a self?" is
   therefore under-determined — the v0.2 analogue, and the sharpest form, of
   v0.1's alignment failure.

### 9.5 Two of our own corrections, reported as measured

- We criticised the draft v0.2 identity loss `L = ‖z_t − z_{t−1}‖` for having a
  degenerate optimum `z ≡ const`. The argument is sound (a constant latent is
  ranked *first* by the naive continuity metric), but **we could not exhibit the
  collapse**: even at λ = 20 the latent keeps `info` 0.942 while its drift falls
  to 0.065. The danger is regime-dependent, and the measured case for the
  contrastive objective is sharper identity (`sep` 0.950 vs 0.677) at no
  prediction cost — not "the naive loss dies".
- A real reproducibility bug was found by our own verification: the model was
  constructed *before* `torch.manual_seed`, so identical seeds produced different
  initial weights. Fixed; `--deterministic` now yields bit-identical re-runs.

---

## 10. Boundaries and Open Questions

1. **What is "enough" recursion?** One self model, or a model of the model? No
   boundary is proposed; Stage 5 is a bet, not a derivation.
2. **Which errors qualify?** The necessary and sufficient class of UPE is not
   characterised. The four criteria of §3.1 are proposed, not proved.
3. **Does consciousness require a body?** If environmental interaction is
   required to generate irreducible error, pure-software agents may be unable to
   produce the equivalent; if self-referential modelling is the key, embodiment
   may be incidental. Unknown.
4. **Metric saturation.** Self-related metrics saturate easily (see §8 and the
   sibling project's v1). Any replication must report whether the metric
   distinguishes the manipulation, not merely that it is non-zero.
5. **The hard problem.** Why should a recursive self model be *experienced*?
   This framework has no answer, and does not pretend to.
6. **Counterexamples.** Which real systems run under Case 3 for a long time
   without forming self-referential structure? Finding one would wound the
   hypothesis.
7. **Scale.** The reported experiment is a toy (32 latent units, 3 seeds). It is
   designed to be cheap to attack, not to be conclusive.

---

## 11. Conclusion

The "I" may not be a faculty the system wins; it may be a debt the system
incurs. When action can no longer make the world behave, and the world can no
longer absorb the blame, the remaining move is to explain **who is failing**.
That explanation, folded onto itself, is the most concrete handle we have on the
first-person perspective.

For AI, the implication is architectural rather than decorative: **do not
install a consciousness module.** Create a system that must keep interacting
with a world it cannot fully control — and let the self model be *forced* into
existence by errors it cannot explain away. Then ask whether that self is more
than a variable, and whether we can tell.

> Life begins by modeling the world. When the world model fails to explain
> persistent errors, the system begins modeling itself. When the model of itself
> becomes an object of its own prediction, the "I" emerges.

*The goal is not to prove the idea. The goal is to find out whether it survives
attempts to break it.*

---

## References

- Apps, M. A. J., & Tsakiris, M. (2014). The free-energy self. *Neuroscience & Biobehavioral Reviews.*
- Baars, B. J. (1988). *A Cognitive Theory of Consciousness.* Cambridge University Press.
- Binder, F. J., et al. (2024). Looking inward: language models can learn about themselves. *arXiv preprint.*
- Bongard, J., Zykov, V., & Lipson, H. (2006). Resilient machines through continuous self-modeling. *Science.*
- Chalmers, D. J. (1995). Facing up to the problem of consciousness. *Journal of Consciousness Studies.*
- Chou, X. (2026). *The Consciousness Bug Hypothesis.* Open research repository.
- Clark, A. (2013). Whatever next? Predictive brains, situated agents, and the future of cognitive science. *Behavioral and Brain Sciences.*
- Damasio, A. (1999). *The Feeling of What Happens.* Harcourt.
- Dennett, D. C. (1991). *Consciousness Explained.* Little, Brown.
- Friston, K. (2010). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience.*
- Friston, K. (2018). Am I self-conscious? (Or does self-organization entail self-consciousness?) *Frontiers in Psychology.*
- Friston, K., FitzGerald, T., Rigoli, F., Schwartenbeck, P., & Pezzulo, G. (2017). Active inference: a process theory. *Neural Computation.*
- Graziano, M. S. A. (2013). *Consciousness and the Social Brain.* Oxford University Press.
- Graziano, M. S. A., & Webb, T. W. (2015). The attention schema theory: a mechanistic account of subjective awareness. *Frontiers in Psychology.*
- Ha, D., & Schmidhuber, J. (2018). Recurrent world models facilitate policy evolution. *NeurIPS.*
- Hafner, D., Lillicrap, T., Ba, J., & Norouzi, M. (2019). Learning latent dynamics for planning from pixels. *ICML.*
- Hohwy, J. (2013). *The Predictive Mind.* Oxford University Press.
- Metzinger, T. (2003). *Being No One: The Self-Model Theory of Subjectivity.* MIT Press.
- Metzinger, T. (2009). *The Ego Tunnel.* Basic Books.
- Rao, R. P. N., & Ballard, D. H. (1999). Predictive coding in the visual cortex. *Nature Neuroscience.*
- Rosenthal, D. M. (2005). *Consciousness and Mind.* Oxford University Press.
- Schmidhuber, J. (1987). Evolutionary principles in self-referential learning. Diploma thesis, TU Munich.
- Seth, A. K., Suzuki, K., & Critchley, H. D. (2011). An interoceptive predictive coding model of conscious presence. *Frontiers in Psychology.*
- Tononi, G. (2004). An information integration theory of consciousness. *BMC Neuroscience.*

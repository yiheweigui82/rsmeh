# References & related work

An annotated map of the literature this hypothesis sits in, and of the
neighbouring projects we deliberately keep separate.

---

## 1. Predictive processing / active inference

- **Rao, R. P. N., & Ballard, D. H. (1999).** Predictive coding in the visual
  cortex. *Nature Neuroscience.* — the origin of predictive coding accounts.
- **Friston, K. (2010).** The free-energy principle: a unified brain theory?
  *Nature Reviews Neuroscience.* — perception and action under one objective.
- **Clark, A. (2013).** Whatever next? Predictive brains, situated agents, and
  the future of cognitive science. *Behavioral and Brain Sciences.*
- **Hohwy, J. (2013).** *The Predictive Mind.* Oxford University Press.
- **Friston, K., FitzGerald, T., Rigoli, F., Schwartenbeck, P., & Pezzulo, G.
  (2017).** Active inference: a process theory. *Neural Computation.*
- **Friston, K. (2018).** Am I self-conscious? (Or does self-organization
  entail self-consciousness?) *Frontiers in Psychology.* — closest existing
  question to ours; RSMEH narrows it from "entailment" to "which variable, under
  which conditions".

**RSMEH's add-on**: the standard framework treats irreducible error as a matter
of precision weighting. RSMEH asks what happens in the region where *perception,
action and precision adjustment all fail*, and argues that the remaining move is
a re-description of the error's **source**.

## 2. Self-models in machines

- **Bongard, J., Zykov, V., & Lipson, H. (2006).** Resilient machines through
  continuous self-modeling. *Science.* — a robot that learns a model of its own
  body and recovers from damage.
- **Lipson, H., & Pollack, J. B. (2000).** Automatic design and manufacture of
  robotic lifeforms. *Nature.*
- **Hafner, D., Lillicrap, T., Ba, J., & Norouzi, M. (2019).** Learning latent
  dynamics for planning from pixels. *ICML* (PlaNet); **Hafner et al. (2020)**
  Dreamer. *ICLR.*
- **Schmidhuber, J. (1987/2003).** Evolutionary principles in self-referential
  learning; Gödel machines. — systems that model or modify their own learning.

**RSMEH's add-on**: these works show a self-model is *useful*. RSMEH asks what
**forces** it into existence, and distinguishes "useful" (Stage 4) from
"recursive" (Stage 5).

## 3. Philosophical accounts of the self

- **Metzinger, T. (2003).** *Being No One.* MIT Press. — self-model theory of
  subjectivity: how a self-model becomes phenomenal.
- **Metzinger, T. (2009).** *The Ego Tunnel.*
- **Rosenthal, D. (2005).** *Consciousness and Mind.* — higher-order thought.
- **Dennett, D. (1991).** *Consciousness Explained.* — the self as a narrative
  centre of gravity.
- **Damasio, A. (1999).** *The Feeling of What Happens.* — the autobiographical
  self and core consciousness.
- **Hofstadter, D. (2007).** *I Am a Strange Loop.* — self-reference as the
  core of selfhood.
- **Chalmers, D. (1995).** Facing up to the problem of consciousness.

**RSMEH's add-on**: Metzinger (and the HOThs) ask how a self-model becomes
*experienced*. RSMEH asks why the self-variable appears **at all** — a genetic
question in the sense of "genesis", not of genetics. The hard problem is left
untouched, deliberately.

## 4. Other nearby frameworks

- **Graziano, M. (2013).** *Consciousness and the Social Brain*; **Graziano &
  Webb (2015).** The attention schema theory. — the brain models its own
  attention to explain its own awareness.
- **Baars, B. (1988).** *A Cognitive Theory of Consciousness.* — global
  workspace.
- **Tononi, G. (2004).** An information integration theory of consciousness.
  *BMC Neuroscience.*
- **Apps, M. A. J., & Tsakiris, M. (2014).** The free-energy self: a predictive
  coding account of self-recognition. *Neuroscience & Biobehavioral Reviews.*
- **Seth, A., Suzuki, K., & Critchley, H. (2011).** An interoceptive predictive
  coding model of conscious presence. *Frontiers in Psychology.*
- **Maturana, H., & Varela, F. (1980).** *Autopoiesis and Cognition.* —
  self-producing systems without a self-model.
- **Binder, F. J., et al. (2024).** Looking inward: language models can learn
  about themselves. *arXiv preprint.* — modern evidence that a predictive
  system can carry accurate self-knowledge; a possible test-bed for Levels 1–2.

## 5. Sibling project (same fuse, different question)

- **Chou, X. (2026).** *The Consciousness Bug Hypothesis* — an open research
  repository. <https://github.com/yiheweigui82/bug-hypothesis-of-consciousness>

| | Bug Hypothesis | RSMEH (this repo) |
|---|---|---|
| Question | is consciousness the byproduct of a system that *cannot* execute cleanly? | how does the "I" **variable** emerge, and what does recursion add? |
| Emphasis | Bug → explain the bug → create the "I" | the Stage 0→5 generative chain, with a bet on Stage 5 (recursion) |
| Metrics | self-modelling cost, error attribution | the same, **plus** recursive modelling gain (RMG) and self-encoding (selfEnc) |
| Shared | "the self may be a debt the system incurs, not a faculty it wins" | |

The two repos share examples, intuitions and a review pipeline, but each claim
must stand or fall on its own experiments. Neither repo treats the other's
results as evidence.

## 6. Methodological warning we are inheriting (and passing on)

- **Metric saturation.** Self-related metrics are trivially saturated: defining
  "self-reference" as "the model predicts its own action" yields 1.000 in every
  condition under a stable policy. Always ask whether a metric *distinguishes
  the experimental manipulation* (Bug Hypothesis, `experiments/code/simulation_v1.py`
  vs `simulation.py`).
- **Self vs memory confound.** Where the environment is autocorrelated, using
  one's own error history is legitimate *world* modelling. See
  `../experiments/PROTOCOL.md` §2.

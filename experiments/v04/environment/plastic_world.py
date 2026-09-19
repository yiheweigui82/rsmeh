"""v0.4 worlds -- a stationary-ish task whose difficulty sits in the AGENT.

v0.3 showed that "predict my own next prediction" is redundant when my next
prediction is a function of my current one (`incrR2` = 0.003, `trivial` R^2 =
0.989).  v0.4 needs a quantity that is

    (a) involved in my FUTURE computation,
    (b) NOT involved in my CURRENT output, and
    (c) unobservable,

so that only my own error history can reveal it.  Three worlds isolate that
quantity, the modulator `m_t` of the agent's own fast-weight learning rate:

    plastic   m_t is a hidden AR(1) process: my plasticity drifts, I cannot read it
    static    m_t == 0: no hidden plasticity at all (the falsifier: nothing to infer)
    visible   m_t is appended to the observation: my plasticity is readable, so
              nothing needs to be inferred (control for "hidden" vs "present")

The world itself is deliberately simple: a persistent state whose action gain
switches between two regimes every ~10 steps, so the agent must adapt *online* --
which is what gives the fast (plastic) pathway a job to do.  The difficulty is
supposed to be in the agent, not in the world.
"""

from __future__ import annotations

import numpy as np

MODES = ("plastic", "static", "visible")

RHO = 0.80
GAIN_LOW, GAIN_HIGH = 0.10, 0.60
SWITCH_EVERY = 10
NOISE = 0.10

M_RHO, M_INNOV = 0.95, 0.15          # the hidden modulator's AR(1) dynamics
KAPPA = 1.5                          # how strongly m_t changes the learning rate


class PlasticWorld:
    """One world; `mode` selects what happens to the modulator."""

    def __init__(self, mode: str = "plastic", seed: int = 0):
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; choose from {MODES}")
        self.mode = mode
        self.obs_dim = 2 if mode == "visible" else 1
        self.base_obs_dim = 1

    def reset(self, batch: int, rng: np.random.Generator):
        self.x = rng.normal(0.0, 0.2, size=batch)
        self.gain = np.where(rng.random(batch) < 0.5, GAIN_LOW, GAIN_HIGH)
        self.m = np.zeros(batch)
        self.t = 0
        self._rng = rng
        return self.observe()

    def observe(self):
        x = self.x.reshape(-1, 1)
        if self.mode == "visible":
            return np.concatenate([x, self.m.reshape(-1, 1)], axis=1)
        return x

    def step(self, action):
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        xi = self._rng.normal(0.0, 1.0, size=a.shape)
        self.x = RHO * self.x + self.gain * a + NOISE * xi
        self.t += 1
        if self.t % SWITCH_EVERY == 0:
            flip = self._rng.random(a.shape) < 0.5
            self.gain = np.where(flip, GAIN_LOW, self.gain)
            self.gain = np.where(~flip, GAIN_HIGH, self.gain)
        if self.mode == "static":
            self.m = np.zeros_like(self.x)
        else:
            self.m = M_RHO * self.m + M_INNOV * self._rng.normal(size=self.x.shape)
        return self.observe()

    # -- the quantity the agent may not read (unless `visible`) --------------
    @property
    def modulator(self):
        return self.m.copy()

    def plasticity_gain(self):
        """What multiplies my own learning rate during the NEXT update."""
        return 1.0 + KAPPA * self.m

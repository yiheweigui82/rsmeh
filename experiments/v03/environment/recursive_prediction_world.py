"""v0.3 worlds -- Recursive Prediction.

v0.2's lesson: a representation's continuity is not evidence of a self; what
makes a hidden state *mine* is its causal position (my own actions move it), and
that position is invisible to the agent.  v0.3 turns the same logic onto the
agent's own **prediction process**:

    R1_prediction_loop   the world's transition contains the agent's OWN previous
                         prediction, scaled by a hidden, slowly drifting gain
    R2_exogenous_control the same extra drive, same AR dynamics and the same
                         marginal spread, but EXOGENOUS: my predictions do not
                         move it (this is what makes R1 interpretable)
    R3_visible_response  the world responds to my predictions with a VISIBLE,
                         constant gain -- learnable as an ordinary world property

Why a *hidden* gain.  The agent trivially knows what it predicted a moment ago:
it produced that number and the trainer hands it back.  "Knowing my own previous
prediction" is therefore free and distinguishes nothing.  What is informative is
that *how the world responds to my predictions* is unknown and non-stationary:
that cannot be read off my own output, it has to be inferred from my own error
history -- the v0.1/v0.2 mechanism, now applied to the loop
prediction -> world -> observation -> prediction.

The loop is closed *step by step*, not generated in one shot: at step t the world
uses the prediction the agent made at t-1 to produce x_{t+1}.  That is what makes
the influence causal rather than decorative.

Self-prediction targets recorded by the trainer:

    pred[t]       the agent's world prediction made at t (for x_{t+1})
    pred_next[t]  the agent's world prediction made at t+1 (for x_{t+2})

`pred_next[t]` is the honest recursion target: at time t it has not been computed
and is not a function of the agent's current latent, so a probe for it can fail.
The circulated draft's `recursion_test` probed the *contemporaneous* prediction
instead -- which is a deterministic function of the latent, so R^2 ~ 1 for every
agent by construction and the test was vacuous.  Both are reported side by side
so the difference is visible.
"""

from __future__ import annotations

import numpy as np

WORLDS = ("R1_prediction_loop", "R2_exogenous_control", "R3_visible_response")

DESCRIPTIONS = {
    "R1_prediction_loop":
        "the world responds to MY previous prediction; the response gain is "
        "hidden and drifts",
    "R2_exogenous_control":
        "the same extra drive, matched in dynamics and spread, but exogenous -- "
        "my predictions do not move it",
    "R3_visible_response":
        "the world responds to MY predictions with a VISIBLE, constant gain "
        "(a learnable world property, not a loop)",
}

# world dynamics --------------------------------------------------------------
# Two calibration lessons from the first v0.3 run are baked into these numbers.
# (1) The loop must be strong enough to matter: with PRED_GAIN = 0.15 the
#     response term had sd 0.0091 against a noise sd of 0.05, i.e. the closed
#     loop was nearly invisible and there was no pressure to model it
#     (`gainInfo` ~ 0.002 for every agent).  It is now comparable to the noise.
# (2) The hidden gain must actually drift for tracking it to be worth anything:
#     GAIN_DRIFT 0.05 -> 0.12.
RHO = 0.85                 # state persistence
ACTION_GAIN = 0.20         # my action's direct effect
PRED_GAIN = 1.00           # response to my own previous prediction
GAIN_BAR = 0.60            # mean response gain
GAIN_DRIFT = 0.12          # sd of the gain's AR(1) innovation
GAIN_RHO = 0.95
NOISE = 0.05

OBS_NOISE_SD = 0.0         # observations are exact (no hidden observation noise)


class RecursivePredictionWorld:
    """One world.  `name` selects the causal structure and nothing else."""

    def __init__(self, name: str):
        if name not in WORLDS:
            raise ValueError(f"unknown world {name!r}; choose from {WORLDS}")
        self.name = name
        self.obs_dim = 1
        self.loop = name == "R1_prediction_loop"
        self.visible = name == "R3_visible_response"
        self.exogenous = name == "R2_exogenous_control"

    def reset(self, batch: int, rng: np.random.Generator):
        self.x = rng.normal(0.0, 0.3, size=batch)
        if self.loop:
            self.gain = np.full(batch, GAIN_BAR)
        else:
            self.gain = np.full(batch, GAIN_BAR)
        self._rng = rng
        return self._obs()

    def _obs(self):
        return self.x.reshape(-1, 1).copy()

    def step(self, action, prev_pred):
        """Advance one step using the agent's own previous prediction.

        action    : (B,) the agent's action
        prev_pred : (B,) the agent's prediction made at the previous step
        """
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        prev_pred = np.asarray(prev_pred, dtype=np.float64).reshape(-1)
        if self.exogenous:
            # Exogenous stand-in, matched BY CONSTRUCTION rather than by trying to
            # tune a noise scale: use ANOTHER episode's prediction (a fixed
            # permutation of the batch).  The marginal distribution of
            # `tanh(prev_pred)` is then identical to R1's, while the dependence on
            # *this* agent's own prediction is removed.  Our first attempt used
            # tanh(N(0,1)) and mis-matched by a factor of ~6 (sd 0.0568 vs
            # 0.0091), which would have made R1 vs R2 a comparison of disturbance
            # size rather than of causal position.
            contrib = PRED_GAIN * self.gain * np.tanh(np.roll(prev_pred, 1))
        else:
            contrib = PRED_GAIN * self.gain * np.tanh(prev_pred)
        xi_x = self._rng.normal(0.0, 1.0, size=prev_pred.shape)
        self.x = (RHO * self.x + ACTION_GAIN * action + contrib + NOISE * xi_x)
        if self.loop:
            xi_g = self._rng.normal(0.0, 1.0, size=prev_pred.shape)
            self.gain = (GAIN_BAR * (1 - GAIN_RHO) + GAIN_RHO * self.gain
                         + GAIN_DRIFT * xi_g)
        info = {"gain": self.gain.copy(), "contrib": contrib.copy()}
        return self._obs(), info

    # -- what the world responds to, per world ------------------------------
    @property
    def responds_to_my_prediction(self) -> bool:
        return not self.exogenous

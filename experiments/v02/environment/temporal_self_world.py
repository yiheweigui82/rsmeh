"""v0.2 worlds -- Temporal Self Model Emergence.

The v0.1 finding was negative in an important way: "the latent encodes a hidden
variable" does not mean "the latent encodes the self".  v0.2 attacks the same
question along the TIME axis: is there a representation that stays *continuous*
across time, and does continuity belong to the self or merely to persistence?

Three worlds, identical marginal statistics, one causal difference:

    T1  self_persistent        the hidden variable IS the agent's own state,
                               it has autonomous drift AND responds to the
                               agent's actions                (mine, and moving)
    T2  self_white             the agent's own state, variance-matched, but
                               with NO temporal structure    (nothing to carry)
    T3  external_persistent    a hidden variable with the *same* AR dynamics as
                               T1, but it belongs to the world and does NOT
                               respond to the agent's actions

T1 vs T3 is the crux: from the agent's point of view they are *informationally
identical* (same statistics, same unobservability, same effect on outcomes).
If a "continuous self representation" appears in both, then continuity is a
property of persistence, not of selfhood.  The only thing that distinguishes
them is causal position: whether MY action moves it.  That is what the
counterfactual branch of the task probes.

Task.  At every step the agent must predict the next H observations under a
*stated* held action ("what if I keep doing c"), for every candidate action c.
The counterfactual futures are generated with the SAME exogenous noise as the
factual trajectory, so any divergence between candidates is caused by the
action -- through the world and, in T1, through the agent's own hidden state.
"""

from __future__ import annotations

import numpy as np

WORLDS = ("T1_self_persistent", "T2_self_white", "T3_external_persistent")

DESCRIPTIONS = {
    "T1_self_persistent":
        "the agent's own hidden state: autonomous drift + driven by my actions",
    "T2_self_white":
        "the agent's own hidden state, variance-matched but with no temporal structure",
    "T3_external_persistent":
        "a hidden world drive with the SAME AR dynamics, but my actions do not move it",
}

# hidden-state dynamics -------------------------------------------------------
AR_RHO, AR_INNOV, AR_INIT = 0.92, 0.06, 0.155    # stationary sd = AR_INNOV/sqrt(1-rho^2) = 0.155
SELF_ACTION_GAIN = 0.25                          # how much MY action moves MY state
WHITE_SD = 0.155                                 # variance-matched white control (T2)
# world dynamics --------------------------------------------------------------
POS_RHO, POS_ACTION_GAIN, POS_NOISE = 0.90, 0.10, 0.10
HIDDEN_GAIN = 0.5                                # how strongly the hidden var enters the outcome
# The action's effect on the world is MODULATED by the hidden variable: the same
# action means something different depending on the state I am in.  Without this
# coupling the counterfactual difference (action +1 vs -1) is a constant that any
# memoryless model can guess, and the counterfactual metric has no discriminating
# power -- a metric failure we hit and fixed.
ACTION_MODULATION = 0.6
HORIZONS = (1, 5, 10)
CANDIDATE_ACTIONS = (-1.0, 1.0)


class TemporalSelfWorld:
    def __init__(self, name: str):
        if name not in WORLDS:
            raise ValueError(f"unknown world {name!r}; choose from {WORLDS}")
        self.name = name
        self.kind = "self" if name.startswith(("T1", "T2")) else "external"
        self.structured = name != "T2_self_white"
        self.horizons = HORIZONS

    @property
    def obs_dim(self) -> int:
        return 1                                  # the agent observes the position only

    @property
    def n_candidates(self) -> int:
        return len(CANDIDATE_ACTIONS)

    @property
    def n_horizons(self) -> int:
        return len(HORIZONS)

    def sample_episode(self, batch: int, seq: int, rng):
        H = max(HORIZONS)
        K = self.n_candidates
        # exogenous noise, drawn once and REUSED by the counterfactual branches
        xi_h = rng.normal(size=(batch, seq + H))
        xi_p = rng.normal(size=(batch, seq + H))

        pos = rng.normal(size=batch) * 0.3
        if not self.structured:
            hid = rng.normal(size=batch) * WHITE_SD
        else:
            hid = rng.normal(size=batch) * AR_INIT

        obs = np.zeros((batch, seq, 1), dtype=np.float32)
        actions = np.zeros((batch, seq), dtype=np.float32)
        hidden = np.zeros((batch, seq), dtype=np.float32)
        data = {
            "obs": obs, "actions": actions, "hidden": hidden,
            "targets": np.zeros((batch, seq, K, len(HORIZONS)), dtype=np.float32),
            "hidden_future": np.zeros((batch, seq, K, len(HORIZONS)), dtype=np.float32),
            "pos_t": np.zeros((batch, seq), dtype=np.float32),
        }
        gain = SELF_ACTION_GAIN if self.kind == "self" else 0.0   # T3: my action cannot move it

        for t in range(seq):
            obs[:, t, 0] = pos
            hidden[:, t] = hid
            data["pos_t"][:, t] = pos
            a = np.where(pos > 0, -1.0, 1.0).astype(np.float32)
            a = np.where(rng.random(batch) < 0.15, -a, a)
            actions[:, t] = a

            # counterfactual branch: hold candidate action c from t onward, same noise
            for k, c in enumerate(CANDIDATE_ACTIONS):
                pp, hh = pos.copy(), hid.copy()
                for step in range(1, max(HORIZONS) + 1):
                    pp = (POS_RHO * pp + POS_ACTION_GAIN * c
                          + HIDDEN_GAIN * hh + ACTION_MODULATION * hh * c
                          + POS_NOISE * xi_p[:, t + step - 1])
                    hh = (AR_RHO * hh + gain * c + AR_INNOV * xi_h[:, t + step - 1]) \
                        if self.structured else (WHITE_SD * xi_h[:, t + step - 1])
                    if step in HORIZONS:
                        j = HORIZONS.index(step)
                        data["targets"][:, t, k, j] = pp
                        data["hidden_future"][:, t, k, j] = hh

            # factual transition
            pos = (POS_RHO * pos + POS_ACTION_GAIN * a + HIDDEN_GAIN * hid
                   + ACTION_MODULATION * hid * a + POS_NOISE * xi_p[:, t])
            hid = (AR_RHO * hid + gain * a + AR_INNOV * xi_h[:, t]) if self.structured \
                else (WHITE_SD * xi_h[:, t])

        return data

    # -- ground truth for the post-hoc probes ---------------------------------
    def self_variable(self, ep):
        """The hidden variable of this world (only T1/T2 are *the agent's*)."""
        return ep["hidden"]

    def hidden_future(self, ep):
        """Counterfactual hidden trajectories: (batch, seq, K, n_horizons)."""
        return ep["hidden_future"]

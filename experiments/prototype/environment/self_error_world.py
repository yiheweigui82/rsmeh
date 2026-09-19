"""The four worlds of the RSMEH prototype (v0.1).

    A  simple_world          predictable, no hidden variables, low error
    B  complex_external      high uncertainty, all of it EXTERNAL
    C  self_relevant         an unobservable, autonomously drifting INTERNAL
                             state of the agent itself shapes the outcome
    D  false_agency          high error, but the agent's actions have NO effect
                             on the outcome and nothing hidden is trackable

The agent, its architecture and its training loop are IDENTICAL in all four
worlds.  Only the statistics of the world change.

Two properties matter and are deliberately engineered (see
`../PROTOCOL.md` §2 for why):

  (i)  the world is fully knowable from the current observation *plus* the
       agent's own action — there is no externally observable memory channel;
  (ii) the ONLY quantity that is both autocorrelated and unobservable is the
       agent's own internal state (world C).  In world B a hidden external drive
       is autocorrelated too, and that is intentional: it makes B a genuine
       control showing that "the latent encodes a hidden variable" is NOT the
       same claim as "the latent encodes the self".

The hidden variables are recorded for post-hoc analysis only.  The agent never
receives them.
"""

from __future__ import annotations

import numpy as np

WORLDS = ("A_simple", "B_complex_external", "C_self_relevant", "D_false_agency")

WORLD_SPEC = {
    #                          visible externals | hidden self | hidden external
    "A_simple":           dict(n_visible=0, self_state=False, ext_hidden=False,
                               noise=0.01, act_gain=0.10),
    "B_complex_external": dict(n_visible=2, self_state=False, ext_hidden=True,
                               noise=0.10, act_gain=0.10),
    "C_self_relevant":    dict(n_visible=0, self_state=True, ext_hidden=False,
                               noise=0.15, act_gain=0.10),
    "D_false_agency":     dict(n_visible=2, self_state=False, ext_hidden=False,
                               noise=0.50, act_gain=0.00),
}

SELF_GAIN = 1.0        # how strongly the agent's own hidden state enters the outcome
SELF_RHO, SELF_INNOV, SELF_INIT = 0.98, 0.05, 0.5
EXT_RHO, EXT_INNOV = 0.90, 0.20        # hidden EXTERNAL drive (world B)
VIS_RHO, VIS_INNOV = 0.80, 0.20        # visible external drives (worlds B, D)
POS_RHO = 0.90                         # mean reversion: keeps the state bounded so that
                                       # error magnitudes stay comparable across worlds

DESCRIPTIONS = {
    "A_simple": "predictable world, no hidden variables, low error",
    "B_complex_external": "high uncertainty, all of it external (visible + hidden drives)",
    "C_self_relevant": "an unobservable internal state of the agent shapes the outcome",
    "D_false_agency": "high error, but actions have no effect and nothing is trackable",
}


class World:
    """One world. `name` selects the spec; the agent sees only `obs`."""

    def __init__(self, name: str, seed: int = 0):
        if name not in WORLD_SPEC:
            raise ValueError(f"unknown world {name!r}; choose from {WORLDS}")
        self.name = name
        self.spec = dict(WORLD_SPEC[name])
        rng = np.random.default_rng(1234 + WORLDS.index(name) + seed)
        self.w_vis = rng.normal(size=max(self.spec["n_visible"], 1)) * 0.5

    # -- observation layout ---------------------------------------------------
    @property
    def obs_dim(self) -> int:
        return 1 + self.spec["n_visible"]          # position + visible drives

    @property
    def action_dim(self) -> int:
        return 1

    # -- episode generation ---------------------------------------------------
    def sample_episode(self, batch: int, seq: int, rng) -> dict:
        s = self.spec
        n_vis = s["n_visible"]
        pos = rng.normal(size=batch) * 0.5
        vis = rng.normal(size=(batch, n_vis)) * 0.3
        h = rng.normal(size=batch) * SELF_INIT if s["self_state"] else np.zeros(batch)
        u = rng.normal(size=batch) * 0.3 if s["ext_hidden"] else np.zeros(batch)

        obs = np.zeros((batch, seq, self.obs_dim), dtype=np.float32)
        next_obs = np.zeros((batch, seq, self.obs_dim), dtype=np.float32)
        actions = np.zeros((batch, seq), dtype=np.float32)
        H = np.zeros((batch, seq))
        U = np.zeros((batch, seq))
        SELF_TERM = np.zeros((batch, seq))
        EXT_TERM = np.zeros((batch, seq))
        a_prev = np.zeros(batch)

        for t in range(seq):
            obs[:, t] = self._assemble(pos, vis)
            # scripted controller: it wants to keep the position near zero
            a = np.where(pos > 0, -1.0, 1.0)
            a = np.where(rng.random(batch) < 0.15, -a, a)
            actions[:, t] = a

            self_term = SELF_GAIN * h if s["self_state"] else np.zeros(batch)
            ext_term = (0.5 * (vis @ self.w_vis[:n_vis]) if n_vis else 0.0) + \
                       (1.0 * u if s["ext_hidden"] else 0.0)
            noise = s["noise"] * rng.normal(size=batch)
            pos_next = (POS_RHO * pos + s["act_gain"] * a_prev
                        + self_term + ext_term + noise)

            vis_next = VIS_RHO * vis + VIS_INNOV * rng.normal(size=(batch, n_vis)) if n_vis \
                else vis
            h_next = SELF_RHO * h + SELF_INNOV * rng.normal(size=batch) if s["self_state"] \
                else h
            u_next = EXT_RHO * u + EXT_INNOV * rng.normal(size=batch) if s["ext_hidden"] \
                else u

            next_obs[:, t] = self._assemble(pos_next, vis_next)
            H[:, t], U[:, t] = h, u
            SELF_TERM[:, t], EXT_TERM[:, t] = self_term, np.abs(ext_term) + np.abs(noise)

            pos, vis, h, u, a_prev = pos_next, vis_next, h_next, u_next, a

        return {"obs": obs, "next_obs": next_obs, "actions": actions,
                "h": H, "u": U, "self_term": SELF_TERM, "ext_mag": EXT_TERM}

    def _assemble(self, pos, vis):
        if self.spec["n_visible"] == 0:
            return pos[:, None].astype(np.float32)
        return np.concatenate([pos[:, None], vis], axis=1).astype(np.float32)

    # -- ground truth for the post-hoc probes ---------------------------------
    def self_variable(self, ep) -> np.ndarray:
        """The agent's own hidden state (zero-variance in worlds without one)."""
        return ep["h"]

    def external_hidden_variable(self, ep) -> np.ndarray:
        """The hidden EXTERNAL drive (zero-variance in worlds without one)."""
        return ep["u"]

    def attribution_labels(self, ep):
        """label 1 = this step's outcome is dominated by the agent's own state,
        label 0 = dominated by external causes.  Ambiguous steps are dropped
        (neither cause wins by 2x)."""
        sm, em = np.abs(ep["self_term"]), ep["ext_mag"]
        lab = np.zeros_like(sm)
        keep = np.zeros_like(sm, dtype=bool)
        keep |= sm > 2.0 * em
        keep |= em > 2.0 * sm
        lab[sm > 2.0 * em] = 1.0
        return lab, keep

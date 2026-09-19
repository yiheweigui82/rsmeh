"""v0.5 -- the universe and its sensor interfaces.

The one design decision that makes this experiment meaningful: the universe's
observable quantities are generated **jointly from a small set of shared latent
factors**.  Only then is "three embodiments of one world" a fact about the world
rather than an assumption.  If each modality's variables were independent, the
three agents would simply live in three different universes and there would be
nothing to align -- the degenerate version of the circulated draft, whose
`Universe.step()` returned unrelated quantities (light, sound, infrared,
causal_structure) with no common cause.

Ground rules enforced here:

* **No embodiment may observe a shared factor directly.**  Each receives a
  different fixed projection of them, plus private channels.  (The draft's
  `AbstractSensorWorld` handed the agent `hidden_vector_1/2` and
  `causal_structure` -- i.e. the answer -- which would have made "the abstract
  agent recovers the hidden factors best" true by definition.)
* **Shared vs disjoint is a world-level switch**, not a code fork: same
  dimensionality, same dynamics, same noise.  `shared` = one latent block drives
  all three modalities; `disjoint` = three independent blocks with identical
  statistics drive one modality each (the falsifier: no common cause, so no
  alignable reality should appear).
* Every embodiment also has **private** autoregressive channels, so each really
  does perceive something the others cannot -- "different worlds" content that
  is measurable rather than asserted.
"""

from __future__ import annotations

import numpy as np

SHARED_DIM = 3
RHO = 0.90
INNOV = 0.10
PRIVATE_RHO = 0.90
PRIVATE_INNOV = 0.10

MODES = ("shared", "disjoint")

# each embodiment: (n_shared_channels, n_private_channels)
EMBODIMENTS = {
    "human":    {"shared_channels": 2, "private_channels": 1,
                 "description": "three sensors, coarse embedding of the shared factors"},
    "ai":       {"shared_channels": 2, "private_channels": 2,
                 "description": "four sensors, wider private bandwidth"},
    "abstract": {"shared_channels": 2, "private_channels": 1,
                 "description": "nonlinear (tanh) mixing of two shared directions"},
}


class Universe:
    """One world; `mode` selects whether a common cause exists at all."""

    def __init__(self, mode: str = "shared", seed: int = 0):
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; choose from {MODES}")
        self.mode = mode
        rng = np.random.default_rng(seed)
        # fixed sensor maps: (2 x SHARED_DIM) per embodiment, unit-ish scale
        self.maps = {}
        for name in EMBODIMENTS:
            self.maps[name] = rng.normal(0.0, 1.0, size=(2, SHARED_DIM))

    @property
    def obs_dims(self):
        return {n: v["shared_channels"] + v["private_channels"]
                for n, v in EMBODIMENTS.items()}

    def rollout(self, batch: int, T: int, rng: np.random.Generator):
        """Generate one episode of every embodiment's observation stream.

        Returns dict with, for each embodiment: (T, batch, obs_dim) observations,
        and the ground-truth latent factors (which no agent ever sees):

            factors (shared block)      s : (T, batch, SHARED_DIM)
            private factors             p[name] : (T, batch, n_private)
            private_blocks[name]        only used in `disjoint` mode as the
                                        stand-in for the shared block
        """
        # --- shared block (or per-embodiment blocks in the disjoint control) --
        s = np.zeros((T, batch, SHARED_DIM))
        x = rng.normal(0.0, 0.3, size=(batch, SHARED_DIM))
        for t in range(T):
            s[t] = x
            x = RHO * x + INNOV * rng.normal(size=(batch, SHARED_DIM))

        blocks = {}
        if self.mode == "shared":
            for name in EMBODIMENTS:
                blocks[name] = s
        else:
            # identical statistics, no common cause: each embodiment's "shared"
            # channels are driven by its OWN block
            for name in EMBODIMENTS:
                b = np.zeros((T, batch, SHARED_DIM))
                y = rng.normal(0.0, 0.3, size=(batch, SHARED_DIM))
                for t in range(T):
                    b[t] = y
                    y = RHO * y + INNOV * rng.normal(size=(batch, SHARED_DIM))
                blocks[name] = b

        # --- private factors ---------------------------------------------------
        priv = {}
        for name, cfg in EMBODIMENTS.items():
            np_ = cfg["private_channels"]
            p = np.zeros((T, batch, np_))
            y = rng.normal(0.0, 0.3, size=(batch, np_))
            for t in range(T):
                p[t] = y
                y = PRIVATE_RHO * y + PRIVATE_INNOV * rng.normal(size=(batch, np_))
            priv[name] = p

        # --- observations ------------------------------------------------------
        obs = {}
        for name, cfg in EMBODIMENTS.items():
            b = blocks[name]
            proj = np.einsum("sd,tbd->tbs", self.maps[name], b)   # (T,batch,2)
            if name == "abstract":
                # nonlinear mixing, still not the factors themselves
                proj = np.stack([np.tanh(proj[..., 0] + proj[..., 1]),
                                 proj[..., 1]], axis=-1)
            obs[name] = np.concatenate([proj, priv[name]], axis=-1)

        return {
            "obs": obs,
            "shared_factors": s,          # ground truth: what "reality" is here
            "private_factors": priv,
            "blocks": blocks,
            "mode": self.mode,
        }


def sensor_slice(name: str):
    """Index ranges of an embodiment's observation: shared-driven vs private."""
    cfg = EMBODIMENTS[name]
    return slice(0, cfg["shared_channels"]), slice(cfg["shared_channels"], None)

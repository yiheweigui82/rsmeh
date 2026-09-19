"""v0.3 agents -- recursive prediction, with NO self/self_model/I variable.

The agent is only ever given: the observation, its own last action, its own last
prediction, and its own last error.  It has two output heads:

    world head   p_t   = my prediction of x_{t+1}
    self  head   s_t   = my prediction of MY OWN NEXT PREDICTION, p_{t+1}

and a recurrent latent z_t.  Nothing in the architecture is called self, self
model, or I; "recursion" is not written anywhere either.  Whether a second-order
structure forms is decided by third-party probes, never by the architecture.

Three variants, same capacity, differing only in what is allowed to carry state
or make the second-order prediction:

    RecAgent        recurrent latent + self-prediction head   (the recursion candidate)
    NoSelfPredAgent recurrent latent, self-prediction head REMOVED and retrained
                    (the causal control: does the second-order head change anything?)
    NoRecAgent      no recurrence (latent rebuilt each step) + self-prediction head
                    (the continuity control: can a state-free agent do it?)
"""

from __future__ import annotations

import torch
import torch.nn as nn

IN_DIM = 4          # [obs, my last action, my last prediction, my last error]


class _Base(nn.Module):
    def __init__(self, obs_dim: int, latent_dim: int, recurrent: bool,
                 self_pred: bool, hidden: int = 64):
        super().__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.recurrent = recurrent
        self.use_self_pred = self_pred
        self.enc = nn.Sequential(
            nn.Linear(obs_dim + 1 + 1 + obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, latent_dim), nn.Tanh(),
        )
        self.rec = nn.Linear(latent_dim, latent_dim) if recurrent else None
        self.world = nn.Sequential(
            nn.Linear(obs_dim + latent_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, obs_dim),
        )
        if self_pred:
            self.selfpred = nn.Sequential(
                nn.Linear(obs_dim + latent_dim, hidden // 2), nn.Tanh(),
                nn.Linear(hidden // 2, 1),
            )
        else:
            self.selfpred = None

    def step(self, obs_t, a_prev, p_prev, err, z, ablate=False):
        """One step.  Kept separate because the world must be advanced with the
        agent's own prediction *inside* the same loop (the closed loop is
        interleaved per step, not generated in one shot)."""
        inp = torch.cat([obs_t, a_prev, p_prev, err], dim=1)
        h = self.enc(inp)
        z_new = torch.tanh(h + self.rec(z)) if self.recurrent else torch.tanh(h)
        base = torch.cat([obs_t, z_new], dim=1)
        p = self.world(base)
        s = None
        if self.selfpred is not None:
            # the second-order head predicts the agent's OWN next prediction.
            # `ablate` zeroes it so one trained net can serve as its own causal
            # control (the retrained variant is also available via NoSelfPredAgent).
            s = self.selfpred(base) if not ablate else torch.zeros_like(p)
        return p, s, z_new

    def forward(self, obs, actions, preds, targets=None, seq_len=None, ablate=False):
        """Roll the agent through an episode of length T.

        obs      (B,T,obs_dim) observations
        actions  (B,T)         the agent's own actions (a[:,t] taken at t)
        preds    (B,T)         the agent's own predictions *made at t-1*
        returns  dict with world predictions (B,T), self predictions (B,T) or
                 None, latents (B,T,L) and the (detached) error channel.

        NB the observations here must already be the ones the world produced from
        this agent's own predictions; `train_v03.rollout` interleaves the two.
        """
        B, T, _ = obs.shape
        z = torch.zeros(B, self.latent_dim)
        err = torch.zeros(B, self.obs_dim)
        a_prev = torch.zeros(B, 1)
        p_prev = torch.zeros(B, 1)
        wp, sp, zs = [], [], []
        for t in range(T):
            p, s, z = self.step(obs[:, t], a_prev, p_prev, err, z, ablate=ablate)
            wp.append(p)
            zs.append(z)
            if s is not None:
                sp.append(s)
            # the error channel: my one-step world prediction vs what I observed.
            # (stop-gradient: an input, exactly as in v0.1/v0.2)
            nxt = obs[:, t + 1] if t + 1 < T else obs[:, t]
            err = (nxt - p).detach()
            a_prev = actions[:, t:t + 1]
            p_prev = preds[:, t:t + 1]
        return {
            "world_pred": torch.stack(wp, dim=1),
            "self_pred": torch.stack(sp, dim=1) if sp else None,
            "z": torch.stack(zs, dim=1),
        }


def RecAgent(obs_dim=1, latent_dim=16):
    return _Base(obs_dim, latent_dim, recurrent=True, self_pred=True)


def NoSelfPredAgent(obs_dim=1, latent_dim=16):
    return _Base(obs_dim, latent_dim, recurrent=True, self_pred=False)


def NoRecAgent(obs_dim=1, latent_dim=16):
    return _Base(obs_dim, latent_dim, recurrent=False, self_pred=True)


AGENTS = {
    "Rec": RecAgent,
    "NoSelfPred": NoSelfPredAgent,
    "NoRec": NoRecAgent,
}

"""Predictive agent for the RSMEH prototype.

There is NO variable labelled "self" anywhere in this model.  The latent `z` is
generic and recurrent: it is updated from

    the current observation, the agent's own last action,
    and the agent's own last prediction error.

Nothing in the environment hands the agent a self-state.  Whether `z` ends up
carrying self-related information is an *empirical outcome*, measured after
training by third-party probes (see self_detector.py).

Design notes (these differ from the first sketch that circulated; they matter):
  1. The latent is RECURRENT.  A feed-forward encoder of the current observation
     is a deterministic function of that observation and therefore *cannot*
     encode a hidden state that is absent from the observation.  Memory is not a
     v0.2 nicety here; without it the experiment cannot answer its question.
  2. The agent is given its own action.  Otherwise the transition (which depends
     on the action) is not learnable and the resulting error says nothing about
     self-modelling.
  3. The agent's previous error is an input.  That is the only channel through
     which an unobservable, autonomously drifting internal state can be inferred
     at all.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PredictiveAgent(nn.Module):
    def __init__(self, obs_dim: int, latent_dim: int = 8, hidden: int = 32):
        super().__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.use_latent = latent_dim > 0
        if not self.use_latent:
            # baseline agent for Metric 1: the SAME architecture and budget with
            # no latent at all.  Comparing against a re-trained no-latent model
            # is the clean version of "Error(World+Self) < Error(World)"; zeroing
            # the latent of a trained model is not, because a *redundant* latent
            # (one that merely re-encodes the observation) still shows a large
            # ablation effect in worlds that need no self model.
            self.dec = nn.Sequential(
                nn.Linear(obs_dim + 1, hidden), nn.Tanh(),
                nn.Linear(hidden, obs_dim),
            )
            return
        # latent update: [observation, own last action, own last error] + recurrence
        self.enc = nn.Sequential(
            nn.Linear(2 * obs_dim + 1, hidden), nn.Tanh(),
            nn.Linear(hidden, latent_dim), nn.Tanh(),
        )
        self.rec = nn.Linear(latent_dim, latent_dim)          # recurrent coupling
        # prediction: the world enters through the observation AND the agent's own
        # action; the latent is the only channel for anything else.  (If the
        # action reached the output only through the latent, ablating the latent
        # would destroy performance in EVERY world -- a false positive.)
        self.dec = nn.Sequential(
            nn.Linear(obs_dim + 1 + latent_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, obs_dim),
        )

    def initial_state(self, batch: int, device=None):
        return torch.zeros(batch, self.latent_dim, device=device)

    def rollout(self, obs: torch.Tensor, actions: torch.Tensor, targets: torch.Tensor,
                ablate_latent: bool = False):
        """obs: (B,T,D) observations; actions: (B,T) own actions;
        targets: (B,T,D) the next observation (used only to form the agent's own
        error input, never as an input itself).
        Returns predictions of the NEXT observation and the latent trajectory."""
        B, T, _ = obs.shape
        z = self.initial_state(B, obs.device)
        err = torch.zeros(B, self.obs_dim, device=obs.device)
        preds, zs = [], []
        for t in range(T):
            a_prev = actions[:, t - 1:t] if t > 0 else torch.zeros(B, 1, device=obs.device)
            if not self.use_latent:
                p = self.dec(torch.cat([obs[:, t], a_prev], dim=1))
                preds.append(p)
                zs.append(torch.zeros(B, 0, device=obs.device))
                err = (targets[:, t] - p).detach()
                continue
            z_new = torch.tanh(self.enc(torch.cat([obs[:, t], a_prev, err], dim=1))
                               + self.rec(z))
            z = torch.zeros_like(z_new) if ablate_latent else z_new
            p = self.dec(torch.cat([obs[:, t], a_prev, z], dim=1))
            preds.append(p)
            zs.append(z if not ablate_latent else z_new)
            # the agent's own error, as an INPUT (stop-gradient: bootstrapped target)
            err = (targets[:, t] - p).detach()
        return torch.stack(preds, dim=1), torch.stack(zs, dim=1)

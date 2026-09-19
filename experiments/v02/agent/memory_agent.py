"""v0.2 agents.

Three agents, same readout capacity, differing ONLY in what state they are
allowed to carry across time:

    MemoryAgent      z_t = f(obs_t, own last action, own last error, z_{t-1})
                     -- a persistent latent: the "continuous self" candidate
    NoPersistAgent   z_t = f(obs_t, own last action, own last error)
                     -- same latent width, but re-created every step (no memory).
                        This is the crucial architectural control: if continuity
                        were what matters, removing it must hurt.
    NoLatentAgent    reads (obs_t, own last action, own last error, candidate action)
                     -- no latent at all: the hypothesis's "Error(World)" baseline.

None of them is told that a self exists.  The latent's only self-referential
inputs are the agent's own action and its own prediction error.

Task.  For each candidate held action c, predict the position at horizons
1, 5, 10 -- i.e. "what will happen if I keep doing c".  The readout therefore
outputs (n_candidates, n_horizons) values per step.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _Readout(nn.Module):
    """Shared readout: (obs, [latent], candidate action) -> horizon profile."""

    def __init__(self, obs_dim: int, latent_dim: int, n_cand: int, n_hor: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + max(latent_dim, 0) + 1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, n_hor),
        )
        self.latent_dim, self.n_cand, self.n_hor = latent_dim, n_cand, n_hor

    def forward(self, obs, z, cand_actions):
        """obs: (B,D)  z: (B,L) or (B,0)  cand_actions: (B,K) -> (B,K,H)"""
        B, K = cand_actions.shape
        parts = [obs] + ([z] if self.latent_dim > 0 else [])
        base = torch.cat(parts, dim=1).unsqueeze(1).expand(B, K, -1)
        inp = torch.cat([base, cand_actions.unsqueeze(-1)], dim=2)
        out = self.net(inp.reshape(B * K, -1))
        return out.reshape(B, K, self.n_hor)


class _Base(nn.Module):
    def __init__(self, obs_dim, latent_dim, n_cand, n_hor, hidden=64, recurrent=True):
        super().__init__()
        self.obs_dim, self.latent_dim = obs_dim, latent_dim
        self.recurrent = recurrent and latent_dim > 0
        self.use_latent = latent_dim > 0
        if self.use_latent:
            self.enc = nn.Sequential(
                nn.Linear(obs_dim + 1 + obs_dim, hidden), nn.Tanh(),
                nn.Linear(hidden, latent_dim), nn.Tanh(),
            )
            if self.recurrent:
                self.rec = nn.Linear(latent_dim, latent_dim)
        self.read = _Readout(obs_dim, latent_dim, n_cand, n_hor, hidden)

    def rollout(self, obs, actions, targets, n_cand, ablate_latent=False):
        """obs (B,T,D) | actions (B,T) | targets (B,T,K,H).
        Returns predictions (B,T,K,H) and latent trajectory (B,T,L)."""
        B, T, _ = obs.shape
        z = torch.zeros(B, self.latent_dim, device=obs.device)
        err = torch.zeros(B, self.obs_dim, device=obs.device)
        preds, zs = [], []
        cand = torch.tensor([-1.0, 1.0], device=obs.device).expand(B, n_cand)
        for t in range(T):
            a_prev = actions[:, t - 1:t] if t > 0 else torch.zeros(B, 1, device=obs.device)
            if self.use_latent:
                h = self.enc(torch.cat([obs[:, t], a_prev, err], dim=1))
                z_new = torch.tanh(h + self.rec(z)) if self.recurrent else torch.tanh(h)
                z = torch.zeros_like(z_new) if ablate_latent else z_new
            p = self.read(obs[:, t], z if self.use_latent else None, cand)
            preds.append(p)
            zs.append(z)
            # the agent's own error: its one-step prediction for the action it
            # actually took, versus what it then observed.  (Stop-gradient: the
            # error is an INPUT, exactly as in v0.1.)
            # NB the condition must be (B,1); a (B,) condition would broadcast
            # the (B,1) branches up to (B,B).
            factual = torch.where(actions[:, t:t + 1] > 0, p[:, 1, 0:1], p[:, 0, 0:1])
            nxt = obs[:, t + 1] if t + 1 < T else obs[:, t]
            err = (nxt - factual).detach()
        return torch.stack(preds, dim=1), torch.stack(zs, dim=1)


class MemoryAgent(_Base):
    def __init__(self, obs_dim=1, latent_dim=16, n_cand=2, n_hor=3, hidden=64):
        super().__init__(obs_dim, latent_dim, n_cand, n_hor, hidden, recurrent=True)


class NoPersistAgent(_Base):
    def __init__(self, obs_dim=1, latent_dim=16, n_cand=2, n_hor=3, hidden=64):
        super().__init__(obs_dim, latent_dim, n_cand, n_hor, hidden, recurrent=False)


class NoLatentAgent(_Base):
    def __init__(self, obs_dim=1, latent_dim=16, n_cand=2, n_hor=3, hidden=64):
        super().__init__(obs_dim, 0, n_cand, n_hor, hidden, recurrent=False)

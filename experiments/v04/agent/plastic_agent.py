"""v0.4 agent -- slow weights + FAST weights + a second-order head.

The new ingredient is a **fast-weight associative memory** updated online by the
agent's own error:

    F_t = decay * F_{t-1} + lr * g_t * outer(err_{t-1}, phi(x_{t-1}))
    p_t = slow_out(z_t) + F_t phi(x_t)

where `g_t = 1 + KAPPA * m_t` is the world's hidden plasticity modulator.  `m_t`
enters the update rule and **nothing else**: it is not an input to the encoder, not
part of `z_t`, and never appears in the prediction pathway.  That is the whole
point of the design -- the agent's own plasticity drifts, and it can only see the
*consequences* (its own errors), exactly like a system that cannot introspect its
own neuromodulation.

Consequences that make the second-order task non-redundant (the three conditions
v0.3's design failed to satisfy):

    (a) m_t is involved in my FUTURE prediction (it sets the next update's gain),
    (b) m_t is NOT involved in my CURRENT output  (p_t was computed before it
        applies), and
    (c) m_t is unobservable unless the world mode is `visible`.

So predicting `p_{t+1}` from known quantities needs an *extra* estimate of my own
current plasticity -- which can only come from my own error history.

`NoFastAgent` is the causal control: identical in every respect except that the
fast (plastic) pathway is removed, so the agent's own coupling to m_t is gone.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PlasticAgent(nn.Module):
    def __init__(self, obs_dim: int, latent_dim: int = 32, feat_dim: int = 16,
                 hidden: int = 64, use_fast: bool = True, self_pred: bool = True,
                 fast_decay: float = 0.90, fast_lr: float = 0.60):
        super().__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.feat_dim = feat_dim
        self.use_fast = use_fast
        self.use_self_pred = self_pred
        self.fast_decay = fast_decay
        self.fast_lr = fast_lr

        self.encoder = nn.Sequential(          # features for the fast memory
            nn.Linear(obs_dim, feat_dim), nn.Tanh(),
        )
        self.rnn_in = nn.Sequential(           # recurrent latent
            nn.Linear(obs_dim + 1 + 1 + 1, hidden), nn.Tanh(),
            nn.Linear(hidden, latent_dim), nn.Tanh(),
        )
        self.rec = nn.Linear(latent_dim, latent_dim)
        self.slow_out = nn.Linear(latent_dim, 1)
        if self_pred:
            self.self_head = nn.Sequential(
                nn.Linear(latent_dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        else:
            self.self_head = None

    def rollout(self, obs, actions, mod_gain, rng_actions=None):
        """Interleaved rollout: obs (T,B,d), actions (T,B), mod_gain (T,B).

        `mod_gain[t] = 1 + KAPPA * m_t` comes from the world and is used ONLY in
        the fast-weight update -- never fed to the network.
        """
        T, B, _ = obs.shape
        z = torch.zeros(B, self.latent_dim)
        err = torch.zeros(B, 1)
        a_prev = torch.zeros(B, 1)
        p_prev = torch.zeros(B, 1)
        F = torch.zeros(B, 1, self.feat_dim)          # fast weights, no grad
        phi_prev = torch.zeros(B, self.feat_dim)
        preds, spreds, zs = [], [], []
        for t in range(T):
            inp = torch.cat([obs[t], a_prev, p_prev, err], dim=1)
            z = torch.tanh(self.rnn_in(inp) + self.rec(z))
            phi = self.encoder(obs[t])
            p = self.slow_out(z)
            if self.use_fast:
                # NORMALISED retrieval + bounded storage.  Raw fast weights give a
                # positive feedback path (my error changes my memory, which
                # changes my prediction), and the first version diverged to 1e17
                # within 40 iterations.  Dividing by the key norm turns the memory
                # into a kernel average of past (bounded) errors.
                denom = (phi * phi).sum(-1, keepdim=True) + 1.0
                p = p + torch.bmm(F, phi.unsqueeze(-1)).squeeze(-1) / denom
            preds.append(p)
            zs.append(z)
            if self.self_head is not None:
                spreds.append(self.self_head(z))
            # --- the fast-weight update: only here does the plasticity enter ---
            if self.use_fast:
                g = mod_gain[t].reshape(B, 1, 1)
                F = (self.fast_decay * F
                     + self.fast_lr * g * torch.tanh(err).unsqueeze(-1)
                     * phi_prev.unsqueeze(1))
            phi_prev = phi.detach()
            err = (obs[t + 1 if t + 1 < T else t][:, :1] - p).detach()
            a_prev = actions[t].reshape(B, 1)
            p_prev = p.detach()
        return {
            "pred": torch.stack(preds, 0),
            "self_pred": torch.stack(spreds, 0) if spreds else None,
            "z": torch.stack(zs, 0),
        }

    def loss(self, rec, obs):
        """Next-step prediction + the second-order target (my OWN next prediction)."""
        T = obs.shape[0]
        tgt = torch.cat([obs[1:, :, :1], obs[-1:, :, :1]], dim=0)
        m = torch.ones_like(tgt)
        m[-1] = 0.0
        world = (((rec["pred"] - tgt) ** 2) * m).sum() / m.sum()
        out = {"world": world}
        if rec["self_pred"] is not None:
            nxt = torch.cat([rec["pred"][1:].detach(), rec["pred"][-1:].detach()], dim=0)
            sm = torch.ones_like(nxt)
            sm[-1] = 0.0
            out["self"] = (((rec["self_pred"] - nxt) ** 2) * sm).sum() / sm.sum()
        return out


def NoFastAgent(obs_dim, latent_dim=32, feat_dim=16):
    """Causal control: no fast (plastic) pathway at all."""
    return PlasticAgent(obs_dim, latent_dim, feat_dim, use_fast=False)


def NoSelfPredAgent(obs_dim, latent_dim=32, feat_dim=16):
    """Causal control: plasticity present, second-order head removed."""
    return PlasticAgent(obs_dim, latent_dim, feat_dim, use_fast=True, self_pred=False)


AGENTS = {"Plastic": PlasticAgent, "NoFast": NoFastAgent, "NoSelfPred": NoSelfPredAgent}

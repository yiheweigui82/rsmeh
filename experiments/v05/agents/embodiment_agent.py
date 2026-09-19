"""v0.5 agent -- a predictive agent, not an autoencoder.

The circulated draft used `decoder(encoder(x))`, i.e. reconstruction of the
*current* observation.  Its latent is then just a compression of the input, and a
comparison between two such latents measures how much variance two sensor
projections share -- a property of the *sensors*, not of the agents.

Here the task is the one that makes latent content meaningful: **predict the next
observation**, with a recurrent latent.  Nothing named self / reality / concept
exists in the architecture; every claim about "internal reality" is made by
third-party probes in `compare_reality.py`.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EmbodimentAgent(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 32, hidden: int = 128,
                 recurrent: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.recurrent = recurrent
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, latent_dim), nn.Tanh(),
        )
        self.rec = nn.Linear(latent_dim, latent_dim) if recurrent else None
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, input_dim),
        )

    def forward(self, obs):
        """obs: (T, batch, input_dim) -> predictions (T, batch, input_dim),
        latents (T, batch, latent_dim) and the one-step errors."""
        T, B, _ = obs.shape
        z = torch.zeros(B, self.latent_dim)
        preds, zs = [], []
        for t in range(T):
            h = self.encoder(obs[t])
            z = torch.tanh(h + self.rec(z)) if self.recurrent else torch.tanh(h)
            preds.append(self.decoder(z))
            zs.append(z)
        return torch.stack(preds, 0), torch.stack(zs, 0)

    def loss(self, obs):
        """Next-step prediction error (the only training signal)."""
        pred, _ = self.forward(obs)
        target = torch.cat([obs[1:], obs[-1:]], dim=0)     # x_{t+1} (last repeats)
        m = torch.ones_like(target)
        m[-1] = 0.0                                        # no target at T-1
        return (((pred - target) ** 2) * m).sum() / m.sum()

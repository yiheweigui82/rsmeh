"""Identity test -- is the latent continuous across time, and is "continuous"
worth anything?

    python experiments/v02/analysis/identity_test.py --world T1_self_persistent

This script exists to make a specific trap visible.  The draft's identity loss

    L_identity = || z_t - z_{t-1} ||

is minimised perfectly by z_t = const -- a dead latent that carries nothing.  So
is the draft's identity metric, `mean_t ||z_{t+1} - z_t||`: the constant latent
scores the BEST possible value on it.

We therefore report three numbers, and we line them up against two deliberately
degenerate baselines (a constant latent and a random one) so the reader can see
which metric can be gamed and which cannot:

    drift       mean ||z_{t+1} - z_t|| / mean ||z_t||     LOW = "stable"
                constant latent: 0.00  <-- looks perfect, is worthless
    separable   P(own next step closer than a foreign trajectory's)
                constant latent: 0.00 (own distance is 0, foreign is 0 too,
                so the strict inequality never holds) -- correctly FAILS
                random latent: ~0.50 (chance)
    info        R^2 decoding the world's hidden variable from z
                both degenerate baselines: 0.00

A genuine continuous self representation must score low drift AND high separable
AND high info.  No single number is allowed to carry the claim.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.memory_agent import MemoryAgent                      # noqa: E402
from agent.temporal_metrics import collect, _r2                 # noqa: E402
from environment.temporal_self_world import WORLDS, TemporalSelfWorld  # noqa: E402
from train_v02 import train                                     # noqa: E402


def identity_scores(z, hidden, rng):
    """drift / separable / info for one latent trajectory z:(B,T,L)."""
    B, T, L = z.shape
    zs = z.reshape(-1, L)
    drift = float(np.linalg.norm((z[:, 1:] - z[:, :-1]).reshape(-1, L), axis=-1).mean()
                  / (np.linalg.norm(zs, axis=-1).mean() + 1e-12))
    own = np.linalg.norm(z[:, :-1] - z[:, 1:], axis=-1)
    foreign = np.linalg.norm(z[rng.permutation(B), :-1] - z[:, 1:], axis=-1)
    sep = float((own < foreign).mean())
    hid = hidden.reshape(-1)
    info = _r2(zs, hid) if float(np.var(hid)) > 1e-12 else float("nan")
    return drift, sep, info


def per_dimension(z, hidden):
    hid = hidden.reshape(-1)
    if float(np.var(hid)) < 1e-12:
        return []
    zs = z.reshape(-1, z.shape[-1])
    return [(i, _r2(zs[:, [i]], hid)) for i in range(z.shape[-1])]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--world", default="T1_self_persistent", choices=list(WORLDS))
    p.add_argument("--iters", type=int, default=1200)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--batch", type=int, default=192)
    p.add_argument("--plot", action="store_true")
    args = p.parse_args()

    world = TemporalSelfWorld(args.world)
    model = MemoryAgent(world.obs_dim, 16, world.n_candidates, world.n_horizons)
    print(f"training a persistent-latent agent on {args.world} for {args.iters} iters ...")
    train(world, model, iters=args.iters, seed=args.seed)

    rng = np.random.default_rng(args.seed + 31)
    rec = collect(model, world, rng, batch=args.batch)
    z, hidden = rec["z"], world.self_variable(rec["ep"])
    B, T, L = z.shape

    print(f"\n{'latent':<26}{'drift':>8}{'separable':>11}{'info':>8}   verdict")
    variants = [
        ("trained latent", z),
        ("constant latent (z=0)", np.zeros_like(z)),
        ("random latent", rng.normal(size=z.shape) * float(z.std() or 1.0)),
    ]
    for name, zz in variants:
        d, s, i = identity_scores(zz, hidden, np.random.default_rng(args.seed))
        if name.startswith("trained"):
            verdict = "continuous AND informative" if (s > 0.6 and i > 0.3) else "see numbers"
        elif name.startswith("constant"):
            verdict = "** the naive metric's best score; carries nothing **"
        else:
            verdict = "unstable (high drift), carries nothing"
        print(f"{name:<26}{d:>8.3f}{s:>11.3f}{i:>8.3f}   {verdict}")

    dims = per_dimension(z, hidden)
    if dims:
        dims.sort(key=lambda kv: -kv[1])
        print("\nper-dimension info (R^2 about the hidden variable):")
        for i, r2 in dims[:6]:
            print(f"    z{i:<3} {r2:>6.3f}")
    print("\nreading: only `separable` + `info` together can rule out a dead self.")
    print("`drift` on its own rewards the dead latent -- do not use it alone.")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        best = dims[0][0] if dims else 0
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        ax[0].plot(z[0, :, best], marker="o", ms=3, color="#1f3b73")
        ax[0].set_title(f"z{best} across one episode (continuity?)")
        ax[0].set_xlabel("t"); ax[0].set_ylabel(f"z{best}")
        ax[1].scatter(z[:, :, best].reshape(-1), hidden.reshape(-1), s=3, alpha=0.2,
                      color="#1f3b73")
        ax[1].set_title("z vs hidden variable (informativeness)")
        ax[1].set_xlabel(f"z{best}"); ax[1].set_ylabel("hidden")
        out = os.path.join(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))), "..", "assets",
            f"v02_identity_{args.world}.png")
        out = os.path.abspath(out)
        fig.tight_layout(); fig.savefig(out, dpi=140)
        print(f"\nsaved {out}")


if __name__ == "__main__":
    main()

"""v0.5 -- train one predictive agent per embodiment in one universe.

    python experiments/v05/train_v05.py --iters 700 --seeds 1,2,3
    python experiments/v05/train_v05.py --mode disjoint          # the falsifier
    python experiments/v05/train_v05.py --deterministic

All agents in a run consume streams from the SAME episodes of the SAME universe,
so any cross-embodiment comparison is between observers of one process.
`human_b` is a second agent on the human interface with a different seed: the
same-sensor control (its latent alignment with `human` is the ceiling).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.embodiment_agent import EmbodimentAgent          # noqa: E402
from environments.universe import EMBODIMENTS, Universe       # noqa: E402

AGENT_NAMES = ("human", "ai", "abstract", "human_b")
INTERFACES = {"human": "human", "ai": "ai", "abstract": "abstract", "human_b": "human"}


def norm_pred_err(obs_tb, pred_tb):
    """Next-step error normalised by the sensor variance.

    Takes the UNFLATTENED (T, B, d) arrays: rolling a flattened (T*B, d) array by
    one sample pairs (t, b) with (t, b+1) -- the same instant of a NEIGHBOURING
    trajectory -- which made the first version of this table report errors of
    ~1.8 ("worse than the mean") while the training loss was falling normally.
    """
    obs = np.asarray(obs_tb, float)
    pred = np.asarray(pred_tb, float)
    tgt = np.roll(obs, -1, axis=0)          # (T,B,d), last step wraps harmlessly
    err = ((pred[:-1] - tgt[:-1]) ** 2).mean()
    return float(err / (obs.var() + 1e-12))


def to_tensor(obs, torch_):
    return torch_.as_tensor(obs, dtype=torch_.float32)


def train_agent(universe, interface, latent_dim, iters, batch, seq, lr, seed,
                deterministic=False, verbose=False):
    torch.manual_seed(seed)                       # BEFORE construction (v0.2 bug)
    rng = np.random.default_rng(seed)
    obs_dim = universe.obs_dims[interface]
    model = EmbodimentAgent(obs_dim, latent_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for it in range(iters):
        ep = universe.rollout(batch, seq, rng)
        x = torch.as_tensor(ep["obs"][interface], dtype=torch.float32)
        loss = model.loss(x)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if verbose and it % 200 == 0:
            print(f"    {interface} it {it:5d} predErr {loss.item():.5f}")
    return model


def train_all(mode="shared", iters=700, batch=64, seq=40, lr=3e-3, latent_dim=32,
              seed=1, verbose=False):
    universe = Universe(mode=mode, seed=seed)     # fixed sensor maps per seed
    agents = {}
    for name in AGENT_NAMES:
        s = seed if name != "human_b" else seed + 100
        agents[name] = train_agent(universe, INTERFACES[name], latent_dim, iters,
                                   batch, seq, lr, s, verbose=verbose)
    return universe, agents


def collect(universe, agents, batch=192, seq=40, seed=999):
    """One evaluation episode shared by every agent; returns numpy streams."""
    rng = np.random.default_rng(seed)
    ep = universe.rollout(batch, seq, rng)
    out = {"obs": {}, "z": {}, "factors": ep["shared_factors"],
           "private": ep["private_factors"], "blocks": ep["blocks"],
           "mode": universe.mode}
    with torch.no_grad():
        for name, model in agents.items():
            if model is None:
                continue
            iface = INTERFACES[name]
            x = torch.as_tensor(ep["obs"][iface], dtype=torch.float32)
            pred, z = model(x)
            out["obs"][name] = ep["obs"][iface]
            out["z"][name] = z.numpy()
            out["pred"] = out.get("pred", {})
            out["pred"][name] = pred.numpy()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="shared", choices=["shared", "disjoint"])
    p.add_argument("--iters", type=int, default=700)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--seq", type=int, default=40)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--seeds", type=str, default="1,2,3")
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    if args.deterministic:
        torch.set_num_threads(1)
    seeds = [int(s) for s in args.seeds.split(",")]

    print("=== RSMEH v0.5 -- Multi-Embodiment Reality Model ===")
    print(f"world mode: {args.mode} | latent {args.latent_dim}-dim | iters "
          f"{args.iters} | seeds {seeds}")
    print("\n  one universe, three sensor interfaces; NO embodiment observes the")
    print("  shared factors directly, and every agent's only task is to predict")
    print("  its own next observation.\n")
    for name, cfg in EMBODIMENTS.items():
        print(f"  {name:<9} obs {cfg['shared_channels']}+{cfg['private_channels']}"
              f"  {cfg['description']}")
    print("\ntraining ...")
    rows = {}
    for seed in seeds:
        _, agents = train_all(args.mode, iters=args.iters, batch=args.batch,
                              seq=args.seq, lr=args.lr,
                              latent_dim=args.latent_dim, seed=seed,
                              verbose=args.verbose)
        rec = collect(Universe(mode=args.mode, seed=seed), agents, batch=96,
                      seq=args.seq, seed=seed + 77)
        for name in AGENT_NAMES:
            rows.setdefault(name, []).append(
                norm_pred_err(rec["obs"][name], rec["pred"][name]))
    print(f"\n{'agent':<10}{'normalised predErr (MSE / var of its own sensors)':>52}")
    for name in AGENT_NAMES:
        print(f"{name:<10}{np.mean(rows[name]):>52.4f}")
    print("\n(compare the agents' internal realities with:")
    print("   python experiments/v05/compare_reality.py --mode "
          f"{args.mode})")


if __name__ == "__main__":
    main()

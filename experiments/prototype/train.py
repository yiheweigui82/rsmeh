"""RSMEH prototype v0.1 -- run me.

    python experiments/prototype/train.py --iters 1500 --seeds 1,2,3

Trains the SAME agent (no self variable anywhere) in four worlds and reports,
for each world, whether a latent dimension ends up carrying information about
the agent's own hidden state.

This is the "black box" experiment: the agent is never told that it has an
internal state, and no analysis is used during training.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.predictive_agent import PredictiveAgent          # noqa: E402
from agent.self_detector import (HEADER, metrics, print_reading_notes,  # noqa: E402
                                 print_row, train_baseline)
from environment.self_error_world import DESCRIPTIONS, WORLDS, World  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=1500)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--seq", type=int, default=40)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--latent-dim", type=int, default=8)
    p.add_argument("--seeds", type=str, default="1,2,3")
    p.add_argument("--eval-episodes", type=int, default=256)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    import torch  # noqa: F401  (imported here so --help works without torch)

    from agent.self_detector import train

    seeds = [int(s) for s in args.seeds.split(",")]
    print("=== RSMEH prototype v0.1 -- latent self-information emergence ===")
    print(f"latent {args.latent_dim}-dim | no self variable in the architecture"
          f" | iters {args.iters} | seeds {seeds}\n")
    print("worlds:")
    for w in WORLDS:
        print(f"  {w:<22} {DESCRIPTIONS[w]}")
    print("\n" + HEADER)

    summary = {}
    for name in WORLDS:
        per_seed = []
        for seed in seeds:
            world = World(name, seed=seed)
            model = PredictiveAgent(world.obs_dim, args.latent_dim)
            train(world, model, iters=args.iters, batch=args.batch, seq=args.seq,
                  lr=args.lr, seed=seed, verbose=args.verbose)
            base_err = train_baseline(world, np.random.default_rng(seed + 99),
                                      iters=args.iters, batch=args.batch,
                                      seq=args.seq, lr=args.lr, seed=seed)
            rng = np.random.default_rng(seed + 777)
            per_seed.append(metrics(model, world, rng, batch=args.eval_episodes,
                                    seq=args.seq, base_err=base_err))
        m = {}
        for k in per_seed[0]:
            vals = [d[k] for d in per_seed if d[k] == d[k]]           # drop NaN
            m[k] = float(np.mean(vals)) if vals else float("nan")
        summary[name] = m
        print_row(name, m)

    print_reading_notes()
    print("\nsee ../experiment_results.md for the recorded output and its reading.")

if __name__ == "__main__":
    main()

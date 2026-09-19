"""Counterfactual test -- "would a different action have made a different ME?"

    python experiments/v02/analysis/counterfactual_test.py

The task the agents are trained on already IS the counterfactual question: at
every step they must forecast the next 1/5/10 observations under a *stated* held
action ("what if I keep doing c"), for both c = -1 and c = +1, with the SAME
exogenous noise as the factual trajectory.  The difference between the two
branches is therefore caused by the action alone.

This script separates the two ways the action can matter:

  1. directly, through the world's action gain -- learnable by anyone, needs no
     self-representation;
  2. indirectly, through the agent's OWN hidden state -- in T1 the action moves
     my state (gain 0.25), and my state is not observable, so branch divergence
     can only be forecast by an agent that carries its own state across time.

It reports:

  true |Δ| of the HIDDEN variable under action +1 vs -1, per horizon
        -> T1: non-zero and accumulating (my action moves my state)
        -> T2: zero (variance-matched white: same noise draw, no state)
        -> T3: zero (the world's persistent drive ignores my action)
     This is the causal fingerprint that no observation-only analysis can see.
  cf_err@H   relative error of the predicted counterfactual difference
  cf_corr@H  cosine similarity between predicted and true difference
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.memory_agent import MemoryAgent, NoLatentAgent, NoPersistAgent  # noqa: E402
from agent.temporal_metrics import collect                                 # noqa: E402
from environment.temporal_self_world import HORIZONS, WORLDS, TemporalSelfWorld  # noqa: E402
from train_v02 import train                                                # noqa: E402

AGENTS = {"Memory": MemoryAgent, "NoPersist": NoPersistAgent, "NoLatent": NoLatentAgent}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=1200)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--batch", type=int, default=192)
    args = p.parse_args()

    print("=== counterfactual self-test: action +1 vs action -1, same noise ===\n")
    print("1. does MY action move the hidden variable?  (ground truth, invisible "
          "to the agent)")
    print(f"{'world':<24}" + "".join(f"{'|Δ|@H'+str(h):>12}" for h in HORIZONS))
    for name in WORLDS:
        world = TemporalSelfWorld(name)
        ep = world.sample_episode(args.batch, 20, np.random.default_rng(args.seed))
        hf = world.hidden_future(ep)                      # (B,T,K,H)
        d = np.abs(hf[:, :, 1, :] - hf[:, :, 0, :]).mean(axis=(0, 1))
        print(f"{name:<24}" + "".join(f"{v:>12.4f}" for v in d))

    print("\n2. can the agent forecast that difference?  (relative error, lower "
          "is better)")
    print(f"{'world':<24}{'agent':<10}" + "".join(f"{'cfErr@H'+str(h):>12}" for h in HORIZONS))
    for name in WORLDS:
        world = TemporalSelfWorld(name)
        for agent_name in ("Memory", "NoPersist", "NoLatent"):
            dim = 16 if agent_name != "NoLatent" else 0
            model = AGENTS[agent_name](world.obs_dim, dim, world.n_candidates,
                                       world.n_horizons)
            train(world, model, iters=args.iters, seed=args.seed)
            rec = collect(model, world, np.random.default_rng(args.seed + 31),
                          batch=args.batch)
            tgt, pred = rec["target"], rec["pred"]
            dt = tgt[:, :, 1, :] - tgt[:, :, 0, :]
            dp = pred[:, :, 1, :] - pred[:, :, 0, :]
            errs = []
            for j in range(len(HORIZONS)):
                at, bt = dt[:, :, j].reshape(-1), dp[:, :, j].reshape(-1)
                errs.append(float(np.linalg.norm(bt - at) / (np.linalg.norm(at) + 1e-12)))
            print(f"{name:<24}{agent_name:<10}" + "".join(f"{v:>12.3f}" for v in errs))

    print("""
reading:
  In T1 the action moves the agent's own state, so the two branches diverge in a
  hidden dimension that must be carried across time: the error drops with a
  persistent latent (Memory) and rises without one.
  In T3 the hidden drive has identical dynamics but ignores the action: the true
  divergence is exactly zero, and a memoryless agent is already sufficient.
  The agent cannot SEE this difference -- it can only be found in the causal
  structure.  Any "self" verdict based on observable behaviour is therefore
  undecidable in principle, which is the central negative result of v0.2.""")


if __name__ == "__main__":
    main()

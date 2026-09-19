"""v0.3 -- Recursive Prediction: run me.

    python experiments/v03/train_v03.py --iters 900 --seeds 1,2,3
    python experiments/v03/train_v03.py --iters 900 --seeds 1 --only R1_prediction_loop
    python experiments/v03/train_v03.py --iters 900 --seeds 1 --deterministic
    python experiments/v03/analysis/recursion_test.py

Three worlds x three agents.  The world's transition contains the agent's OWN
previous prediction (R1), a matched exogenous drive instead (R2), or a visible
constant response (R3).  No agent has any variable named self / self model / I,
and "recursion" is never written into the architecture -- it is only ever probed
from the outside.

What counts as evidence is spelled out in ../PROTOCOL_V03.md.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.recursive_agent import AGENTS                          # noqa: E402
from agent.recursive_metrics import HEADER, metrics, print_row    # noqa: E402
from environment.recursive_prediction_world import (              # noqa: E402
    DESCRIPTIONS, WORLDS, RecursivePredictionWorld,
)


def controller(obs_t, rng, flip=0.15):
    """A fixed mean-reverting controller: the agent's action is NOT learned in
    v0.3 (the design question is about prediction, not control)."""
    a = -np.sign(obs_t.reshape(-1))
    a[rng.random(a.shape) < flip] *= -1.0
    return a


def rollout(world: RecursivePredictionWorld, model, rng, batch: int, T: int,
            ablate=False):
    """Interleave world and agent: the world is advanced with the agent's OWN
    prediction, step by step.  That is what makes the loop causal."""
    obs = world.reset(batch, rng)
    z = torch.zeros(batch, model.latent_dim)
    err = torch.zeros(batch, model.obs_dim)
    a_prev = torch.zeros(batch, 1)
    p_prev = torch.zeros(batch, 1)

    O, A, P, S, Z, G = [], [], [], [], [], []
    C = []                                            # the world's response term
    for t in range(T):
        x_t = torch.as_tensor(obs, dtype=torch.float32)
        a_t = controller(obs, rng)
        p, s, z = model.step(x_t, a_prev, p_prev, err, z, ablate=ablate)
        O.append(x_t)
        A.append(torch.as_tensor(a_t, dtype=torch.float32))
        P.append(p)
        if s is not None:
            S.append(s)
        Z.append(z)
        next_obs, info = world.step(a_t, p.detach().numpy().reshape(-1))
        G.append(info["gain"])
        C.append(info["contrib"])
        err = torch.as_tensor(next_obs - p.detach().numpy(), dtype=torch.float32)
        a_prev = torch.as_tensor(a_t, dtype=torch.float32).reshape(-1, 1)
        p_prev = p.detach()
        obs = next_obs

    rec = {
        "obs": torch.stack(O, dim=1),
        "action": torch.stack(A, dim=1),
        "pred": torch.stack(P, dim=1),
        "self_pred": torch.stack(S, dim=1) if S else None,
        "z": torch.stack(Z, dim=1),
        "gain": np.stack(G, axis=1),
        "contrib": np.stack(C, axis=1),
    }
    return rec


def losses(rec):
    """World loss + the honest recursive loss.

    The recursive target is the agent's OWN NEXT prediction p_{t+1} -- computed
    one step later, detached, and unknown when the prediction is made.  The
    draft's version compared two heads of the same network at the SAME step,
    which is a consistency penalty with no external content.
    """
    x = rec["obs"][:, :, 0]                       # (B,T)
    p = rec["pred"][:, :, 0]
    x_next = torch.roll(x, -1, dims=1)
    world = ((p - x_next) ** 2).mean()
    out = {"world": world}
    if rec["self_pred"] is not None:
        s = rec["self_pred"][:, :, 0]
        p_next = torch.roll(p.detach(), -1, dims=1)
        # the last step of each episode has no next prediction -> mask it out
        m = torch.ones_like(s)
        m[:, -1] = 0.0
        out["self"] = (((s - p_next) ** 2) * m).sum() / m.sum()
    return out


def train(world, model, iters=900, batch=64, seq=40, lr=3e-3, seed=1,
          lam_self=1.0, verbose=False):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for it in range(iters):
        rec = rollout(world, model, rng, batch, seq)
        L = losses(rec)
        total = L["world"] + (lam_self * L["self"] if "self" in L else 0.0)
        opt.zero_grad()
        total.backward()
        opt.step()
        if verbose and it % 200 == 0:
            extra = f" self {L['self'].item():.5f}" if "self" in L else ""
            print(f"    it {it:5d} world {L['world'].item():.5f}{extra}")
    return model


def evaluate(model, world, rng, batch=192, seq=40):
    with torch.no_grad():
        rec = rollout(world, model, rng, batch, seq)
        rec_np = {k: (v.numpy() if isinstance(v, torch.Tensor) else v)
                  for k, v in rec.items()}
    m = metrics(rec_np, world)
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=900)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--seq", type=int, default=40)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--latent-dim", type=int, default=16)
    p.add_argument("--seeds", type=str, default="1,2,3")
    p.add_argument("--lam-self", type=float, default=1.0,
                   help="weight of the second-order (own next prediction) loss")
    p.add_argument("--eval-episodes", type=int, default=192)
    p.add_argument("--only", type=str, default="")
    p.add_argument("--deterministic", action="store_true",
                   help="pin a single CPU thread for bit-identical re-runs")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    if args.deterministic:
        torch.set_num_threads(1)
    seeds = [int(s) for s in args.seeds.split(",")]
    worlds = (args.only,) if args.only else WORLDS

    print("=== RSMEH v0.3 -- Recursive Prediction ===")
    print(f"latent {args.latent_dim}-dim | iters {args.iters} | seeds {seeds} | "
          f"lambda_self {args.lam_self}")
    print("\n  no self / self-model / I variable exists anywhere in the agent; "
          "the\n  world's transition contains the agent's OWN previous prediction "
          "(R1).\n")
    for w in worlds:
        print(f"  {w:<22} {DESCRIPTIONS[w]}")
    print("\n" + HEADER)

    keys = None
    for world_name in worlds:
        world = RecursivePredictionWorld(world_name)
        for agent_name in ("Rec", "NoSelfPred", "NoRec"):
            per_seed = []
            for seed in seeds:
                torch.manual_seed(seed)          # BEFORE construction (see v0.2)
                model = AGENTS[agent_name](world.obs_dim, args.latent_dim)
                train(world, model, iters=args.iters, batch=args.batch, seq=args.seq,
                      lr=args.lr, seed=seed, lam_self=args.lam_self,
                      verbose=args.verbose)
                per_seed.append(evaluate(model, world,
                                         np.random.default_rng(seed + 31),
                                         batch=args.eval_episodes, seq=args.seq))
            def _nanmean(vals):
                vals = [v for v in vals if v == v]        # drop NaNs
                return float(np.mean(vals)) if vals else float("nan")

            mean = {k: _nanmean([d[k] for d in per_seed]) for k in per_seed[0]}
            keys = list(mean)
            print_row(world_name, agent_name, mean)
        print()

    print("reading:")
    print("  worldErr       one-step prediction error of the world model")
    print("  vacuousR2      R^2 from the latent to the agent's OWN CURRENT")
    print("                 prediction -- the draft's recursion test.  ~1 for")
    print("                 everyone by construction: it cannot fail, so it")
    print("                 tests nothing.  Kept visible on purpose.")
    print("  selfPred       error of the second-order head on the agent's OWN")
    print("                 NEXT prediction")
    print("  trivial        the bar to beat: predicting 'my next prediction =")
    print("                 my current prediction'")
    print("  selfR2fut      latent -> my next prediction (third-party probe)")
    print("  incrR2         extra R^2 from the latent for my next prediction,")
    print("                 beyond my current prediction + observation + action")
    print("  gainInfo       latent -> the world's HIDDEN response gain, i.e. the")
    print("                 part of the loop unreadable from my own output")
    print("\nsee ../experiment_results.md SS10-12 and ../PROTOCOL_V03.md.")


if __name__ == "__main__":
    main()

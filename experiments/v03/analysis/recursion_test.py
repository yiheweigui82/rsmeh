"""Recursion test (v0.3) -- the honest version of the draft's test.

    python experiments/v03/analysis/recursion_test.py --world R1_prediction_loop

The circulated draft proposed, as the recursion test:

    R^2( z_t , prediction_t )      -- i.e. does the latent explain the agent's
                                      OWN CURRENT prediction?

That quantity is ~1.0 for EVERY agent in EVERY world, because the prediction is
computed from the latent: `prediction = decoder(z)`.  A test that cannot fail
tests nothing.  This script prints it next to the real tests so the difference is
visible in one screen:

    vacuous       R^2(z_t -> p_t)      the prediction the latent already produced
    future        R^2(z_t -> p_{t+1})  my OWN NEXT prediction (unknown at t)
    incremental   extra R^2 for p_{t+1} from z_t, beyond p_t + x_t + a_t
    trivial       R^2(p_t -> p_{t+1})  the bar to beat, using no latent at all
    hidden gain   R^2(z_t -> g_t)      the world's hidden response gain, i.e. the
                                       part of the loop unreadable from my output
    loop check    R^2(contrib_t | my previous prediction) vs (| exogenous noise)
                                       -- is the world's response to me learnable?

It also runs the causal controls: the Retrained no-second-order-head agent and the
no-recurrence agent, plus a retrained baseline whose latent is disconnected from
the world (shuffled latent) to show what a null looks like on each metric.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.recursive_agent import AGENTS                       # noqa: E402
from agent.recursive_metrics import _mse, _r2                  # noqa: E402
from environment.recursive_prediction_world import WORLDS, RecursivePredictionWorld  # noqa: E402
from train_v03 import rollout, train                           # noqa: E402


def probe_table(rec, world, rng):
    """All the probes, computed on one evaluation rollout."""
    z = rec["z"].reshape(-1, rec["z"].shape[-1])
    x = rec["obs"][:, :, 0].reshape(-1)
    p = rec["pred"][:, :, 0].reshape(-1)
    a = rec["action"].reshape(-1)
    g = rec["gain"].reshape(-1)
    T = rec["obs"].shape[1]
    B = rec["obs"].shape[0]
    x_next = np.roll(rec["obs"][:, :, 0], -1, axis=1).reshape(-1)
    p_next = np.roll(rec["pred"][:, :, 0], -1, axis=1).reshape(-1)
    ok = np.tile(np.arange(T) < T - 1, B)

    rows = []
    rows.append(("vacuous      R2(z_t -> p_t)   [the draft's test]",
                 _r2(z[ok], p[ok])))
    rows.append(("future       R2(z_t -> p_t+1) [my next prediction]",
                 _r2(z[ok], p_next[ok])))
    ctrl = np.stack([p[ok], x[ok], a[ok]], axis=1)
    rows.append(("incremental  beyond p_t,x_t,a_t",
                 _r2(np.concatenate([ctrl, z[ok]], axis=1), p_next[ok])
                 - _r2(ctrl, p_next[ok])))
    rows.append(("trivial      R2(p_t -> p_t+1) [no latent]",
                 _r2(p[ok][:, None], p_next[ok])))
    rows.append(("hidden gain  R2(z_t -> g_t)",
                 _r2(z[ok], g[ok])))
    # null control: the same probes with the latent rows shuffled in time, which
    # destroys any genuine structure while keeping the marginal distribution.
    z_shuf = z[rng.permutation(z.shape[0])]
    rows.append(("NULL  (shuffled latent) future",
                 _r2(z_shuf[ok], p_next[ok])))
    rows.append(("NULL  (shuffled latent) vacuous",
                 _r2(z_shuf[ok], p[ok])))
    return rows


def loop_check(rec, world):
    """Is the world's response to MY prediction learnable, or exogenous?"""
    T = rec["obs"].shape[1]
    B = rec["obs"].shape[0]
    contrib = rec["contrib"].reshape(-1)
    # prev_pred at step t is the prediction made at t-1 = rec["pred"][:, t-1]
    p = rec["pred"][:, :, 0]
    p_prev = np.roll(p, 1, axis=1).reshape(-1)
    x = rec["obs"][:, :, 0].reshape(-1)
    ok = np.tile(np.arange(T) > 0, B)
    # CONDITION ON THE STATE: the batch elements share dynamics, so a raw
    # regression of the response on my own prediction picks up common structure
    # and reports a spurious fit even in the exogenous control (we measured 0.638
    # there, which cannot be causal).  The honest question is whether my
    # prediction explains the response BEYOND what the observation already does.
    base = _r2(x[ok][:, None], contrib[ok])
    full = _r2(np.stack([x[ok], p_prev[ok]], axis=1), contrib[ok])
    r2_from_my_pred = full - base
    # variance of the response term (matched across R1/R2 by construction)
    return r2_from_my_pred, float(np.std(contrib[ok])), base, full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="R1_prediction_loop", choices=list(WORLDS))
    ap.add_argument("--iters", type=int, default=900)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--batch", type=int, default=192)
    args = ap.parse_args()

    world = RecursivePredictionWorld(args.world)
    rng = np.random.default_rng(args.seed)
    print(f"training each agent on {args.world} for {args.iters} iters ...\n")
    print(f"{'agent':<12}{'probe':<44}{'value':>9}")
    for name in ("Rec", "NoSelfPred", "NoRec"):
        torch.manual_seed(args.seed)
        model = AGENTS[name](world.obs_dim, 16)
        train(world, model, iters=args.iters, seed=args.seed)
        with torch.no_grad():
            rec = rollout(world, model, np.random.default_rng(args.seed + 31),
                          args.batch, 40)
            rec = {k: (v.numpy() if isinstance(v, torch.Tensor) else v)
                   for k, v in rec.items()}
        for label, val in probe_table(rec, world, np.random.default_rng(7)):
            v = "   n/a" if val != val else f"{val:+.3f}"
            print(f"{name:<12}{label:<44}{v:>9}")
        r2, sd, base, full = loop_check(rec, world)
        print(f"{name:<12}{'loop check  R2(response | my pred, beyond x_t)':<44}"
              f"{r2:+.3f}   (response sd {sd:.4f}; x_t alone {base:+.3f}, "
              f"with my pred {full:+.3f})")
        print()

    print("""reading:
  * `vacuous` is ~1.0 for all three agents: the draft's test cannot fail.
  * `future` / `incremental` are the real tests, and they must clear the NULL
    row (shuffled latent) to mean anything.
  * `loop check` shows whether the world's response to my prediction is
    learnable from something I know (R1) or an unreadable exogenous drive (R2).
  * the Rec vs NoSelfPred comparison is the causal control: it asks whether the
    second-order head changes anything at all, or is a passenger.""")


if __name__ == "__main__":
    main()

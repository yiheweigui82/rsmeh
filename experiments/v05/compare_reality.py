"""v0.5 -- compare the internal realities of different embodiments.

    python experiments/v05/compare_reality.py                     # both modes
    python experiments/v05/compare_reality.py --mode shared

What is measured, and why each metric is the honest one:

  representation difference   mean canonical correlation between two agents'
                              latents.  On its own, "different sensors -> different
                              latents" is a tautology (3-dim vs 4-dim projections
                              cannot be equal); the informative question is the
                              INVERSE -- are the two representations equivalent up
                              to a linear transformation, i.e. do they share a core?
  reality recovery            R^2 from an agent's latent to the universe's HIDDEN
                              shared factors.  Here "reality" is by construction
                              the generative process, so this is the operational
                              meaning of "closer to reality" -- and it is measured
                              against ground truth the agent never observes.
  private content             R^2 from the latent to its OWN private factor (should
                              be high) and to another embodiment's private factor
                              (should be ~0: a sanity control on the probes).
  translation                 fit a linear map A->B on half the trajectories, test
                              on the held-out half ("can A understand B's reality?"),
                              with a time-shuffled NULL for the floor.
  prediction ability          normalised next-step error.  Reported next to reality
                              recovery on purpose: a powerful predictor need not be
                              closer to reality, and this table is where that
                              dissociation would show up.

The `disjoint` mode is the falsifier: same dimensionality, same dynamics, same
noise, but NO common cause.  Any alignment that survives there is an artefact.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_v05 import AGENT_NAMES, collect, norm_pred_err, train_all   # noqa: E402
from environments.universe import Universe              # noqa: E402


def r2_multi(X, Y):
    """Joint R^2 of a linear map X -> Y (all output dims at once)."""
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    if X.ndim == 1:
        X = X[:, None]
    Y = Y.reshape(len(X), -1)
    A = np.concatenate([X, np.ones((len(X), 1))], axis=1)
    beta, *_ = np.linalg.lstsq(A, Y, rcond=None)
    res = float(((Y - A @ beta) ** 2).sum())
    tot = float(((Y - Y.mean(0)) ** 2).sum())
    return 1.0 - res / (tot + 1e-12)


def cca_corrs(X, Y, k=3, ridge=1e-3):
    """Canonical correlations between two latent spaces (whitened SVD)."""
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    n = len(X)

    def whiten(M):
        C = M.T @ M / n + ridge * np.eye(M.shape[1])
        w, V = np.linalg.eigh(C)
        w = np.clip(w, 1e-8, None)
        # unit-variance whitening: the canonical correlations are then just the
        # singular values of the whitened cross-covariance, and they live in
        # [0, 1].  (An earlier version multiplied by sqrt(dim) here, which
        # reported "correlations" of 27.8 -- a scaling bug, caught by the
        # range check.)
        return (M @ V) / np.sqrt(w)

    Xw, Yw = whiten(X), whiten(Y)
    C = Xw.T @ Yw / n
    s = np.linalg.svd(C, compute_uv=False)
    return s[:k]


def translation(X, Y, split=0.5):
    """Linear A->B map fitted on the first `split` trajectories only, then
    evaluated on the held-out ones (no leakage across trajectories)."""
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    n_tr = int(len(X) * split)
    Xtr = np.concatenate([X[:n_tr], np.ones((n_tr, 1))], axis=1)
    beta, *_ = np.linalg.lstsq(Xtr, Y[:n_tr], rcond=None)
    Xte = np.concatenate([X[n_tr:], np.ones((len(X) - n_tr, 1))], axis=1)
    return _r2_pred(Xte @ beta, Y[n_tr:])


def _r2_pred(pred, Y):
    Y = np.asarray(Y, float)
    res = float(((Y - pred) ** 2).sum())
    tot = float(((Y - Y.mean(0)) ** 2).sum())
    return 1.0 - res / (tot + 1e-12)


def flat(a):
    """(T, B, d) -> (T*B, d) with a time index kept alongside."""
    T, B, d = a.shape
    return a.reshape(T * B, d)


def analyse(mode, iters, seed, latent_dim=32):
    universe, agents = train_all(mode=mode, iters=iters, latent_dim=latent_dim,
                                 seed=seed)
    rec = collect(universe, agents, batch=192, seq=40, seed=seed + 77)
    factors = flat(rec["factors"])                     # ground-truth shared factors
    priv = {k: flat(v) for k, v in rec["private"].items()}
    blocks = {k: flat(v) for k, v in rec["blocks"].items()}
    zs = {k: flat(v) for k, v in rec["z"].items()}
    obs = {k: flat(v) for k, v in rec["obs"].items()}
    pred = {k: flat(v) for k, v in rec["pred"].items()}

    # --- per-agent: prediction ability and reality recovery -----------------
    rows = {}
    for name in AGENT_NAMES:
        iface = "human" if name == "human_b" else name
        rows[name] = {
            # NB: computed from the (T,B,d) arrays -- see norm_pred_err's docstring
            "predErr": norm_pred_err(rec["obs"][name], rec["pred"][name]),
            "sharedR2": r2_multi(zs[name], factors),
            # the block that actually drives THIS embodiment's sensors: the
            # shared block in `shared` mode, its own block in `disjoint` mode.
            "ownBlockR2": r2_multi(zs[name], blocks[iface]),
            "ownPrivateR2": r2_multi(zs[name], priv[iface]),
            "otherPrivateR2": r2_multi(zs[name], priv["ai" if iface != "ai" else "human"]),
        }

    # --- cross-embodiment alignment -----------------------------------------
    pairs = [("human", "ai"), ("human", "abstract"), ("ai", "abstract")]
    align = {}
    for a, b in pairs:
        c = cca_corrs(zs[a], zs[b])
        null = cca_corrs(zs[a], zs[b][np.random.default_rng(0).permutation(len(zs[b]))])
        align[(a, b)] = (float(c.mean()), float(null.mean()), float(c[0]))
    c = cca_corrs(zs["human"], zs["human_b"])
    same = float(c.mean())

    # --- translation ("can A understand B's reality?") ----------------------
    trans = {}
    for a, b in pairs:
        t = translation(zs[a], zs[b])
        t_null = translation(zs[a], zs[b][np.random.default_rng(1).permutation(len(zs[b]))])
        trans[(a, b)] = (t, t_null)
    return {"mode": mode, "rows": rows, "align": align, "same": same,
            "trans": trans, "agents": list(AGENT_NAMES)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--modes", default="shared,disjoint")
    p.add_argument("--iters", type=int, default=700)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--deterministic", action="store_true")
    args = p.parse_args()
    if args.deterministic:
        import torch
        torch.set_num_threads(1)

    res = {}
    for mode in args.modes.split(","):
        print(f"training three embodiments in a `{mode}` universe ...")
        res[mode] = analyse(mode, args.iters, args.seed, args.latent_dim)

    print("\n=== 1. per-agent: prediction ability vs reality recovery ===")
    print(f"{'agent':<10}" + "".join(f"{m+' predErr':>14}{m+' sharedR2':>14}"
                                     f"{m+' ownBlock':>15}" for m in res))
    for name in AGENT_NAMES:
        line = f"{name:<10}"
        for mode in res:
            r = res[mode]["rows"][name]
            line += f"{r['predErr']:>14.4f}{r['sharedR2']:>14.3f}"
            line += f"{r['ownBlockR2']:>15.3f}"
        print(line)
    print("\n   sanity control (latent -> a PRIVATE factor that is not mine):")
    for name in ("human", "ai", "abstract"):
        line = f"   {name:<7}"
        for mode in res:
            r = res[mode]["rows"][name]
            line += (f"  own {r['ownPrivateR2']:+.2f} / other {r['otherPrivateR2']:+.2f}"
                     f" ({mode})")
        print(line)

    print("\n=== 2. cross-embodiment alignment (mean canonical correlation) ===")
    print(f"{'pair':<22}{'CCA':>8}{'NULL':>8}{'top1':>8}   |   "
          f"{'CCA':>8}{'NULL':>8}{'top1':>8}")
    print(f"{'':<22}{'shared':>24}   |   {'disjoint':>24}")
    for (a, b) in [("human", "ai"), ("human", "abstract"), ("ai", "abstract")]:
        line = f"{a + ' <-> ' + b:<22}"
        for mode in res:
            c, n, t1 = res[mode]["align"][(a, b)]
            line += f"{c:>8.3f}{n:>8.3f}{t1:>8.3f}"
        print(line)
    line = f"{'human <-> human_b (control)':<22}"
    for mode in res:
        line += f"{res[mode]['same']:>8.3f}{'--':>8}{'--':>8}"
    print(line)

    print("\n=== 3. translation: can A understand B's reality? (held-out R^2) ===")
    print(f"{'A -> B':<22}" + "".join(f"{m:>11}{'NULL':>9}" for m in res))
    for (a, b) in [("human", "ai"), ("human", "abstract"), ("ai", "abstract")]:
        line = f"{a + ' -> ' + b:<22}"
        for mode in res:
            t, tn = res[mode]["trans"][(a, b)]
            line += f"{t:>11.3f}{tn:>9.3f}"
        print(line)

    print("""
reading:
  * `predErr` is normalised by each agent's own sensor variance, so the numbers
    are comparable across embodiments with different input dimensions.
  * `sharedR2` (latent -> the universe's hidden factors) is the operational
    meaning of "closer to reality": here reality is the generative process by
    construction -- this experiment cannot say anything about whether a real
    objective world exists, only how embodiment shapes what is recoverable.
  * the `disjoint` column is the falsifier: same statistics, no common cause.  If
    alignment and translation are as high there as under `shared`, the metrics are
    measuring the probes, not a shared reality.
  * `human <-> human_b` is the same-interface ceiling: two observers with
    identical sensors should align almost perfectly.""")


if __name__ == "__main__":
    main()

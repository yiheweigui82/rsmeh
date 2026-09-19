"""v0.2 metrics -- designed so that a DEAD (constant) latent cannot pass.

The v0.1 lesson was "a metric must be shown to distinguish the manipulation".
v0.2 has an extra trap specific to time: any continuity/identity metric invites
the degenerate solution z_t = const, which is maximally "stable" and carries
nothing.  Every continuity number is therefore reported as a PAIR:

    drift       mean_t ||z_{t+1} - z_t|| / mean_t ||z_t||       (low = smooth)
                 -> a constant latent scores 0.00 here, so it is useless alone
    separable   P( ||z_t - z_{t+1}|| < ||z_t - z'_t|| )         (0.5 = chance)
                 -> a constant latent scores ~0.5; a genuinely continuous AND
                    informative latent scores near 1.0
    info        R^2 with which the world's hidden variable is decodable from z
                 -> a constant latent scores 0.0

`separable` is the anti-degenerate test: it asks whether the latent identifies
*this* moment of *this* trajectory (short distance to its own future, long
distance to a foreign trajectory), which continuity alone cannot buy.

The counterfactual metric compares the model's action-conditioned divergence
with the true one:
    cf_corr     mean cosine similarity between predicted and true future
                difference (action +1 vs -1, same exogenous noise)
    cf_amp      mean||predicted difference|| / mean||true difference||
"""

from __future__ import annotations

import numpy as np
import torch


def _r2(X, y):
    A = np.concatenate([X, np.ones((len(X), 1))], axis=1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    v = float(np.var(y))
    return 1.0 - float(np.var(resid)) / v if v > 1e-12 else 0.0


def collect(model, world, rng, batch=192, seq=20):
    ep = world.sample_episode(batch, seq, rng)
    obs = torch.as_tensor(ep["obs"], dtype=torch.float32)
    act = torch.as_tensor(ep["actions"], dtype=torch.float32)
    tgt = torch.as_tensor(ep["targets"], dtype=torch.float32)
    with torch.no_grad():
        pred, z = model.rollout(obs, act, tgt, world.n_candidates)
        pred_abl, _ = model.rollout(obs, act, tgt, world.n_candidates, ablate_latent=True)
    return {"ep": ep, "pred": pred.numpy(), "pred_abl": pred_abl.numpy(),
            "z": z.numpy(), "target": tgt.numpy()}


def metrics(model, world, rng, batch=192, seq=20, base_errors=None):
    rec = collect(model, world, rng, batch, seq)
    tgt, pred, z = rec["target"], rec["pred"], rec["z"]
    out = {}
    # --- prediction quality per horizon (mean over candidate actions) --------
    # pred/tgt are (batch, T, n_candidates, n_horizons)
    err_h = np.mean((pred - tgt) ** 2, axis=(0, 1, 2))       # (n_horizons,)
    for j, h in enumerate(world.horizons):
        out[f"errH{h}"] = float(err_h[j])
    if base_errors is not None:
        for j, h in enumerate(world.horizons):
            out[f"gainH{h}"] = float(base_errors[j] - err_h[j])

    # --- continuity, reported as a triple ------------------------------------
    if z.shape[-1] > 0:
        zs = z.reshape(-1, z.shape[-1])
        step = z[:, 1:] - z[:, :-1]                           # (B,T-1,L)
        drift = float(np.linalg.norm(step, axis=-1).mean()
                      / (np.linalg.norm(zs, axis=-1).mean() + 1e-12))
        # anti-degenerate: is my own next step closer than a foreign trajectory?
        B, T, L = z.shape
        own = np.linalg.norm(z[:, :-1] - z[:, 1:], axis=-1)   # (B,T-1)
        perm = rng.permutation(B)
        foreign = np.linalg.norm(z[perm, :-1] - z[:, 1:], axis=-1)
        out["separable"] = float((own < foreign).mean())
        out["drift"] = drift
        hid = world.self_variable(rec["ep"]).reshape(-1)
        out["info"] = _r2(zs, hid) if float(np.var(hid)) > 1e-12 else float("nan")
    else:
        out["separable"] = float("nan")
        out["drift"] = float("nan")
        out["info"] = float("nan")

    # --- counterfactual: action +1 vs action -1, same exogenous noise --------
    d_true = tgt[:, :, 1, :] - tgt[:, :, 0, :]                # (B,T,H)
    d_pred = pred[:, :, 1, :] - pred[:, :, 0, :]
    a, b = d_true.reshape(-1), d_pred.reshape(-1)
    if float(np.linalg.norm(a)) > 1e-9:
        out["cf_corr"] = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        out["cf_amp"] = float(np.linalg.norm(b) / (np.linalg.norm(a) + 1e-12))
    else:
        out["cf_corr"] = float("nan")
        out["cf_amp"] = float("nan")
    # how large is the true counterfactual divergence in this world?
    out["cf_true_mag"] = float(np.linalg.norm(a) / len(a))
    # sharper: the counterfactual difference at the LONGEST horizon, where the
    # self-modulated part has had time to accumulate
    j = -1
    at, bt = d_true[:, :, j].reshape(-1), d_pred[:, :, j].reshape(-1)
    if float(np.linalg.norm(at)) > 1e-9:
        out["cf_corr_H10"] = float(np.dot(at, bt)
                                   / (np.linalg.norm(at) * np.linalg.norm(bt) + 1e-12))
        out["cf_rel_err_H10"] = float(np.linalg.norm(bt - at) / np.linalg.norm(at))
    else:
        out["cf_corr_H10"] = float("nan")
        out["cf_rel_err_H10"] = float("nan")
    # ablation form (secondary): error when the latent is switched off
    out["errH10_abl"] = float(np.mean((rec["pred_abl"][:, :, :, -1]
                                       - tgt[:, :, :, -1]) ** 2))
    return out

HEADER = (f"{'world':<24}{'agent':<10}{'errH1':>8}{'errH5':>8}{'errH10':>8}"
          f"{'drift':>8}{'sep':>7}{'info':>7}{'cfCorr':>8}{'cfH10':>8}{'cfErr10':>9}")


def print_row(world_name, agent_name, m):
    def f(x, nd=4):
        return "   n/a" if x != x else f"{x:.{nd}f}"
    print(f"{world_name:<24}{agent_name:<10}{f(m['errH1']):>8}{f(m['errH5']):>8}"
          f"{f(m['errH10']):>8}{f(m['drift'], 3):>8}{f(m['separable'], 3):>7}"
          f"{f(m['info'], 3):>7}{f(m['cf_corr'], 3):>8}{f(m['cf_corr_H10'], 3):>8}"
          f"{f(m['cf_rel_err_H10'], 3):>9}")

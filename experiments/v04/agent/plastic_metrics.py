"""v0.4 metrics -- the ones that v0.3 failed on, plus the new self probe.

The headline numbers:

  incrR2   extra R^2 for predicting MY OWN NEXT PREDICTION from my latent, beyond
           (p_t, x_t, a_t).  v0.3 measured 0.003 -- the whole second-order target
           was redundant.  If v0.4's mechanism works this must clear its NULL.
  mInfo    R^2 from my latent to a hidden property of MYSELF: the plasticity
           modulator in effect (1 + KAPPA*m).  This is the direct test of "the
           system estimates its own learning state from its own error history".
           It is ~0 in the `static` world (no modulator to estimate) and should
           not be needed at all in `visible` (the modulator is an input).
  NULL     every probe repeated with the latent rows shuffled in time: a metric
           that cannot beat its NULL is measuring nothing.
"""

from __future__ import annotations

import numpy as np


def _r2(X, y):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if X.ndim == 1:
        X = X[:, None]
    ss_tot = float(((y - y.mean()) ** 2).sum())
    if ss_tot < 1e-10:                       # a constant target is not "predictable"
        return float("nan")
    A = np.concatenate([X, np.ones((len(X), 1))], axis=1)
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return 1.0 - float(((y - A @ beta) ** 2).sum()) / ss_tot


def _mse(a, b):
    a = np.asarray(a, float).reshape(-1)
    b = np.asarray(b, float).reshape(-1)
    return float(((a - b) ** 2).mean())


def metrics(ep, rec, world):
    """ep: episode dict from the trainer; rec: the agent's rollout output."""
    obs = ep["obs"][:, :, :1]                       # (T,B,1) the state channel
    T, B, _ = obs.shape
    p = rec["pred"][:, :, 0]
    z = rec["z"]
    a = ep["action"]
    g = ep["mod_gain"]                              # 1 + KAPPA*m in effect
    L = z.shape[-1]

    ok = np.tile(np.arange(T) < T - 1, B)            # drop the wrap-around step
    zf = z.reshape(-1, L)
    pf, of, af, gf = (v.reshape(-1) for v in (p, obs[:, :, 0], a, g))
    x_next = np.roll(obs[:, :, 0], -1, axis=0).reshape(-1)
    p_next = np.roll(p, -1, axis=0).reshape(-1)
    rng = np.random.default_rng(0)
    zs = zf[rng.permutation(len(zf))]                # NULL: latent shuffled in time

    out = {}
    out["worldErr"] = _mse(pf[ok], x_next[ok]) / float(obs.var() + 1e-12)
    out["vacuousR2"] = _r2(zf[ok], pf[ok])           # the draft's metric (~1, vacuous)

    out["trivialSelfErr"] = _mse(pf[ok], p_next[ok])
    if rec["self_pred"] is not None:
        sp = rec["self_pred"][:, :, 0].reshape(-1)
        out["selfPredErr"] = _mse(sp[ok], p_next[ok])
    else:
        out["selfPredErr"] = float("nan")
    out["selfR2fut"] = _r2(zf[ok], p_next[ok])
    ctrl = np.stack([pf[ok], of[ok], af[ok]], axis=1)
    out["r2_ctrl"] = _r2(ctrl, p_next[ok])
    out["incrR2"] = _r2(np.concatenate([ctrl, zf[ok]], axis=1), p_next[ok]) - out["r2_ctrl"]
    out["incrR2_null"] = _r2(np.concatenate([ctrl, zs[ok]], axis=1), p_next[ok]) - out["r2_ctrl"]
    # does my latent know my own current plasticity?
    out["mInfo"] = _r2(zf[ok], gf[ok] - 1.0)
    out["mInfo_null"] = _r2(zs[ok], gf[ok] - 1.0)
    return out


HEADER = (f"{'world':<9}{'agent':<11}{'worldErr':>9}{'selfPred':>9}{'trivial':>9}"
          f"{'selfR2fut':>10}{'incrR2':>8}{'incrNull':>9}{'mInfo':>8}{'mNull':>7}")


def fmt(x, nd=3):
    return "   n/a" if x != x else f"{x:.{nd}f}"


def print_row(world_name, agent_name, m):
    print(f"{world_name:<9}{agent_name:<11}{fmt(m['worldErr']):>9}"
          f"{fmt(m['selfPredErr']):>9}{fmt(m['trivialSelfErr']):>9}"
          f"{fmt(m['selfR2fut']):>10}{fmt(m['incrR2']):>8}{fmt(m['incrR2_null']):>9}"
          f"{fmt(m['mInfo']):>8}{fmt(m['mInfo_null']):>7}")

"""v0.3 metrics -- including the degenerate one, kept visible on purpose.

The circulated draft's recursion test was `R^2(z_t, prediction_t)` -- the latent
explaining the agent's *contemporaneous* prediction.  But the prediction is
computed from the latent, so that R^2 is ~1 for every agent by construction: the
test cannot fail and therefore tests nothing.  We keep it in the table as
`vacuousR2` so the difference from a real test is visible.

The real question is second-order and about the FUTURE:

  selfPredGain   does the agent predict its OWN NEXT prediction better than the
                 trivial baseline "my next prediction = my current prediction"?
  incrR2         does the latent carry information about my next prediction
                 BEYOND what is already known (my current prediction, the
                 observation, my action)?
  gainInfo       does the latent track the world's HIDDEN response gain -- i.e.
                 the part of the closed loop that cannot be read off my own
                 output?
"""

from __future__ import annotations

import numpy as np


def _r2(X, y):
    """R^2 of a linear probe from X (n,k) to y (n,)."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if X.ndim == 1:
        X = X[:, None]
    n = X.shape[0]
    ss_tot = float(((y - y.mean()) ** 2).sum())
    if ss_tot < 1e-10:
        # a CONSTANT target is "perfectly predictable" by any probe (it just
        # outputs the constant), so R^2 is meaningless: report it as undefined
        # rather than as a fake 1.000.
        return float("nan")
    X = np.concatenate([X, np.ones((n, 1))], axis=1)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    ss_res = float(((y - pred) ** 2).sum())
    return 1.0 - ss_res / (ss_tot + 1e-12)


def _mse(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    return float(((a - b) ** 2).mean())


def metrics(rec, world):
    """rec: dict from `train_v03.rollout`; world: the world object."""
    z = rec["z"]                       # (B,T,L)
    x = rec["obs"][:, :, 0]            # (B,T)
    p = rec["pred"][:, :, 0]           # my prediction made at t (for x_{t+1})
    a = rec["action"]                  # (B,T)
    gain = rec["gain"]                 # (B,T) hidden response gain
    B, T, L = z.shape

    zf = z.reshape(-1, L)
    xf, pf, af, gf = (v.reshape(-1) for v in (x, p, a, gain))

    # next-step world prediction target and the agent's own NEXT prediction
    x_next = np.roll(x, -1, axis=1).reshape(-1)
    p_next = np.roll(p, -1, axis=1).reshape(-1)
    # drop the last step of each episode: `np.roll` would wrap across rows there.
    # Every quantity below is flattened, so the mask is flat too.
    ok = np.tile(np.arange(T) < T - 1, B)

    out = {}
    # --- world prediction ---------------------------------------------------
    out["worldErr"] = _mse(pf[ok], x_next[ok])
    # a memoryless sanity floor: predicting "no change" (x_{t+1} = x_t)
    out["floorNoChange"] = _mse(xf[ok], x_next[ok])

    # --- first order: does the latent know about my own prediction? ---------
    # (the draft's metric: ~1 by construction, for everyone)
    out["vacuousR2"] = _r2(zf[ok], pf[ok])

    # --- second order: the agent's OWN NEXT prediction ----------------------
    # the bar to beat is the trivial predictor "my next prediction = my current
    # prediction" (p_t): any second-order machinery must do better than that.
    out["trivialSelfErr"] = _mse(pf[ok], p_next[ok])
    sp = rec.get("self_pred")
    if sp is not None:
        out["selfPredErr"] = _mse(sp[:, :, 0].reshape(-1)[ok], p_next[ok])
        out["selfPredGain"] = out["trivialSelfErr"] - out["selfPredErr"]
    else:
        out["selfPredErr"] = float("nan")
        out["selfPredGain"] = float("nan")
    out["selfPredR2_future"] = _r2(zf[ok], p_next[ok])     # latent -> my next prediction
    out["trivialR2_future"] = _r2(pf[ok][:, None], p_next[ok])   # p_t -> p_{t+1}
    # incremental: beyond what is already known (current prediction, obs, action)
    ctrl = np.stack([pf[ok], xf[ok], af[ok]], axis=1)
    r2_ctrl = _r2(ctrl, p_next[ok])
    r2_full = _r2(np.concatenate([ctrl, zf[ok]], axis=1), p_next[ok])
    out["r2_ctrl_only"] = r2_ctrl
    out["incrR2"] = r2_full - r2_ctrl

    # --- third: the hidden loop gain (unreadable from my own output) --------
    out["gainInfo"] = _r2(zf[ok], gf[ok])
    return out


HEADER = (f"{'world':<22}{'agent':<12}{'worldErr':>10}{'vacuousR2':>11}"
          f"{'selfPred':>10}{'trivial':>9}{'selfR2fut':>11}{'incrR2':>9}"
          f"{'gainInfo':>10}")


def fmt(x, nd=3):
    return "   n/a" if x != x else f"{x:.{nd}f}"


def print_row(world_name, agent_name, m):
    print(f"{world_name:<22}{agent_name:<12}{fmt(m['worldErr']):>10}"
          f"{fmt(m['vacuousR2']):>11}{fmt(m['selfPredErr']):>10}"
          f"{fmt(m['trivialSelfErr']):>9}{fmt(m['selfPredR2_future']):>11}"
          f"{fmt(m['incrR2']):>9}{fmt(m['gainInfo']):>10}")

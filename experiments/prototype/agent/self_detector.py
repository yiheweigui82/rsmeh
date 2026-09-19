"""Shared task/agent metrics for the RSMEH prototype.

Nothing in here is given to the agent: the probes below are *third-party*
analyses run after training, on records of the hidden variables the agent never
sees.
"""

from __future__ import annotations

import numpy as np
import torch

from .predictive_agent import PredictiveAgent


# ----------------------------------------------------------------------------
# training
# ----------------------------------------------------------------------------

def train(world, model, iters=1500, batch=64, seq=40, lr=3e-3, seed=1,
          latent_penalty=1e-3, verbose=False):
    """BPTT training. The agent's own previous error r is an INPUT (detached):
    a bootstrapped-target convention, exactly as in ../code/simulation.py.

    `latent_penalty` is an information-bottleneck term: the latent costs
    lambda * mean(z^2).  Without it, a latent that is *not needed* still gets
    used (the network may route the action through it), and ablating it then
    destroys performance in every world -- a false positive in the ablation
    metric.  With it, the latent survives only where it pays for itself.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for it in range(iters):
        ep = world.sample_episode(batch, seq, rng)
        obs = torch.as_tensor(ep["obs"], dtype=torch.float32)
        act = torch.as_tensor(ep["actions"], dtype=torch.float32)
        tgt = torch.as_tensor(ep["next_obs"], dtype=torch.float32)
        pred, z = model.rollout(obs, act, tgt)
        loss = torch.mean((pred - tgt) ** 2) + latent_penalty * torch.mean(z ** 2)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if verbose and (it + 1) % 250 == 0:
            print(f"    iter {it + 1:5d}  loss {loss.item():.5f}")
    return model


# ----------------------------------------------------------------------------
# probes (all on fresh held-out episodes)
# ----------------------------------------------------------------------------

def collect(model, world, rng, batch=256, seq=40):
    """Roll out the trained model and record predictions, latent states, errors
    and the hidden ground-truth variables."""
    ep = world.sample_episode(batch, seq, rng)
    obs = torch.as_tensor(ep["obs"], dtype=torch.float32)
    act = torch.as_tensor(ep["actions"], dtype=torch.float32)
    tgt = torch.as_tensor(ep["next_obs"], dtype=torch.float32)
    with torch.no_grad():
        pred, z = model.rollout(obs, act, tgt)
        pred_abl, _ = model.rollout(obs, act, tgt, ablate_latent=True)
    return {"ep": ep, "pred": pred.numpy(), "pred_abl": pred_abl.numpy(),
            "z": z.numpy(), "target": tgt.numpy()}


def _r2_from(X, y):
    """R^2 of a linear least-squares probe of y from features X."""
    A = np.concatenate([X, np.ones((len(X), 1))], axis=1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    var = float(np.var(y))
    return 1.0 - float(np.var(resid)) / var if var > 1e-12 else 0.0


def train_baseline(world, rng, iters=1500, batch=64, seq=40, lr=3e-3, seed=1):
    """Metric 1 properly: the SAME agent with NO latent, trained on the same
    world with the same budget.  `Error(World)` in the hypothesis is this model's
    error -- not the error of a trained model with its latent switched off."""
    model = PredictiveAgent(world.obs_dim, latent_dim=0)
    train(world, model, iters=iters, batch=batch, seq=seq, lr=lr, seed=seed)
    ep = world.sample_episode(256, seq, rng)
    obs = torch.as_tensor(ep["obs"], dtype=torch.float32)
    act = torch.as_tensor(ep["actions"], dtype=torch.float32)
    tgt = torch.as_tensor(ep["next_obs"], dtype=torch.float32)
    with torch.no_grad():
        pred, _ = model.rollout(obs, act, tgt)
    return float(torch.mean((pred - tgt) ** 2))


def metrics(model, world, rng, batch=256, seq=40, base_err=float("nan")):
    """Returns a dict of the four metrics of the experimental protocol."""
    rec = collect(model, world, rng, batch, seq)
    tgt, z = rec["target"], rec["z"]
    mse_int = float(np.mean((rec["pred"] - tgt) ** 2))
    mse_abl = float(np.mean((rec["pred_abl"] - tgt) ** 2))
    var_y = float(np.var(tgt))
    gain = base_err - mse_int if base_err == base_err else float("nan")
    out = {
        # Metric 1: Self Prediction Gain
        "pred_err": mse_int,
        "base_err": base_err,
        # primary form: a re-trained model WITHOUT a latent (the hypothesis's
        # Error(World)), minus the latent model's error
        "self_gain": gain,
        "self_share": gain / var_y if (var_y > 1e-12 and gain == gain) else float("nan"),
        # secondary form: zeroing the latent of the trained model.  Reported for
        # completeness only: a redundant latent inflates it even where no self
        # model is needed (see train.py docstring).
        "self_gain_abl": mse_abl - mse_int,
    }

    # Metric 2: Self Information Alignment (per-latent and joint linear probe)
    Z = z.reshape(-1, z.shape[-1])
    self_var = world.self_variable(rec["ep"]).reshape(-1)
    ext_var = world.external_hidden_variable(rec["ep"]).reshape(-1)
    out["align_self"] = _r2_from(Z, self_var) if np.var(self_var) > 1e-12 else float("nan")
    out["align_self_best"] = max((_r2_from(Z[:, [i]], self_var)
                                  for i in range(Z.shape[1])), default=float("nan")) \
        if np.var(self_var) > 1e-12 else float("nan")
    out["align_ext_hidden"] = _r2_from(Z, ext_var) if np.var(ext_var) > 1e-12 else float("nan")

    # Metric 3: attribution -- does the agent's own state explain the residual
    # better than the leftover does?  (Residual on the observable position
    # channel, which is where the hidden state acts.)
    resid = (tgt[..., 0] - rec["pred"][..., 0]).reshape(-1)
    coef, *_ = np.linalg.lstsq(np.concatenate([Z, np.ones((len(Z), 1))], axis=1),
                               resid, rcond=None)
    explained = Z @ coef[:-1] + coef[-1]
    leftover = resid - explained
    predicted_self = np.abs(explained) > np.abs(leftover)
    lab, keep = world.attribution_labels(rec["ep"])
    lab, keep = lab.reshape(-1)[keep.reshape(-1)], predicted_self[keep.reshape(-1)]
    if (lab == 1).any():
        out["attr_self"] = float((lab[lab == 1] == 1).mean())
    else:
        out["attr_self"] = float("nan")
    if (lab == 0).any():
        out["attr_ext"] = float((lab[lab == 0] == 0).mean())
    else:
        out["attr_ext"] = float("nan")

    # Metric 4 (partial): latent predictability -- is the latent itself
    # predictable? This is the *precondition* for a model of the model
    # (Stage 5), not a demonstration of recursion.
    zt = z[:, :-1].reshape(-1, z.shape[-1])
    zn = z[:, 1:].reshape(-1, z.shape[-1])
    out["latent_r2"] = float(np.mean([
        _r2_from(zt, zn[:, i]) for i in range(zn.shape[1])
    ]))
    return out


HEADER = (f"{'world':<22}{'predErr':>9}{'baseErr':>9}{'selfGain':>10}{'selfShare':>11}"
          f"{'gainAbl':>9}{'alignSelf':>10}{'alignExtH':>10}{'attrSelf':>10}{'attrExt':>9}")


def print_row(world_name, m):
    def f(x, nd=4):
        return "  n/a" if x != x else f"{x:.{nd}f}"
    print(f"{world_name:<22}{m['pred_err']:>9.4f}{f(m['base_err']):>9}"
          f"{f(m['self_gain']):>10}{f(m['self_share']):>11}{f(m['self_gain_abl']):>9}"
          f"{f(m['align_self'], 3):>10}{f(m['align_ext_hidden'], 3):>10}"
          f"{f(m['attr_self'], 3):>10}{f(m['attr_ext'], 3):>9}")


def print_reading_notes():
    print("""
reading:
  predErr   prediction error of the agent WITH the generic latent
  baseErr   error of the SAME agent re-trained with NO latent  (the hypothesis's
            "Error(World)")
  selfGain  baseErr - predErr      <- Metric 1: Self Prediction Gain
  selfShare selfGain / var(target) (comparable across worlds)
  gainAbl   error increase when the trained latent is switched off (secondary;
            inflated by redundant latents, see train.py)
  alignSelf R^2 with which the agent's OWN hidden state is decodable from the
            latent (Metric 2; third-party probe -- decodability, not use)
  alignExtH same, for the hidden EXTERNAL drive (world B only): if this fires
            while alignSelf cannot exist, the latent is modelling the WORLD
  attrSelf  Metric 3: accuracy of attributing a self-caused residual to the self
  attrExt   accuracy of attributing an externally caused residual to the world
  (Metric 4, the recursion test, is NOT settled by this prototype: see
   ../experiment_results.md and ../PROTOCOL.md section 5.)""")

"""RSMEH toy simulation v0.1 -- does a predictive system grow a *self* model?

Question (RSMEH, Experiment A/B/C):
    When persistent prediction error is caused by the agent's OWN hidden state
    rather than by the external world, does a generic predictive network
    spontaneously route causal work through a self-referential latent slot?
    And does that slot become an object of its own prediction ("recursion")?

Design
------
Environment (all conditions share the same architecture; only the statistics of
the world change):

  A  pure_prediction      low-dim world, no irreducible error.
                            Prediction is fully solvable from external features.
    B  external_complexity  high-dim world (12 latent dims) + small irreducible
                            white external noise.  Error persists but is
                            exogenous: the correct response is a *bigger world
                            model*, not a self model.
    C  self_coupled         world + irreducible white noise + a slowly drifting
                            *hidden internal bias* b_t of the agent itself
                            (AR(1), unobservable).  The outcome contains
                            + kappa * b_t.  No external feature carries any
                            information about b_t: the only signal about it is
                            the agent's own history of prediction errors.
    D  self_coupled_white   variance-matched control for C: the self-caused term
                             is *white* (no temporal structure) with the same
                             variance.  If "large persistent self-caused error"
                             were enough, D should behave like C.  If *structure*
                            is what matters, D must behave like A and B.
    E  false_agency         high error, but the agent's actions have NO causal
                             effect on the outcome at all (ACTION_PUSH = 0, large
                             external noise).  Control for "any error drives a
                             self model".

Agent: one recurrent network, no head is labelled "self".

    i_t = [a_{t-1}, r_{t-1}, s_{t-1}]           (a = own last action,
                                                 r = own last error)
    s_t = tanh(W_s i_t + c_s)                   ("self slot": generic latent)
    y_hat_t = w_o . [x_t, s_t] + b_o

The self slot sees NOTHING about the external world: the only channel into it is
the agent's own action/error history.  `a` and `r` reach the output ONLY through
the slot, so ablating the slot measures exactly how much prediction rides on
self-referential state -- and it cannot smuggle world information.

Metrics (held-out episodes, frozen weights)
    predErr : MSE of outcome prediction
    selfGain: MSE_self-ablated - MSE_intact = variance removed by the self slot
              (the "Self Prediction Gain" of the hypothesis)
    selfShare: selfGain / Var(outcome) -- comparable across conditions
    SPR     : Self-Path Reliance = 1 - MSE_intact / MSE_self-ablated
              (relative form; unstable when the error floor is near zero)
    selfEnc : R^2 with which the agent's OWN hidden bias b_t is decodable from
              its self slot (third-party analysis: decodability, not use)
    selfAttr: accuracy of a linear readout (from [s_t, r_t]) labelling
              self-caused residuals as self-caused
    extAttr : same, for externally caused residuals
    RMG     : Recursive Modelling Gain = 1 - MSE(meta) / MSE(baseline), where the
              meta predictor predicts the NEXT residual r_{t+1} from
              [s_t, r_t, a_{t-1}] and the baseline from [r_t, a_{t-1}] alone.
              So RMG is the incremental value of the self state over the freshest
              self-referential signals available.
              NOTE: in this design RMG does NOT separate C from A/B -- residual
              autocorrelation left by an imperfect world model inflates the
              baseline gap in A, and in C the dominant irreducible noise dilutes
              the gain.  Reported for completeness; P2 is tested by condition D
              (variance-matched white self term) instead.  See PROTOCOL.md §5.

Usage
    python simulation.py                       # main table, 3 seeds
    python simulation.py --seeds 1 --iters 800
    python simulation.py --b-ablation          # extra row: B with features withheld
    python simulation.py --grad-check          # finite-difference check of BPTT
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

# ----------------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------------

D_S = 32                     # width of the self slot
SEQ = 40                     # episode length
CONDITIONS = ("A", "B", "C", "D", "E")
WORLD_DIM = {"A": 4, "B": 12, "C": 6, "D": 6, "E": 8}
LABEL = {
    "A": "A pure_prediction",
    "B": "B external_complexity",
    "C": "C self_coupled",
    "D": "D self_coupled_white",
    "E": "E false_agency",
}
WHITE_SD = {"A": 0.01, "B": 0.10, "C": 0.15, "D": 0.15, "E": 0.50}
KAPPA = 1.0                                    # strength of the self term (C only)
SELF_RHO, SELF_INNOV, SELF_INIT = 0.98, 0.05, 0.5
SELF_WHITE_SD = 0.4                            # condition D: variance-matched, i.i.d. self term
EXT_AR, EXT_INNOV = 0.80, 0.20                 # visible external drive (learnable)
ACTION_PUSH = 0.2                              # how strongly the action perturbs the world
WORLD_SEED = 20260919                          # world structure is FIXED across episodes


_WORLD_CACHE: dict = {}


def world_params(cond):
    """The world's readout weights.  Fixed for the agent's whole life: only the
    initial conditions and the noise are resampled per episode.  (Resampling the
    world's mapping every episode would make x unlearnable and would let the
    self slot degenerate into a tracker of recent outcomes -- see the
    'confound' note in experiments/PROTOCOL.md.)"""
    if cond not in _WORLD_CACHE:
        rng = np.random.default_rng(WORLD_SEED + WORLD_DIM[cond])
        d_z = WORLD_DIM[cond]
        _WORLD_CACHE[cond] = (rng.normal(size=d_z) / np.sqrt(d_z), rng.normal(size=2) * 0.5)
    return _WORLD_CACHE[cond]


# ----------------------------------------------------------------------------
# environment
# ----------------------------------------------------------------------------

def sample_trajectory(cond, batch, seq, rng, drop_features=False):
    """Roll out `batch` episodes. Returns dict of arrays (batch, seq, ...)."""
    d_z = WORLD_DIM[cond]
    d_x = d_z + 2                       # + visible external drive
    use_self = cond in ("C", "D")

    w_z, w_e = world_params(cond)

    z = rng.normal(size=(batch, d_z))
    e = rng.normal(size=(batch, 2)) * 0.3
    b = rng.normal(size=batch) * SELF_INIT if use_self else np.zeros(batch)
    a_prev = np.zeros(batch)

    X = np.zeros((batch, seq, d_x))
    Y = np.zeros((batch, seq))
    A = np.zeros((batch, seq))
    B_ = np.zeros((batch, seq))
    SELF_MAG = np.zeros((batch, seq))
    EXT_MAG = np.zeros((batch, seq))

    for t in range(seq):
        X[:, t] = np.concatenate([z, e[:, :2]], axis=1)
        if drop_features:                        # diagnostic: hide the world
            X[:, t] = 0.0

        visible_ext = 0.5 * (e @ w_e)
        white = WHITE_SD[cond] * rng.normal(size=batch)
        self_term = KAPPA * b if use_self else np.zeros(batch)
        Y[:, t] = z @ w_z + visible_ext + white + self_term
        B_[:, t] = b
        SELF_MAG[:, t] = np.abs(self_term)
        EXT_MAG[:, t] = np.abs(white)

        # scripted controller (fixed policy, not part of the predictor)
        a = np.where(z[:, 0] > 0, -1.0, 1.0)
        a = np.where(rng.random(batch) < 0.15, -a, a)
        A[:, t] = a

        z = 0.9 * z + 0.15 * rng.normal(size=(batch, d_z))
        z[:, 0] += (0.0 if cond == "E" else ACTION_PUSH) * a_prev
        e = EXT_AR * e + EXT_INNOV * rng.normal(size=(batch, 2))
        if use_self:
            if cond == "C":
                b = SELF_RHO * b + SELF_INNOV * rng.normal(size=batch)
            else:                                   # D: white, same variance
                b = SELF_WHITE_SD * rng.normal(size=batch)
        a_prev = a

    return {"X": X, "Y": Y, "A": A, "self_mag": SELF_MAG, "ext_mag": EXT_MAG,
            "b": B_, "white_sd": WHITE_SD[cond]}


# ----------------------------------------------------------------------------
# agent
# ----------------------------------------------------------------------------

class Predictor:
    """Single-layer recurrent predictor with a generic self slot."""

    def __init__(self, d_x, d_s=D_S, rng=None, scale=1.0):
        rng = rng or np.random.default_rng(0)
        self.d_x, self.d_s = d_x, d_s
        self.d_in = 2 + d_s                       # [a, r, s] -- self-referential only
        self.W_s = rng.normal(size=(d_s, self.d_in)) * scale / np.sqrt(self.d_in)
        self.c_s = np.zeros(d_s)
        self.w_o = rng.normal(size=d_x + d_s) * scale / np.sqrt(d_x + d_s)
        self.b_o = 0.0

    def forward(self, tr, ablate_self=False, return_cache=False, r_override=None):
        """r_override: fixed error-input series (used by the gradient check so
        that finite differences see the same 'r is an input' convention as the
        analytic gradient; also the bootstrapped-target convention in training)."""
        X, A = tr["X"], tr["A"]
        batch, seq, _ = X.shape
        s = np.zeros((batch, self.d_s))
        r = np.zeros(batch)
        yhat = np.zeros((batch, seq))
        cache = {"I": np.zeros((batch, seq, self.d_in)),
                 "S": np.zeros((batch, seq, self.d_s))}
        for t in range(seq):
            a_prev = A[:, t - 1] if t > 0 else np.zeros(batch)
            r_in = r_override[:, t] if r_override is not None else r
            i = np.concatenate([a_prev[:, None], r_in[:, None], s], axis=1)
            u = i @ self.W_s.T + self.c_s
            s = np.tanh(u)
            if ablate_self:
                s = np.zeros_like(s)
            if return_cache:
                cache["I"][:, t] = i
                cache["S"][:, t] = s
            yhat[:, t] = X[:, t] @ self.w_o[:self.d_x] + s @ self.w_o[self.d_x:] + self.b_o
            r = tr["Y"][:, t] - yhat[:, t]      # detached input feature
        if return_cache:
            return yhat, cache
        return yhat

    def loss_and_grads(self, tr, r_override=None):
        yhat, cache = self.forward(tr, return_cache=True, r_override=r_override)
        Y = tr["Y"]
        batch, seq = Y.shape
        err = yhat - Y
        loss = float(np.mean(err ** 2))

        dW_s = np.zeros_like(self.W_s)
        dc_s = np.zeros_like(self.c_s)
        dw_o = np.zeros_like(self.w_o)
        db_o = 0.0
        d_s_next = np.zeros((batch, self.d_s))

        for t in range(seq - 1, -1, -1):
            d = (2.0 * err[:, t] / (batch * seq))[:, None]        # dL/dyhat
            s = cache["S"][:, t]
            i = cache["I"][:, t]
            dw_o[:self.d_x] += (d * tr["X"][:, t]).sum(axis=0)
            dw_o[self.d_x:] += (d * s).sum(axis=0)
            db_o += d.sum()

            d_s = d @ self.w_o[self.d_x:][None, :] + d_s_next
            du = d_s * (1.0 - s ** 2)                             # tanh'
            dW_s += du.T @ i
            dc_s += du.sum(axis=0)
            d_s_next = du @ self.W_s[:, -self.d_s:]
        return loss, {"W_s": dW_s, "c_s": dc_s, "w_o": dw_o, "b_o": db_o}

    def params(self):
        return {"W_s": self.W_s, "c_s": self.c_s, "w_o": self.w_o, "b_o": self.b_o}


# ----------------------------------------------------------------------------
# training (hand-rolled Adam, numpy only)
# ----------------------------------------------------------------------------

def train(cond, iters, batch, seq, seed, lr=3e-3, verbose=False):
    rng = np.random.default_rng(seed)
    agent = Predictor(d_x=WORLD_DIM[cond] + 2, rng=rng)
    m = {k: np.zeros_like(v) for k, v in agent.params().items()}
    v = {k: np.zeros_like(v) for k, v in agent.params().items()}
    b1, b2, eps = 0.9, 0.999, 1e-8
    for it in range(iters):
        tr = sample_trajectory(cond, batch, seq, rng)
        loss, g = agent.loss_and_grads(tr)
        for k in m:
            p = agent.params()[k]
            gk = np.clip(g[k], -5.0, 5.0)
            m[k] = b1 * m[k] + (1 - b1) * gk
            v[k] = b2 * v[k] + (1 - b2) * gk ** 2
            mh = m[k] / (1 - b1 ** (it + 1))
            vh = v[k] / (1 - b2 ** (it + 1))
            p -= lr * mh / (np.sqrt(vh) + eps)
        if verbose and (it + 1) % 200 == 0:
            print(f"    iter {it + 1:5d}  loss {loss:.4f}")
    return agent


# ----------------------------------------------------------------------------
# evaluation
# ----------------------------------------------------------------------------

@dataclass
class Metrics:
    pred_err: float
    self_gain: float
    self_share: float
    spr: float
    self_enc: float
    self_attr: float
    ext_attr: float
    rmg: float


def slot_series(agent, tr):
    """Self-slot states s_t and the agent's own error series r_t."""
    batch, seq, _ = tr["X"].shape
    ss = np.zeros((batch, seq, agent.d_s))
    rr = np.zeros((batch, seq))
    st = np.zeros((batch, agent.d_s))
    rt = np.zeros(batch)
    yhat = agent.forward(tr)
    for t in range(seq):
        a_prev = tr["A"][:, t - 1] if t > 0 else np.zeros(batch)
        i = np.concatenate([a_prev[:, None], rt[:, None], st], axis=1)
        st = np.tanh(i @ agent.W_s.T + agent.c_s)
        ss[:, t] = st
        rr[:, t] = tr["Y"][:, t] - yhat[:, t]
        rt = rr[:, t]
    return ss, rr


def _readout(train_tr, agent, test_tr):
    """Attribution test: does the agent's own state explain the residual better
    than the leftover does?

    The readout is not a raw linear probe on [s_t, r_t] (which cannot express
    "this residual is explained by my own state"): the residual r_t is first
    regressed on the slot (alpha fitted on held-out training episodes), and the
    two candidate explanations are compared in magnitude:

        self-explained  component : alpha . s_t
        unexplained     leftover  : r_t - alpha . s_t
        predicted label           : |alpha . s_t| > |r_t - alpha . s_t|

    Ground truth label: the step is *self-caused* if the self term dominates the
    external noise term by a 2x margin, *external* if the external term wins by
    2x; ambiguous steps are dropped.

    NOTE (honesty): the readout is trained with generator labels.  It measures
    whether the information is *decodable* from the agent's internal state, not
    that the agent itself uses it.
    """
    def labels(tr):
        sm, em = tr["self_mag"].reshape(-1), tr["ext_mag"].reshape(-1)
        lab = np.zeros_like(sm)
        keep = np.zeros_like(sm, dtype=bool)
        keep |= sm > 2.0 * em
        keep |= em > 2.0 * sm
        lab[sm > 2.0 * em] = 1.0
        return lab, keep

    # fit the self-explanation coefficient on the training episodes
    s_tr, r_tr = slot_series(agent, train_tr)
    S_tr = s_tr.reshape(-1, agent.d_s)
    r_vec = r_tr.reshape(-1)
    alpha, *_ = np.linalg.lstsq(
        np.concatenate([S_tr, np.ones((len(S_tr), 1))], axis=1), r_vec, rcond=None)

    s_te, r_te = slot_series(agent, test_tr)
    S_te = s_te.reshape(-1, agent.d_s)
    r_te = r_te.reshape(-1)
    self_part = S_te @ alpha[:-1] + alpha[-1]
    leftover = r_te - self_part
    pred = np.abs(self_part) > np.abs(leftover)

    lab, keep = labels(test_tr)
    yte, pred = lab[keep], pred[keep]
    self_acc = float((pred[yte == 1] == 1).mean()) if (yte == 1).any() else float("nan")
    ext_acc = float((pred[yte == 0] == 0).mean()) if (yte == 0).any() else float("nan")
    return self_acc, ext_acc


def evaluate(agent, cond, rng, episodes=200, seq=SEQ, drop_features=False):
    tr = sample_trajectory(cond, episodes, seq, rng, drop_features=drop_features)
    yhat = agent.forward(tr)
    var_y = float(np.var(tr["Y"]))
    mse_int = float(np.mean((yhat - tr["Y"]) ** 2))
    yhat_abl = agent.forward(tr, ablate_self=True)
    mse_abl = float(np.mean((yhat_abl - tr["Y"]) ** 2))
    self_gain = mse_abl - mse_int
    spr = 1.0 - mse_int / mse_abl if mse_abl > 1e-9 else 0.0
    self_share = self_gain / var_y if var_y > 1e-9 else float("nan")

    # meta-predictor of the next residual from [s_t, r_t, a_{t-1}]  (recursion
    # test).  The baseline gets the same inputs MINUS the self slot, so RMG is
    # the incremental value of the self state over the freshest self-referential
    # signals available (own residual, own action).
    tr2 = sample_trajectory(cond, episodes, seq, rng, drop_features=drop_features)
    s2, r2 = slot_series(agent, tr2)
    a_prev2 = np.concatenate([np.zeros((r2.shape[0], 1)), tr2["A"][:, :-1]], axis=1)
    F = np.concatenate([s2[:, :-1], r2[:, :-1, None], a_prev2[:, :-1, None]],
                       axis=2).reshape(-1, agent.d_s + 2)
    tgt = r2[:, 1:].reshape(-1)
    A_ = np.concatenate([F, np.ones((len(F), 1))], axis=1)
    coef, *_ = np.linalg.lstsq(A_, tgt, rcond=None)
    pred_r = A_ @ coef
    B_ = np.concatenate([F[:, agent.d_s:], np.ones((len(F), 1))], axis=1)
    coef_b, *_ = np.linalg.lstsq(B_, tgt, rcond=None)
    pred_b = B_ @ coef_b
    mse_full = float(np.mean((pred_r - tgt) ** 2))
    mse_base = float(np.mean((pred_b - tgt) ** 2))
    rmg = 1.0 - mse_full / mse_base if mse_base > 1e-12 else 0.0
    rmg = float(np.clip(rmg, -1.0, 1.0))

    # how much of the agent's OWN hidden bias b_t is decodable from the slot?
    if float(np.var(tr2["b"])) > 1e-9:
        bb = tr2["b"].reshape(-1)
        S5 = np.concatenate([s2.reshape(-1, D_S), np.ones((s2.size // D_S, 1))], axis=1)
        coef, *_ = np.linalg.lstsq(S5, bb, rcond=None)
        resid = bb - S5 @ coef
        self_enc = 1.0 - float(np.var(resid)) / float(np.var(bb))
    else:
        self_enc = float("nan")

    F = np.concatenate([s2[:, :-1], r2[:, :-1, None]], axis=2).reshape(-1, agent.d_s + 1)
    tgt = r2[:, 1:].reshape(-1)
    A_ = np.concatenate([F, np.ones((len(F), 1))], axis=1)
    coef, *_ = np.linalg.lstsq(A_, tgt, rcond=None)
    pred_r = A_ @ coef
    # baseline: predict the next residual from the current residual ALONE, so
    # that autocorrelation of a mis-fit residual cannot look like recursion.
    B_ = np.concatenate([r2[:, :-1].reshape(-1, 1), np.ones((F.shape[0], 1))], axis=1)
    coef_b, *_ = np.linalg.lstsq(B_, tgt, rcond=None)
    pred_b = B_ @ coef_b
    mse_full = float(np.mean((pred_r - tgt) ** 2))
    mse_base = float(np.mean((pred_b - tgt) ** 2))
    rmg = 1.0 - mse_full / mse_base if mse_base > 1e-12 else 0.0
    rmg = float(np.clip(rmg, -1.0, 1.0))
    # attribution readout, trained on a fresh set of episodes
    self_acc, ext_acc = _readout(
        sample_trajectory(cond, episodes, seq, rng, drop_features=drop_features), agent, tr2)

    return Metrics(mse_int, self_gain, self_share, spr, self_enc, self_acc, ext_acc, rmg)


# ----------------------------------------------------------------------------
# gradient check
# ----------------------------------------------------------------------------

def residual_series(agent, tr):
    """The agent's own error series r_t, produced by its own closed loop."""
    batch, seq, _ = tr["X"].shape
    st = np.zeros((batch, agent.d_s))
    rt = np.zeros(batch)
    rr = np.zeros((batch, seq))
    yhat = agent.forward(tr)
    for t in range(seq):
        a_prev = tr["A"][:, t - 1] if t > 0 else np.zeros(batch)
        i = np.concatenate([a_prev[:, None], rt[:, None], st], axis=1)
        st = np.tanh(i @ agent.W_s.T + agent.c_s)
        rr[:, t] = tr["Y"][:, t] - yhat[:, t]
        rt = rr[:, t]
    return rr


def grad_check(seed=0):
    """Finite differences vs the analytic BPTT gradient, under the same
    convention: r_{t-1} is an observed input (stop-gradient)."""
    rng = np.random.default_rng(seed)
    cond, batch, seq = "C", 2, 4
    agent = Predictor(d_x=WORLD_DIM[cond] + 2, rng=rng, scale=0.7)
    tr = sample_trajectory(cond, batch, seq, rng)
    r_ov = residual_series(agent, tr)                 # frozen error inputs
    _, g = agent.loss_and_grads(tr, r_override=r_ov)
    worst = 0.0
    for name in ("W_s", "c_s", "w_o"):
        flat = agent.params()[name].ravel()
        for idx in rng.choice(flat.size, size=min(6, flat.size), replace=False):
            eps = 1e-6
            old = flat[idx]
            flat[idx] = old + eps
            lp, _ = agent.loss_and_grads(tr, r_override=r_ov)
            flat[idx] = old - eps
            lm, _ = agent.loss_and_grads(tr, r_override=r_ov)
            flat[idx] = old
            num = (lp - lm) / (2 * eps)
            ana = g[name].ravel()[idx]
            rel = abs(num - ana) / max(abs(num) + abs(ana), 1e-12)
            worst = max(worst, rel)
    print(f"grad-check: worst relative error over W_s/c_s/w_o = {worst:.3e}")
    return worst


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def run_one(cond, args, seed, drop_features=False):
    agent = train(cond, args.iters, args.batch, args.seq, seed, lr=args.lr)
    rng = np.random.default_rng(seed + 10_000)
    return evaluate(agent, cond, rng, episodes=args.eval_episodes, seq=args.seq,
                    drop_features=drop_features)


def print_table(title, rows):
    print(f"\n{title}")
    print(f"{'condition':<24}{'predErr':>9}{'selfGain':>10}{'selfShare':>11}"
          f"{'selfEnc':>9}{'selfAttr':>10}{'extAttr':>9}{'RMG':>8}")
    for name, m in rows:
        print(f"{name:<24}{m['pred_err']:>9.4f}{m['self_gain']:>10.4f}"
              f"{m['self_share']:>11}{m['self_enc']:>9}{m['self_attr']:>10}"
              f"{m['ext_attr']:>9}{m['rmg']:>8.3f}")


def _f(x, nd=3):
    return "  n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=1200)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--seq", type=int, default=SEQ)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--eval-episodes", type=int, default=200)
    p.add_argument("--seeds", type=str, default="1,2,3")
    p.add_argument("--b-ablation", action="store_true",
                   help="extra table: B with world features withheld")
    p.add_argument("--grad-check", action="store_true")
    args = p.parse_args()

    if args.grad_check:
        grad_check()
        return

    seeds = [int(s) for s in args.seeds.split(",")]
    print("=== RSMEH toy simulation v0.1 ===")
    print(f"architecture: {D_S}-unit self slot | conditions {'/'.join(CONDITIONS)}"
          f" | seeds {seeds} | iters {args.iters}")

    rows = []
    for cond in CONDITIONS:
        ms = [run_one(cond, args, s) for s in seeds]
        def agg(key):
            vals = [getattr(m, key) for m in ms]
            vals = [v for v in vals if not np.isnan(v)]
            return float(np.mean(vals)) if vals else float("nan")
        rows.append((LABEL[cond], {
            "pred_err": float(np.mean([m.pred_err for m in ms])),
            "self_gain": float(np.mean([m.self_gain for m in ms])),
            "self_share": _f(float(np.mean([m.self_share for m in ms])), 4),
            "self_enc": _f(agg("self_enc")),
            "self_attr": _f(agg("self_attr")),
            "ext_attr": _f(agg("ext_attr")),
            "rmg": float(np.mean([m.rmg for m in ms])),
            "spr": float(np.mean([m.spr for m in ms])),
            "spr_sd": float(np.std([m.spr for m in ms])),
        }))
    print_table("Prediction: the self path is used ONLY in C.  (The RMG column does\n"
                "not discriminate in this design -- see PROTOCOL.md section 5.)", rows)
    floors = ", ".join(f"{c} {WHITE_SD[c] ** 2:.1e}" for c in CONDITIONS)
    print(f"\nirreducible external-noise floors (variance): {floors}")
    print("in C the floor is white + whatever part of the agent's own drift it fails to track.")
    for name, m in rows:
        print(f"    {name:<24} SPR(relative) mean {m['spr']:.3f}  sd {m['spr_sd']:.3f}")

    if args.b_ablation:
        print("\n--- diagnostic: same trained network, external features withheld ---")
        print("    (shows how much of the prediction rides on the world model)")
        for cond in ("B", "C", "E"):
            withheld = np.mean([run_one(cond, args, s, drop_features=True).pred_err
                                for s in seeds])
            intact = dict(rows)[LABEL[cond]]["pred_err"]
            print(f"    {LABEL[cond]:<24} intact {intact:.4f} -> features withheld {withheld:.4f}")


if __name__ == "__main__":
    main()

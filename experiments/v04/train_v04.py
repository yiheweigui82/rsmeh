"""v0.4 -- Plastic Self: run me.

    python experiments/v04/train_v04.py --iters 800 --seeds 1,2,3
    python experiments/v04/train_v04.py --mode static        # the falsifier
    python experiments/v04/train_v04.py --mode visible       # the modulator is readable
    python experiments/v04/analysis/plasticity_test.py

v0.4 repairs v0.3's failure.  v0.3 found that "predict my own next prediction" was
redundant (`incrR2` = 0.003): my next prediction was almost a function of my
current one.  Here the agent's own **plasticity** is modulated by a hidden AR(1)
process that (a) shapes my next update, (b) does not shape my current output, and
(c) cannot be observed -- so predicting my next prediction requires an estimate of
my own current learning state, obtainable only from my own error history.

The world is deliberately easy (a persistent state with a switching action gain);
the difficulty is supposed to live inside the agent.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.plastic_agent import AGENTS                        # noqa: E402
from agent.plastic_metrics import HEADER, metrics, print_row   # noqa: E402
from environment.plastic_world import MODES, PlasticWorld     # noqa: E402

DESCRIPTIONS = {
    "plastic": "the agent's own plasticity drifts (hidden AR(1) modulator) -- "
               "unobservable, only its consequences are visible",
    "static": "m == 0: no hidden plasticity at all  [the falsifier]",
    "visible": "the modulator is appended to the observation  [readable self]",
}


def collect(world: PlasticWorld, model, rng, batch: int, T: int):
    """Interleave world and agent; `mod_gain[t]` is what multiplies the update
    performed at step t (= 1 + KAPPA*m produced at step t-1)."""
    obs0 = world.reset(batch, rng)
    O, A, G = [obs0], [], [world.plasticity_gain()]
    for t in range(T):
        a = -np.sign(world.x)                      # fixed controller, no learning
        A.append(a)
        O.append(world.step(a))
        G.append(world.plasticity_gain())
    obs = torch.as_tensor(np.stack(O[:-1], 0), dtype=torch.float32)   # (T,B,d)
    act = torch.as_tensor(np.stack(A, 0), dtype=torch.float32)
    mod = torch.as_tensor(np.stack(G[:T], 0), dtype=torch.float32)
    rec = model.rollout(obs, act, mod)
    return {"obs": obs, "action": act, "mod_gain": mod}, rec


def train(world, model, iters=800, batch=64, T=40, lr=3e-3, seed=1,
          lam_self=1.0, verbose=False):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for it in range(iters):
        ep, rec = collect(world, model, rng, batch, T)
        L = model.loss(rec, ep["obs"])
        loss = L["world"] + (lam_self * L["self"] if "self" in L else 0.0)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if verbose and it % 200 == 0:
            extra = f" self {L['self'].item():.5f}" if "self" in L else ""
            print(f"    it {it:5d} world {L['world'].item():.5f}{extra}")
    return model


def evaluate(world, model, rng, batch=192, T=40):
    with torch.no_grad():
        ep, rec = collect(world, model, rng, batch, T)
        ep_np = {k: (v.numpy() if isinstance(v, torch.Tensor) else v)
                 for k, v in ep.items()}
        rec_np = {k: (v.numpy() if isinstance(v, torch.Tensor) else v)
                  for k, v in rec.items()}
    return metrics(ep_np, rec_np, world)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="plastic", choices=list(MODES))
    p.add_argument("--modes", default="", help="comma-separated, overrides --mode")
    p.add_argument("--iters", type=int, default=800)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--seq", type=int, default=40)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--seeds", type=str, default="1,2,3")
    p.add_argument("--lam-self", type=float, default=1.0)
    p.add_argument("--only", type=str, default="")
    p.add_argument("--eval-episodes", type=int, default=192)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    if args.deterministic:
        torch.set_num_threads(1)
    seeds = [int(s) for s in args.seeds.split(",")]
    modes = (args.modes.split(",") if args.modes else [args.mode])
    agents = (args.only,) if args.only else ("Plastic", "NoFast", "NoSelfPred")

    print("=== RSMEH v0.4 -- Plastic Self (the v0.3 repair) ===")
    print(f"latent {args.latent_dim}-dim | iters {args.iters} | seeds {seeds} | "
          f"lambda_self {args.lam_self}")
    print("\n  the hidden modulator enters ONLY the fast-weight update rule -- never")
    print("  the encoder, the latent or the prediction path.\n")
    for m in modes:
        print(f"  {m:<8} {DESCRIPTIONS[m]}")
    print("\n" + HEADER)

    for mode in modes:
        for agent_name in agents:
            per_seed = []
            for seed in seeds:
                world = PlasticWorld(mode, seed=seed)
                torch.manual_seed(seed)              # before construction (v0.2)
                model = AGENTS[agent_name](world.obs_dim, args.latent_dim)
                train(world, model, iters=args.iters, batch=args.batch, T=args.seq,
                      lr=args.lr, seed=seed, lam_self=args.lam_self,
                      verbose=args.verbose)
                per_seed.append(evaluate(world, model, np.random.default_rng(seed + 31),
                                         batch=args.eval_episodes, T=args.seq))

            def mean(k):
                v = [d[k] for d in per_seed if d[k] == d[k]]
                return float(np.mean(v)) if v else float("nan")

            print_row(mode, agent_name, {k: mean(k) for k in per_seed[0]})
        print()

    print("reading:")
    print("  worldErr   normalised next-step prediction error (vs a persistence floor)")
    print("  selfPred   error of the second-order head on MY OWN NEXT prediction")
    print("  trivial    the bar to beat: 'my next prediction = my current prediction'")
    print("  selfR2fut  latent -> my next prediction (third-party probe)")
    print("  incrR2     EXTRA R^2 for my next prediction beyond (p_t, x_t, a_t)")
    print("             -- v0.3 measured 0.003 here; this is the number v0.4 must move")
    print("  incrNull   the same probe with the latent shuffled in time (the floor)")
    print("  mInfo      latent -> my own plasticity modulator in effect (a hidden")
    print("             property of MYSELF)   mNull: its shuffled-latent floor")
    print("\nsee ../experiment_results.md SS17-19 and ../PROTOCOL_V04.md.")


if __name__ == "__main__":
    main()

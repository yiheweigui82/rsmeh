"""v0.2 -- Temporal Self Model Emergence: run me.

    python experiments/v02/train_v02.py --iters 1200 --seeds 1,2,3

Trains three agents (persistent latent / no persistence / no latent) in three
worlds (self-persistent / self-white / external-persistent) and reports, per
configuration, the horizon-wise prediction error, the *pair* of continuity
metrics (an anti-degenerate separability test next to the drift), the
informativeness of the latent about the hidden variable, and the counterfactual
accuracy.

The identity-loss ablation (`--identity-loss {none,naive,contrastive}`) is run on
the T1 world, because the naive penalty from the draft
`L_identity = ||z_t - z_{t-1}||` has a degenerate optimum: z_t = const satisfies
it perfectly while carrying no information.  We measure that.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.memory_agent import MemoryAgent, NoLatentAgent, NoPersistAgent  # noqa: E402
from agent.temporal_metrics import HEADER, metrics, print_row              # noqa: E402
from environment.temporal_self_world import DESCRIPTIONS, WORLDS, TemporalSelfWorld  # noqa: E402


def identity_term(identity_loss, z, temperature=0.1):
    """The optional continuity objective on the latent trajectory z:(B,T,L)."""
    if identity_loss == "none" or z.shape[-1] == 0 or z.shape[1] < 2:
        return torch.tensor(0.0, device=z.device)
    B, T, L = z.shape
    zt = z[:, :-1].reshape(-1, L)
    zn = z[:, 1:].reshape(-1, L)
    d_pos = ((zt - zn) ** 2).sum(-1) / L
    if identity_loss == "naive":                       # the draft's loss
        return d_pos.mean()
    # contrastive: my own next step must be closer than a foreign trajectory's
    d_neg = ((zt - torch.roll(zn, shifts=T - 1, dims=0)) ** 2).sum(-1) / L
    p = torch.exp(-d_pos / temperature)
    n = torch.exp(-d_neg / temperature)
    return -torch.log((p + 1e-9) / (p + n + 1e-9)).mean()


def train(world, model, iters=1200, batch=64, seq=20, lr=3e-3, seed=1,
          identity_loss="none", lam=1.0, verbose=False):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for it in range(iters):
        ep = world.sample_episode(batch, seq, rng)
        obs = torch.as_tensor(ep["obs"], dtype=torch.float32)
        act = torch.as_tensor(ep["actions"], dtype=torch.float32)
        tgt = torch.as_tensor(ep["targets"], dtype=torch.float32)
        pred, z = model.rollout(obs, act, tgt, world.n_candidates)
        loss = torch.mean((pred - tgt) ** 2) + lam * identity_term(identity_loss, z)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if verbose and (it + 1) % 200 == 0:
            print(f"    iter {it + 1:5d}  loss {loss.item():.5f}")
    return model


AGENTS = {
    "Memory": MemoryAgent,
    "NoPersist": NoPersistAgent,
    "NoLatent": NoLatentAgent,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=1200)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--seq", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--latent-dim", type=int, default=16)
    p.add_argument("--seeds", type=str, default="1,2,3")
    p.add_argument("--identity-loss", type=str, default="none",
                   choices=["none", "naive", "contrastive"])
    p.add_argument("--lam", type=float, default=1.0,
                   help="weight of the identity term (raise it to expose the "
                        "degenerate optimum of the naive penalty)")
    p.add_argument("--deterministic", action="store_true",
                   help="force a single CPU thread so that two runs with the same "
                        "seeds are bit-identical.  The numbers reported in "
                        "experiment_results.md were produced WITHOUT this flag: "
                        "torch CPU reduction order shifts the last digits between "
                        "runs (see PROTOCOL_V02.md section 8).  Effects reported "
                        "here are 3-4x, not marginal, so the shift does not change "
                        "any conclusion.")
    p.add_argument("--eval-episodes", type=int, default=192)
    p.add_argument("--only", type=str, default="",
                   help="restrict to one world (for the identity-loss ablation)")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    if args.deterministic:
        torch.set_num_threads(1)
    seeds = [int(s) for s in args.seeds.split(",")]
    worlds = (args.only,) if args.only else WORLDS

    print("=== RSMEH v0.2 -- Temporal Self Model Emergence ===")
    print(f"latent {args.latent_dim}-dim | one architecture, three carriers of state"
          f" | iters {args.iters} | seeds {seeds} | identity loss: {args.identity_loss}\n")
    for w in WORLDS:
        print(f"  {w:<24} {DESCRIPTIONS[w]}")
    print("\n" + HEADER)

    rows = {}
    for world_name in worlds:
        world = TemporalSelfWorld(world_name)
        # per-world, per-seed
        for agent_name in ("Memory", "NoPersist", "NoLatent"):
            if agent_name == "NoLatent" and world_name != WORLDS[0]:
                continue                     # the no-latent baseline is world-independent-ish
            per_seed = []
            for seed in seeds:
                dim = args.latent_dim if agent_name != "NoLatent" else 0
                # seed BEFORE construction: otherwise the initial weights come
                # from an unseeded global RNG and two runs with the same seed
                # still differ (a real reproducibility bug, caught by the
                # --deterministic check).
                torch.manual_seed(seed)
                model = AGENTS[agent_name](world.obs_dim, dim, world.n_candidates,
                                           world.n_horizons)
                train(world, model, iters=args.iters, batch=args.batch, seq=args.seq,
                      lr=args.lr, seed=seed, identity_loss=args.identity_loss,
                      lam=args.lam,
                      verbose=args.verbose)
                per_seed.append(metrics(model, world, np.random.default_rng(seed + 31),
                                        batch=args.eval_episodes, seq=args.seq))
            m = {}
            for k in per_seed[0]:
                vals = [d[k] for d in per_seed if d[k] == d[k]]
                m[k] = float(np.mean(vals)) if vals else float("nan")
            rows[(world_name, agent_name)] = m
            print_row(world_name, agent_name, m)
        if world_name == WORLDS[0]:
            # gains relative to the no-latent, memory-free baseline of T1
            base = rows[(world_name, "NoLatent")]
            for h in (1, 5, 10):
                print(f"      gainH{h} vs no-latent baseline:"
                      f" Memory {base[f'errH{h}'] - rows[(world_name, 'Memory')][f'errH{h}']:+.4f}"
                      f" | NoPersist {base[f'errH{h}'] - rows[(world_name, 'NoPersist')][f'errH{h}']:+.4f}")

    print("""
reading:
  errH1/5/10  prediction error at horizons 1, 5, 10 (mean over candidate actions)
  drift       mean_t ||z_{t+1}-z_t|| / mean_t ||z_t||   -- LOW is "smooth", but a
              constant latent scores 0.00 here, so it proves nothing on its own
  sep         P(own next step is closer to me than a foreign trajectory's) --
              the ANTI-DEGENERATE continuity test: a constant latent is at chance
              (0.50); a continuous AND informative latent approaches 1.00
  info        R^2 with which the world's hidden variable is decodable from z
  cf_corr     cosine similarity between the predicted and the true future
              difference for action +1 vs -1 (same exogenous noise)
  cf_amp      predicted / true magnitude of that counterfactual difference""")
    print("\nsee ../PROTOCOL_V02.md and ../experiment_results.md for the reading.")


if __name__ == "__main__":
    main()

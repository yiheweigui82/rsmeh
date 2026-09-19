"""Figures for the RSMEH prototype.

    python experiments/prototype/visualize.py --world C_self_relevant
    python experiments/prototype/visualize.py --world B_complex_external --out assets/latent_alignment_B.png

Produces, for one world:
  * left:  scatter of the best-aligned latent dimension against the hidden
           variable the agent never receives
  * right: per-latent-dimension R^2 (how much of the hidden variable each
           dimension carries)

Matplotlib is the only optional dependency of this repo.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.predictive_agent import PredictiveAgent           # noqa: E402
from agent.self_detector import collect, train, _r2_from     # noqa: E402
from environment.self_error_world import World               # noqa: E402

HIDDEN_LABEL = {
    "A_simple": ("(no hidden variable in this world)", "n/a"),
    "B_complex_external": ("hidden EXTERNAL drive (never observed)", "u_t"),
    "C_self_relevant": ("the agent's OWN hidden state (never observed)", "h_t"),
    "D_false_agency": ("(no hidden variable in this world)", "n/a"),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--world", default="C_self_relevant")
    p.add_argument("--iters", type=int, default=1200)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    world = World(args.world, seed=args.seed)
    model = PredictiveAgent(world.obs_dim, latent_dim=8)
    print(f"training {args.world} for {args.iters} iterations ...")
    train(world, model, iters=args.iters, seed=args.seed)
    rec = collect(model, world, np.random.default_rng(args.seed + 777), batch=256)

    z = rec["z"]
    if args.world == "B_complex_external":
        hidden = world.external_hidden_variable(rec["ep"])
        ylab = HIDDEN_LABEL[args.world][1]
    else:
        hidden = world.self_variable(rec["ep"])
        ylab = HIDDEN_LABEL[args.world][1]

    zf = z.reshape(-1, z.shape[-1])
    hf = hidden.reshape(-1)
    if np.var(hf) < 1e-12:
        print("this world has no hidden variable to analyse "
              "(that is the point of A and D -- nothing is trackable).")
        return

    r2 = np.array([_r2_from(zf[:, [i]], hf) for i in range(zf.shape[1])])
    best = int(np.argmax(r2))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), facecolor="white")
    ax[0].scatter(zf[:, best], hf, s=3, alpha=0.25, color="#1f3b73")
    ax[0].set_xlabel(f"latent dimension z{best}  (agent internal, never told what it means)")
    ax[0].set_ylabel(f"{ylab}  (ground truth, never observed)")
    ax[0].set_title(f"{args.world}\nlatent z{best} vs hidden variable  (R² = {r2[best]:.2f})")

    ax[1].bar(range(len(r2)), r2, color="#F5C518", edgecolor="#0A0A0A", linewidth=0.5)
    ax[1].set_xlabel("latent dimension")
    ax[1].set_ylabel("R² (linear probe)")
    ax[1].set_title("how much of the hidden variable each dimension carries")
    ax[1].set_ylim(0, 1)

    fig.tight_layout()
    out = args.out or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "..", "assets", f"latent_alignment_{args.world}.png")
    out = os.path.abspath(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"saved {out}")


if __name__ == "__main__":
    main()

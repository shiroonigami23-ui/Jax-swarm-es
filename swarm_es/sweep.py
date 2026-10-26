"""Grid search over ES hyperparameters.

    python -m swarm_es.sweep --env pendulum --pop 1024 --gens 300 \
        --lrs 0.005 0.01 0.02 0.05 --sigmas 0.03 0.1 --lr-decays 1.0 0.995

Every combination is trained from the same seed. A config's score is the
mean-policy return averaged over the last `--score-window` generations,
so one lucky generation can't win. Writes sweep.csv (sorted, best first),
best.json, and curves.png to --out.
"""
import argparse
import csv
import itertools
import json
import time
from pathlib import Path

from .train import Config, train


def run_sweep(env, pop, n_eval, gens, lrs, sigmas, lr_decays, seed=0,
              score_window=20, out="runs/sweep", verbose=True):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    grid = list(itertools.product(lrs, sigmas, lr_decays))
    rows, curves = [], {}

    for i, (lr, sigma, lr_decay) in enumerate(grid, 1):
        cfg = Config(env=env, pop=pop, n_eval=n_eval, gens=gens, sigma=sigma,
                     lr=lr, lr_decay=lr_decay, seed=seed, log_every=10)
        t0 = time.perf_counter()
        _, _, history = train(cfg, verbose=False)
        seconds = time.perf_counter() - t0

        curve = [r["mean_policy_return"] for r in history]
        window = curve[-score_window:]
        row = {
            "lr": lr, "sigma": sigma, "lr_decay": lr_decay,
            "score": sum(window) / len(window),
            "best": max(curve), "final": curve[-1], "seconds": seconds,
        }
        if "mean_policy_hits" in history[0]:
            hits = [r["mean_policy_hits"] for r in history][-score_window:]
            row["hits"] = sum(hits) / len(hits)
        rows.append(row)
        curves[(lr, sigma, lr_decay)] = curve
        if verbose:
            print(f"[{i:2d}/{len(grid)}] lr={lr:<6} sigma={sigma:<5} lr_decay={lr_decay:<6} "
                  f"score={row['score']:9.2f}  best={row['best']:9.2f}  ({seconds:5.1f}s)",
                  flush=True)

    rows.sort(key=lambda r: r["score"], reverse=True)
    with open(out / "sweep.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    best = rows[0]
    (out / "best.json").write_text(json.dumps(
        {"env": env, "pop": pop, "n_eval": n_eval, "gens": gens, "seed": seed,
         "score_window": score_window, **best}, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    best_key = (best["lr"], best["sigma"], best["lr_decay"])
    for key, curve in curves.items():
        if key != best_key:
            ax.plot(range(1, len(curve) + 1), curve, lw=0.8, alpha=0.35, color="gray")
    ax.plot(range(1, len(curves[best_key]) + 1), curves[best_key], lw=2, color="tab:red",
            label=f"best: lr={best_key[0]}, sigma={best_key[1]}, lr_decay={best_key[2]}")
    ax.set_xlabel("generation")
    ax.set_ylabel("mean-policy return")
    ax.set_title(f"{env} sweep: {len(grid)} configs (gray) vs best (red)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "curves.png", dpi=130)
    plt.close(fig)
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", default="pendulum", choices=["drone", "pendulum"])
    p.add_argument("--pop", type=int, default=1024)
    p.add_argument("--n-eval", type=int, default=5)
    p.add_argument("--gens", type=int, default=300)
    p.add_argument("--lrs", type=float, nargs="+", default=[0.005, 0.01, 0.02, 0.05])
    p.add_argument("--sigmas", type=float, nargs="+", default=[0.03, 0.1])
    p.add_argument("--lr-decays", type=float, nargs="+", default=[1.0, 0.995])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--score-window", type=int, default=20)
    p.add_argument("--out", default="runs/sweep")
    a = p.parse_args(argv)
    run_sweep(a.env, a.pop, a.n_eval, a.gens, a.lrs, a.sigmas, a.lr_decays,
              a.seed, a.score_window, a.out)


if __name__ == "__main__":
    main()

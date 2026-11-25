"""Train the same config from several seeds and report how often it works.

    python -m swarm_es.seeds --env pendulum --seeds 8 --pop 512 --n-eval 4 --gens 300

ES on the pendulum is bimodal: with identical hyperparameters some seeds swing
up and balance (return around -150), others get stuck spinning near the bottom
(around -760). A single run therefore says little about a config; this script
reports the distribution instead.

For every seed it records the score (mean-policy return averaged over the last
`--score-window` generations) and a final evaluation of the trained mean on
`--final-episodes` fresh full-task episodes. A seed counts as a success if its
final evaluation beats `--success` (default -200 for the pendulum).

Writes seeds.csv, summary.json and curves.png to --out, and the best seed's run
(params, config, history) to --out/best/<env>/ so swarm_es.render can draw it.
"""
import argparse
import csv
import json
import shutil
import time
from dataclasses import replace
from pathlib import Path

import jax
import numpy as np

from .envs import make_env
from .rollout import rollout
from .train import Config, save_run, train

DEFAULT_SUCCESS = {"pendulum": -200.0, "drone": 50.0}


def final_eval(cfg: Config, policy, mean, episodes: int, seed: int = 12345):
    """Mean and std of the return of `mean` on fresh full-task episodes."""
    env = make_env(cfg.env)
    keys = jax.random.split(jax.random.key(seed), episodes)
    rets, _ = jax.jit(jax.vmap(lambda k: rollout(env, policy, mean, k)))(keys)
    rets = np.asarray(rets)
    return float(rets.mean()), float(rets.std())


def run_seeds(base: Config, n_seeds: int, out: str, score_window=20, final_episodes=100,
              success=None, verbose=True):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    success = DEFAULT_SUCCESS.get(base.env, 0.0) if success is None else success
    rows, curves = [], {}

    for seed in range(n_seeds):
        cfg = replace(base, seed=seed, out=str(out / "runs" / f"seed{seed}"))
        t0 = time.perf_counter()
        policy, state, history = train(cfg, verbose=False)
        seconds = time.perf_counter() - t0
        save_run(cfg, policy, state, history)

        curve = [r["mean_policy_return"] for r in history]
        window = curve[-score_window:]
        fe_mean, fe_std = final_eval(cfg, policy, state.mean, final_episodes)
        row = {"seed": seed, "score": float(np.mean(window)), "best": max(curve),
               "final_eval": fe_mean, "final_eval_std": fe_std,
               "success": int(fe_mean >= success), "seconds": seconds}
        rows.append(row)
        curves[seed] = curve
        if verbose:
            print(f"seed {seed:2d}  score {row['score']:9.2f}  final_eval {fe_mean:9.2f} "
                  f"± {fe_std:6.2f}  {'SOLVED' if row['success'] else 'stuck '}  ({seconds:5.1f}s)",
                  flush=True)

    with open(out / "seeds.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    finals = np.array([r["final_eval"] for r in rows])
    best = max(rows, key=lambda r: r["final_eval"])
    summary = {
        "env": base.env, "config": {k: v for k, v in vars(base).items() if k not in ("seed", "out")},
        "n_seeds": n_seeds, "success_threshold": success,
        "success_rate": float(np.mean([r["success"] for r in rows])),
        "final_eval_median": float(np.median(finals)),
        "final_eval_mean": float(finals.mean()),
        "final_eval_best": best["final_eval"], "best_seed": best["seed"],
        "seconds_total": float(sum(r["seconds"] for r in rows)),
        "device": jax.devices()[0].device_kind,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    best_dir = out / "best" / base.env
    if best_dir.exists():
        shutil.rmtree(best_dir)
    shutil.copytree(out / "runs" / f"seed{best['seed']}" / base.env, best_dir)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for seed, curve in curves.items():
        ok = rows[seed]["success"]
        ax.plot(range(1, len(curve) + 1), curve, lw=1.4 if ok else 0.9,
                color="tab:green" if ok else "tab:gray", alpha=0.9 if ok else 0.6)
    ax.axhline(success, ls="--", color="tab:red", lw=1, label=f"success threshold ({success:g})")
    ax.set_xlabel("generation")
    ax.set_ylabel("mean-policy return (full task)")
    ax.set_title(f"{base.env}: {n_seeds} seeds, {int(summary['success_rate'] * n_seeds)} solved "
                 f"(green) vs stuck (gray)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "curves.png", dpi=130)
    plt.close(fig)
    if verbose:
        print(f"success rate {summary['success_rate']:.0%}  median final {summary['final_eval_median']:.1f}  "
              f"best seed {best['seed']} ({best['final_eval']:.1f})")
    return rows, summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", default="pendulum", choices=["drone", "pendulum"])
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--pop", type=int, default=512)
    p.add_argument("--n-eval", type=int, default=4)
    p.add_argument("--gens", type=int, default=300)
    p.add_argument("--sigma", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--weight-decay", type=float, default=0.005)
    p.add_argument("--curriculum-gens", type=int, default=0)
    p.add_argument("--hidden", type=int, nargs="+", default=[32, 32])
    p.add_argument("--score-window", type=int, default=20)
    p.add_argument("--final-episodes", type=int, default=100)
    p.add_argument("--success", type=float, default=None)
    p.add_argument("--out", default="runs/seeds")
    a = p.parse_args(argv)
    base = Config(env=a.env, pop=a.pop, n_eval=a.n_eval, gens=a.gens, sigma=a.sigma, lr=a.lr,
                  weight_decay=a.weight_decay, curriculum_gens=a.curriculum_gens,
                  hidden=tuple(a.hidden), log_every=10, out=a.out)
    run_seeds(base, a.seeds, a.out, a.score_window, a.final_episodes, a.success)


if __name__ == "__main__":
    main()

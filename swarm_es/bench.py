"""Measure how env throughput scales with the number of parallel envs.

    python -m swarm_es.bench --env drone --sizes 1 64 1024 8192 32768

Each env gets its own policy parameters (as in a real ES population), and
all of them run a full episode inside one jitted, vmapped call.
"""
import argparse
import csv
import time
from pathlib import Path

import jax

from .envs import make_env
from .policy import MLPPolicy
from .rollout import rollout


def bench(env_name: str, sizes, repeats: int = 3, out: str = "runs/bench"):
    env = make_env(env_name)
    policy = MLPPolicy(env.obs_dim, env.act_dim)

    @jax.jit
    def run(params, keys):
        rets, _ = jax.vmap(lambda p, k: rollout(env, policy, p, k))(params, keys)
        return rets

    rows = []
    print(f"env={env_name}  device={jax.devices()[0].platform}  episode={env.max_steps} steps")
    print(f"{'envs':>8} {'seconds':>9} {'env-steps/s':>14}")
    for n in sizes:
        key = jax.random.key(n)
        k_noise, k_env = jax.random.split(key)
        params = policy.init_params + 0.05 * jax.random.normal(
            k_noise, (n, policy.num_params)
        )
        keys = jax.random.split(k_env, n)
        run(params, keys).block_until_ready()  # compile

        best = float("inf")
        for _ in range(repeats):
            t0 = time.perf_counter()
            run(params, keys).block_until_ready()
            best = min(best, time.perf_counter() - t0)
        sps = n * env.max_steps / best
        rows.append({"envs": n, "seconds": best, "steps_per_sec": sps})
        print(f"{n:>8,} {best:>9.3f} {sps:>14,.0f}")

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    platform = jax.devices()[0].platform
    with open(out_dir / f"{env_name}_{platform}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog([r["envs"] for r in rows], [r["steps_per_sec"] for r in rows], "o-")
    ax.set_xlabel("parallel envs")
    ax.set_ylabel("env steps / second")
    ax.set_title(f"{env_name} throughput on {platform}")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_dir / f"{env_name}_{platform}.png", dpi=130)
    plt.close(fig)
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env", default="drone", choices=["drone", "pendulum"])
    p.add_argument("--sizes", type=int, nargs="+", default=[1, 64, 1024, 8192, 32768])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--out", default="runs/bench")
    a = p.parse_args(argv)
    bench(a.env, a.sizes, a.repeats, a.out)


if __name__ == "__main__":
    main()

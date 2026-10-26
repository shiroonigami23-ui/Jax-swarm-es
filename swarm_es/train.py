"""Train a policy with evolution strategies over thousands of parallel envs.

    python -m swarm_es.train --env drone --pop 2048 --n-eval 5 --gens 300

pop * n_eval environments run in parallel every generation (10,240 with the
defaults). Generations are grouped into jitted lax.scan chunks, so Python
only wakes up once per `--log-every` generations to print progress.
"""
import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from . import es
from .envs import make_env
from .policy import MLPPolicy
from .rollout import rollout


@dataclass
class Config:
    env: str = "drone"
    pop: int = 2048          # population size (must be even)
    n_eval: int = 5          # episodes per member; pop * n_eval envs in parallel
    gens: int = 300
    sigma: float = 0.05
    lr: float = 0.02
    weight_decay: float = 0.005
    lr_decay: float = 1.0     # lr multiplied by this every generation
    sigma_decay: float = 1.0  # sigma multiplied by this every generation
    sigma_min: float = 0.0    # floor for the decayed sigma
    hidden: tuple = field(default=(32, 32))
    seed: int = 0
    log_every: int = 10
    out: str = "runs"


def make_generation(env, policy, cfg: Config):
    def evaluate(flat, keys):
        rets, infos = jax.vmap(lambda k: rollout(env, policy, flat, k))(keys)
        return rets.mean(), jax.tree_util.tree_map(jnp.mean, infos)

    def generation(state: es.ESState, key):
        k_ask, k_eval, k_test = jax.random.split(key, 3)
        sigma = jnp.maximum(cfg.sigma * cfg.sigma_decay ** state.step, cfg.sigma_min)
        lr = cfg.lr * cfg.lr_decay ** state.step
        population, eps = es.ask(k_ask, state, cfg.pop, sigma)

        # Common random numbers: every member sees the same episodes this
        # generation, so ranking compares policies rather than luck.
        eval_keys = jax.random.split(k_eval, cfg.n_eval)
        fitness, infos = jax.vmap(evaluate, in_axes=(0, None))(population, eval_keys)

        new_state = es.tell(
            state, eps, fitness, sigma, lr, cfg.weight_decay
        )

        # How good is the mean itself, on fresh episodes?
        test_ret, test_info = evaluate(
            new_state.mean, jax.random.split(k_test, cfg.n_eval)
        )
        metrics = {
            "fitness_mean": fitness.mean(),
            "fitness_max": fitness.max(),
            "mean_policy_return": test_ret,
            "lr": lr,
            "sigma": sigma,
        }
        for name, value in test_info.items():
            metrics[f"mean_policy_{name}"] = value
        return new_state, metrics

    return generation


def train(cfg: Config, verbose: bool = True):
    env = make_env(cfg.env)
    policy = MLPPolicy(env.obs_dim, env.act_dim, tuple(cfg.hidden), cfg.seed)
    state = es.init(policy.init_params)
    generation = make_generation(env, policy, cfg)

    @jax.jit
    def run_chunk(state, key):
        keys = jax.random.split(key, cfg.log_every)
        return jax.lax.scan(generation, state, keys)

    if verbose:
        n_envs = cfg.pop * cfg.n_eval
        print(
            f"env={cfg.env}  params={policy.num_params}  pop={cfg.pop}  "
            f"envs/gen={n_envs:,}  steps/gen={n_envs * env.max_steps:,}  "
            f"device={jax.devices()[0].platform}"
        )

    key = jax.random.key(cfg.seed)
    history = []
    t0 = time.perf_counter()
    n_chunks = max(1, cfg.gens // cfg.log_every)
    for chunk in range(n_chunks):
        key, sub = jax.random.split(key)
        state, metrics = run_chunk(state, sub)
        metrics = jax.device_get(metrics)
        for i in range(cfg.log_every):
            row = {"gen": chunk * cfg.log_every + i + 1}
            row.update({k: float(v[i]) for k, v in metrics.items()})
            history.append(row)
        if verbose:
            last = history[-1]
            extras = "  ".join(
                f"{k.removeprefix('mean_policy_')}={v:.2f}"
                for k, v in last.items()
                if k.startswith("mean_policy_") and k != "mean_policy_return"
            )
            print(
                f"gen {last['gen']:5d}  pop_mean {last['fitness_mean']:9.2f}  "
                f"pop_max {last['fitness_max']:9.2f}  "
                f"mean_policy {last['mean_policy_return']:9.2f}  {extras}  "
                f"[{time.perf_counter() - t0:6.1f}s]"
            )

    return policy, state, history


def save_run(cfg: Config, policy, state, history) -> Path:
    out = Path(cfg.out) / cfg.env
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "params.npz", mean=np.asarray(state.mean))
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    with open(out / "history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gens = [r["gen"] for r in history]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(gens, [r["fitness_mean"] for r in history], label="population mean")
    ax.plot(gens, [r["fitness_max"] for r in history], label="population best", alpha=0.6)
    ax.plot(gens, [r["mean_policy_return"] for r in history], label="mean policy (fresh episodes)")
    ax.set_xlabel("generation")
    ax.set_ylabel("episode return")
    ax.set_title(f"{cfg.env}: ES with pop {cfg.pop} x {cfg.n_eval} episodes")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "learning_curve.png", dpi=130)
    plt.close(fig)
    return out


def parse_args(argv=None) -> Config:
    defaults = Config()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", default=defaults.env, choices=["drone", "pendulum"])
    p.add_argument("--pop", type=int, default=defaults.pop)
    p.add_argument("--n-eval", type=int, default=defaults.n_eval)
    p.add_argument("--gens", type=int, default=defaults.gens)
    p.add_argument("--sigma", type=float, default=defaults.sigma)
    p.add_argument("--lr", type=float, default=defaults.lr)
    p.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    p.add_argument("--lr-decay", type=float, default=defaults.lr_decay)
    p.add_argument("--sigma-decay", type=float, default=defaults.sigma_decay)
    p.add_argument("--sigma-min", type=float, default=defaults.sigma_min)
    p.add_argument("--hidden", type=int, nargs="+", default=list(defaults.hidden))
    p.add_argument("--seed", type=int, default=defaults.seed)
    p.add_argument("--log-every", type=int, default=defaults.log_every)
    p.add_argument("--out", default=defaults.out)
    a = p.parse_args(argv)
    return Config(
        env=a.env, pop=a.pop, n_eval=a.n_eval, gens=a.gens, sigma=a.sigma,
        lr=a.lr, weight_decay=a.weight_decay, lr_decay=a.lr_decay,
        sigma_decay=a.sigma_decay, sigma_min=a.sigma_min, hidden=tuple(a.hidden),
        seed=a.seed, log_every=a.log_every, out=a.out,
    )


def main(argv=None):
    cfg = parse_args(argv)
    policy, state, history = train(cfg)
    out = save_run(cfg, policy, state, history)
    print(f"saved params, history and learning curve to {out}/")


if __name__ == "__main__":
    main()

"""Hand-written pendulum controller: a sanity check that the task is solvable.

    python -m swarm_es.baselines --episodes 100

Energy pumping far from upright, a linear (PD) stabiliser near upright. No
learning involved. If this reaches the "solved" range (about -150) on our
Pendulum, then an ES run stuck near -760 is an optimisation problem, not an
environment bug. Writes results/pendulum_controller.json.
"""
import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .envs import Pendulum


def energy_pd_action(env: Pendulum, s) -> jax.Array:
    angle = ((s.theta + jnp.pi) % (2 * jnp.pi)) - jnp.pi          # 0 = upright
    inertia = env.m * env.l**2 / 3.0
    # Energy relative to resting upright: kinetic + potential (centre of mass at l/2).
    energy = 0.5 * inertia * s.theta_dot**2 + env.m * env.g * env.l / 2 * (jnp.cos(angle) - 1.0)
    pump = jnp.clip(-3.0 * energy * s.theta_dot, -1.0, 1.0)       # drive energy to zero
    pd = jnp.clip(-(8.0 * angle + 1.5 * s.theta_dot) / 2.0, -1.0, 1.0)
    return jnp.where(jnp.abs(angle) < 0.6, pd, pump)[None]


def episode_return(env: Pendulum, key) -> jax.Array:
    state = env.reset(key)

    def body(carry, _):
        s, ret = carry
        s, r, _ = env.step(s, energy_pd_action(env, s), key)
        return (s, ret + r), None

    (_, ret), _ = jax.lax.scan(body, (state, jnp.float32(0.0)), None, length=env.max_steps)
    return ret


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--out", default="results/pendulum_controller.json")
    a = p.parse_args(argv)
    env = Pendulum()
    keys = jax.random.split(jax.random.key(a.seed), a.episodes)
    rets = np.asarray(jax.jit(jax.vmap(lambda k: episode_return(env, k)))(keys))
    summary = {"controller": "energy pumping + PD near upright", "episodes": a.episodes,
               "mean": float(rets.mean()), "std": float(rets.std()), "median": float(np.median(rets)),
               "min": float(rets.min()), "max": float(rets.max())}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

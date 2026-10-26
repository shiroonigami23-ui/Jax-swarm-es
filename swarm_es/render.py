"""Render a trained policy to a GIF.

    python -m swarm_es.render --run runs/drone --episodes 3
"""
import argparse
import json
from pathlib import Path

import jax
import numpy as np

from .envs import make_env
from .policy import MLPPolicy
from .rollout import rollout_trajectory


def _draw_drone(ax, env, states, alive, frames):
    import matplotlib.patches as patches

    pos = np.asarray(states.pos)
    phi = np.asarray(states.phi)
    target = np.asarray(states.target)
    hits = np.asarray(states.hits)

    ax.set_xlim(-env.x_limit, env.x_limit)
    ax.set_ylim(env.y_low, env.y_high)
    ax.set_aspect("equal")
    ax.add_patch(
        patches.Rectangle(
            (-env.x_limit, env.y_low), 2 * env.x_limit, env.y_high - env.y_low,
            fill=False, lw=1.5,
        )
    )
    (trail,) = ax.plot([], [], lw=1, alpha=0.5)
    (body,) = ax.plot([], [], lw=3, color="black")
    (rotors,) = ax.plot([], [], "o", color="tab:red", ms=5)
    (goal,) = ax.plot([], [], "*", color="gold", ms=18, mec="black")
    label = ax.text(-env.x_limit + 0.2, env.y_high - 0.5, "", fontsize=10)
    half = env.arm * 2.0  # drawn larger than life so it is visible

    def update(i):
        x, y = pos[i]
        dx, dy = half * np.cos(phi[i]), half * np.sin(phi[i])
        body.set_data([x - dx, x + dx], [y - dy, y + dy])
        rotors.set_data([x - dx, x + dx], [y - dy, y + dy])
        lo = max(0, i - 150)
        trail.set_data(pos[lo : i + 1, 0], pos[lo : i + 1, 1])
        goal.set_data([target[i, 0]], [target[i, 1]])
        status = "" if alive[i] > 0 else "  CRASHED"
        label.set_text(f"t={i * env.dt:4.1f}s  targets={hits[i]}{status}")
        return trail, body, rotors, goal, label

    return update


def _draw_pendulum(ax, env, states, alive, frames):
    theta = np.asarray(states.theta)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    (rod,) = ax.plot([], [], lw=4, color="tab:blue")
    ax.plot([0], [0], "ko")
    label = ax.text(-1.2, 1.15, "", fontsize=10)

    def update(i):
        # theta = 0 is upright
        x, y = np.sin(theta[i]), np.cos(theta[i])
        rod.set_data([0, x], [0, y])
        label.set_text(f"t={i * env.dt:4.1f}s")
        return rod, label

    return update


def render(run_dir: str, episodes: int = 1, seed: int = 123, stride: int = 2) -> list:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    run = Path(run_dir)
    cfg = json.loads((run / "config.json").read_text())
    env = make_env(cfg["env"])
    policy = MLPPolicy(env.obs_dim, env.act_dim, tuple(cfg["hidden"]), cfg["seed"])
    params = np.load(run / "params.npz")["mean"]

    traj = jax.jit(lambda k: rollout_trajectory(env, policy, params, k))
    drawers = {"drone": _draw_drone, "pendulum": _draw_pendulum}

    paths = []
    for ep, key in enumerate(jax.random.split(jax.random.key(seed), episodes)):
        states, alive = jax.device_get(traj(key))
        frames = list(range(0, env.max_steps, stride))
        if alive[-1] == 0:  # stop shortly after a crash
            frames = [f for f in frames if f <= int(np.argmin(alive)) + 25]

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_title(f"{env.name} episode {ep}")
        update = drawers[env.name](ax, env, states, alive, frames)
        anim = FuncAnimation(fig, update, frames=frames, blit=True)
        path = run / f"episode_{ep}.gif"
        anim.save(path, writer=PillowWriter(fps=int(1 / (env.dt * stride))))
        plt.close(fig)
        paths.append(path)
        print(f"wrote {path}")
    return paths


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default="runs/drone")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--stride", type=int, default=2)
    a = p.parse_args(argv)
    render(a.run, a.episodes, a.seed, a.stride)


if __name__ == "__main__":
    main()

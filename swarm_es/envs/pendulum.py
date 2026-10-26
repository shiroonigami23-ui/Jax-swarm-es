"""Classic pendulum swing-up, rewritten as pure JAX functions.

Same dynamics and cost as Gym's Pendulum-v1, so returns are comparable:
a random policy scores around -1200, a solved one around -150.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp


class PendulumState(NamedTuple):
    theta: jax.Array
    theta_dot: jax.Array
    t: jax.Array


class Pendulum:
    name = "pendulum"
    obs_dim = 3
    act_dim = 1
    max_steps = 200

    g = 10.0
    m = 1.0
    l = 1.0
    dt = 0.05
    max_speed = 8.0
    max_torque = 2.0

    def reset(self, key: jax.Array) -> PendulumState:
        k1, k2 = jax.random.split(key)
        theta = jax.random.uniform(k1, (), minval=-jnp.pi, maxval=jnp.pi)
        theta_dot = jax.random.uniform(k2, (), minval=-1.0, maxval=1.0)
        return PendulumState(theta, theta_dot, jnp.int32(0))

    def obs(self, s: PendulumState) -> jax.Array:
        return jnp.stack(
            [jnp.cos(s.theta), jnp.sin(s.theta), s.theta_dot / self.max_speed]
        )

    def step(self, s: PendulumState, action: jax.Array, key: jax.Array):
        del key  # deterministic dynamics
        u = jnp.clip(action[0], -1.0, 1.0) * self.max_torque
        angle = ((s.theta + jnp.pi) % (2 * jnp.pi)) - jnp.pi
        cost = angle**2 + 0.1 * s.theta_dot**2 + 0.001 * u**2

        theta_dot = s.theta_dot + (
            3 * self.g / (2 * self.l) * jnp.sin(s.theta)
            + 3.0 / (self.m * self.l**2) * u
        ) * self.dt
        theta_dot = jnp.clip(theta_dot, -self.max_speed, self.max_speed)
        theta = s.theta + theta_dot * self.dt

        new_state = PendulumState(theta, theta_dot, s.t + 1)
        return new_state, -cost, jnp.float32(0.0)

    def episode_info(self, s: PendulumState) -> dict:
        return {}

"""Planar quadrotor that has to chase waypoints through gusty wind.

An original environment written as pure JAX functions, so thousands of
copies can be stepped in parallel with jax.vmap.

The drone has two rotors (left, right). Each action component sets one
rotor's thrust relative to hover: a = 0 hovers, a = +1 doubles thrust,
a = -1 cuts it. Spinning the rotors unevenly tilts the body, which is the
only way to move sideways. Wind is an Ornstein-Uhlenbeck process pushing
along x, so the policy has to keep correcting.

Each time the drone gets within `reach_radius` of the target it scores a
bonus and a new target appears. Leaving the arena or tipping past 90
degrees counts as a crash and ends the episode.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp


class DroneState(NamedTuple):
    pos: jax.Array      # (2,) x, y in metres
    vel: jax.Array      # (2,)
    phi: jax.Array      # body angle, counter-clockwise positive
    omega: jax.Array    # angular velocity
    wind: jax.Array     # wind acceleration along x
    target: jax.Array   # (2,)
    hits: jax.Array     # targets reached so far
    t: jax.Array


class Drone:
    name = "drone"
    obs_dim = 8
    act_dim = 2
    max_steps = 500  # 10 seconds at dt = 0.02

    dt = 0.02
    mass = 1.0
    g = 9.81
    arm = 0.25
    inertia = 0.02
    drag = 0.3

    wind_theta = 0.5
    x_limit = 4.0
    y_low = 0.0
    y_high = 8.0
    reach_radius = 0.3

    def __init__(self, wind_sigma: float = 1.5):
        self.wind_sigma = wind_sigma

    def _sample_target(self, key: jax.Array) -> jax.Array:
        kx, ky = jax.random.split(key)
        x = jax.random.uniform(kx, (), minval=-3.0, maxval=3.0)
        y = jax.random.uniform(ky, (), minval=1.0, maxval=6.0)
        return jnp.stack([x, y])

    def reset(self, key: jax.Array) -> DroneState:
        k_pos, k_target = jax.random.split(key)
        pos = jnp.array([0.0, 2.0]) + jax.random.uniform(
            k_pos, (2,), minval=-0.5, maxval=0.5
        )
        zero = jnp.float32(0.0)
        return DroneState(
            pos=pos,
            vel=jnp.zeros(2),
            phi=zero,
            omega=zero,
            wind=zero,
            target=self._sample_target(k_target),
            hits=jnp.int32(0),
            t=jnp.int32(0),
        )

    def obs(self, s: DroneState) -> jax.Array:
        rel = s.target - s.pos
        return jnp.concatenate(
            [
                rel / 4.0,
                s.vel / 4.0,
                jnp.stack(
                    [jnp.sin(s.phi), jnp.cos(s.phi), s.omega / 10.0, s.wind / 5.0]
                ),
            ]
        )

    def step(self, s: DroneState, action: jax.Array, key: jax.Array):
        k_wind, k_target = jax.random.split(key)
        a = jnp.clip(action, -1.0, 1.0)

        hover = self.mass * self.g / 2.0
        thrust = hover * (1.0 + a)            # (left, right)
        total = thrust[0] + thrust[1]
        torque = (thrust[1] - thrust[0]) * self.arm

        direction = jnp.stack([-jnp.sin(s.phi), jnp.cos(s.phi)])
        acc = (
            total / self.mass * direction
            + jnp.stack([s.wind, -self.g])
            - self.drag * s.vel
        )

        wind = (
            s.wind
            - self.wind_theta * s.wind * self.dt
            + self.wind_sigma * jnp.sqrt(self.dt) * jax.random.normal(k_wind, ())
        )

        # semi-implicit Euler
        vel = s.vel + acc * self.dt
        pos = s.pos + vel * self.dt
        omega = s.omega + torque / self.inertia * self.dt
        phi = s.phi + omega * self.dt

        dist = jnp.linalg.norm(s.target - pos)
        reached = dist < self.reach_radius
        target = jnp.where(reached, self._sample_target(k_target), s.target)

        crashed = (
            (pos[1] < self.y_low)
            | (pos[1] > self.y_high)
            | (jnp.abs(pos[0]) > self.x_limit)
            | (jnp.abs(phi) > jnp.pi / 2)
        )

        # Every surviving step is worth something, so crashing early is
        # never a shortcut out of distance penalties.
        reward = (
            0.02
            + 0.2 * jnp.exp(-dist)
            - 0.002 * omega**2
            - 0.01 * jnp.sum(a**2)
            + 5.0 * reached
            - 5.0 * crashed
        )

        new_state = DroneState(
            pos=pos,
            vel=vel,
            phi=phi,
            omega=omega,
            wind=wind,
            target=target,
            hits=s.hits + reached.astype(jnp.int32),
            t=s.t + 1,
        )
        return new_state, reward.astype(jnp.float32), crashed.astype(jnp.float32)

    def episode_info(self, s: DroneState) -> dict:
        return {"hits": s.hits.astype(jnp.float32)}

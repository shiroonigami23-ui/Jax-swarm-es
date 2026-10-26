"""OpenAI-style evolution strategies (Salimans et al., 2017).

The search distribution is an isotropic Gaussian around `mean`. Each
generation:

1. ask:  sample antithetic noise eps and -eps, giving a population
         mean + sigma * eps.
2. evaluate every member (done elsewhere, in parallel).
3. tell: turn raw fitness into centered ranks, estimate the gradient of
         expected fitness as eps^T ranks / (n * sigma), and take an Adam
         step uphill.

Everything here is a pure function on an ESState, so a whole training run
can sit inside jax.jit / lax.scan.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp


class ESState(NamedTuple):
    mean: jax.Array
    m: jax.Array      # Adam first moment
    v: jax.Array      # Adam second moment
    step: jax.Array


def init(mean: jax.Array) -> ESState:
    zeros = jnp.zeros_like(mean)
    return ESState(mean=mean, m=zeros, v=zeros, step=jnp.float32(0.0))


def ask(key, state: ESState, pop_size: int, sigma: float):
    if pop_size % 2:
        raise ValueError("pop_size must be even for antithetic sampling")
    half = jax.random.normal(key, (pop_size // 2, state.mean.shape[0]))
    eps = jnp.concatenate([half, -half], axis=0)
    return state.mean + sigma * eps, eps


def centered_ranks(x: jax.Array) -> jax.Array:
    """Map fitness to evenly spaced values in [-0.5, 0.5] by rank.

    Makes the update invariant to the scale of the reward and robust to
    a few outlier episodes.
    """
    ranks = jnp.argsort(jnp.argsort(x)).astype(jnp.float32)
    return ranks / (x.shape[0] - 1) - 0.5


def tell(
    state: ESState,
    eps: jax.Array,
    fitness: jax.Array,
    sigma: float,
    lr: float,
    weight_decay: float = 0.0,
    b1: float = 0.9,
    b2: float = 0.999,
    adam_eps: float = 1e-8,
) -> ESState:
    shaped = centered_ranks(fitness)
    grad = eps.T @ shaped / (eps.shape[0] * sigma)   # ascent direction
    grad = grad - weight_decay * state.mean

    step = state.step + 1.0
    m = b1 * state.m + (1 - b1) * grad
    v = b2 * state.v + (1 - b2) * grad**2
    m_hat = m / (1 - b1**step)
    v_hat = v / (1 - b2**step)
    mean = state.mean + lr * m_hat / (jnp.sqrt(v_hat) + adam_eps)
    return ESState(mean=mean, m=m, v=v, step=step)

"""A small tanh MLP whose weights live in one flat vector.

Evolution strategies perturb parameters directly, so it is convenient to
treat the whole network as a single 1-D array. `ravel_pytree` gives us the
flat vector plus a function that turns it back into layers.
"""
from typing import Sequence

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree


def _init_layers(key, sizes: Sequence[int]):
    layers = []
    keys = jax.random.split(key, len(sizes) - 1)
    for i, (k, fan_in, fan_out) in enumerate(zip(keys, sizes[:-1], sizes[1:])):
        w = jax.random.normal(k, (fan_in, fan_out)) / jnp.sqrt(fan_in)
        if i == len(sizes) - 2:
            w = w * 0.1  # start near zero action (hover / no torque)
        layers.append((w, jnp.zeros(fan_out)))
    return layers


class MLPPolicy:
    def __init__(self, obs_dim: int, act_dim: int, hidden=(32, 32), seed: int = 0):
        sizes = [obs_dim, *hidden, act_dim]
        layers = _init_layers(jax.random.key(seed), sizes)
        self.init_params, self._unravel = ravel_pytree(layers)
        self.num_params = int(self.init_params.shape[0])
        self.sizes = sizes

    def apply(self, flat_params: jax.Array, obs: jax.Array) -> jax.Array:
        layers = self._unravel(flat_params)
        x = obs
        for w, b in layers[:-1]:
            x = jnp.tanh(x @ w + b)
        w, b = layers[-1]
        return jnp.tanh(x @ w + b)

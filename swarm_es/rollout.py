"""Running one episode as a single lax.scan.

Because the episode is a fixed-length scan of pure functions, jax.vmap can
turn "one episode" into "ten thousand episodes" with no code changes.
Episodes that end early are frozen in place: their state stops updating
and further rewards are masked out.
"""
import jax
import jax.numpy as jnp


def _freeze(alive, new, old):
    return jax.tree_util.tree_map(lambda n, o: jnp.where(alive > 0, n, o), new, old)


def rollout(env, policy, flat_params, key):
    """Return (episode_return, episode_info) for one episode."""
    k_reset, k_steps = jax.random.split(key)
    state = env.reset(k_reset)

    def body(carry, k):
        state, ret, alive = carry
        action = policy.apply(flat_params, env.obs(state))
        new_state, reward, done = env.step(state, action, k)
        ret = ret + reward * alive
        state = _freeze(alive, new_state, state)
        alive = alive * (1.0 - done)
        return (state, ret, alive), None

    keys = jax.random.split(k_steps, env.max_steps)
    init = (state, jnp.float32(0.0), jnp.float32(1.0))
    (state, ret, _), _ = jax.lax.scan(body, init, keys)
    return ret, env.episode_info(state)


def rollout_trajectory(env, policy, flat_params, key):
    """Same episode, but keep every state for rendering.

    Returns (states, alive) where each leaf of `states` has a leading time
    axis of length env.max_steps.
    """
    k_reset, k_steps = jax.random.split(key)
    state = env.reset(k_reset)

    def body(carry, k):
        state, alive = carry
        action = policy.apply(flat_params, env.obs(state))
        new_state, _, done = env.step(state, action, k)
        state = _freeze(alive, new_state, state)
        alive = alive * (1.0 - done)
        return (state, alive), (state, alive)

    keys = jax.random.split(k_steps, env.max_steps)
    _, (states, alive) = jax.lax.scan(body, (state, jnp.float32(1.0)), keys)
    return states, alive

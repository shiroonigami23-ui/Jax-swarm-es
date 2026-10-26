import jax
import jax.numpy as jnp
import pytest

from swarm_es import MLPPolicy, make_env, rollout
from swarm_es.envs import Drone


@pytest.mark.parametrize("name", ["drone", "pendulum"])
def test_reset_step_shapes_and_finite(name):
    env = make_env(name)
    state = env.reset(jax.random.key(0))
    assert env.obs(state).shape == (env.obs_dim,)

    step = jax.jit(env.step)
    key = jax.random.key(1)
    for _ in range(50):
        key, k_act, k_step = jax.random.split(key, 3)
        action = jax.random.uniform(k_act, (env.act_dim,), minval=-1, maxval=1)
        state, reward, done = step(state, action, k_step)
        assert reward.shape == () and done.shape == ()
        assert jnp.all(jnp.isfinite(env.obs(state)))


def test_drone_hovers_with_zero_action_and_no_wind():
    env = Drone(wind_sigma=0.0)
    state = env.reset(jax.random.key(0))
    start_y = state.pos[1]
    step = jax.jit(env.step)
    for i in range(100):
        state, _, done = step(state, jnp.zeros(2), jax.random.key(i))
        assert done == 0
    # Thrust exactly cancels gravity, so it should barely move.
    assert abs(float(state.pos[1] - start_y)) < 1e-3
    assert abs(float(state.phi)) < 1e-6


def test_drone_crashes_when_rotors_cut():
    env = Drone(wind_sigma=0.0)
    state = env.reset(jax.random.key(0))
    crashed = False
    for i in range(env.max_steps):
        state, _, done = env.step(state, -jnp.ones(2), jax.random.key(i))
        if done:
            crashed = True
            break
    assert crashed


def test_drone_uneven_thrust_tilts_body():
    env = Drone(wind_sigma=0.0)
    state = env.reset(jax.random.key(0))
    # Right rotor stronger -> counter-clockwise tilt -> drifts left.
    for i in range(20):
        state, _, _ = env.step(state, jnp.array([-0.05, 0.05]), jax.random.key(i))
    assert state.phi > 0
    assert state.vel[0] < 0


def test_vmapped_rollouts_match_single():
    env = make_env("drone")
    policy = MLPPolicy(env.obs_dim, env.act_dim, (8,))
    keys = jax.random.split(jax.random.key(3), 4)
    single = [rollout(env, policy, policy.init_params, k)[0] for k in keys]
    batched, info = jax.jit(
        jax.vmap(lambda k: rollout(env, policy, policy.init_params, k))
    )(keys)
    assert batched.shape == (4,)
    assert info["hits"].shape == (4,)
    assert jnp.allclose(jnp.stack(single), batched, rtol=1e-3, atol=1e-2)

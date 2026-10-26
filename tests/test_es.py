import jax
import jax.numpy as jnp

from swarm_es import es
from swarm_es.train import Config, train


def test_centered_ranks():
    r = es.centered_ranks(jnp.array([10.0, -3.0, 7.0, 100.0]))
    assert jnp.allclose(r, jnp.array([1 / 6, -0.5, -1 / 6, 0.5]))


def test_ask_is_antithetic():
    state = es.init(jnp.zeros(5))
    pop, eps = es.ask(jax.random.key(0), state, 8, 0.1)
    assert pop.shape == (8, 5)
    assert jnp.allclose(eps[:4], -eps[4:])


def test_es_climbs_a_quadratic():
    target = 3.0
    fitness_fn = jax.vmap(lambda x: -jnp.sum((x - target) ** 2))

    @jax.jit
    def gen(state, key):
        pop, eps = es.ask(key, state, 64, 0.1)
        return es.tell(state, eps, fitness_fn(pop), sigma=0.1, lr=0.05), None

    state = es.init(jnp.zeros(10))
    state, _ = jax.lax.scan(gen, state, jax.random.split(jax.random.key(0), 400))
    assert jnp.max(jnp.abs(state.mean - target)) < 0.3


def test_train_smoke_pendulum():
    cfg = Config(env="pendulum", pop=16, n_eval=2, gens=4, log_every=2, hidden=(8,))
    policy, state, history = train(cfg, verbose=False)
    assert len(history) == 4
    assert state.mean.shape == (policy.num_params,)
    assert all(jnp.isfinite(r["mean_policy_return"]) for r in history)


def test_train_smoke_drone():
    cfg = Config(env="drone", pop=8, n_eval=2, gens=2, log_every=2, hidden=(8,))
    _, _, history = train(cfg, verbose=False)
    assert "mean_policy_hits" in history[0]

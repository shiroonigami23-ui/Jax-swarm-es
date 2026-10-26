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


def test_lr_and_sigma_decay_schedules():
    cfg = Config(env="pendulum", pop=8, n_eval=1, gens=4, log_every=2, hidden=(4,),
                 lr=0.1, lr_decay=0.5, sigma=0.2, sigma_decay=0.5, sigma_min=0.04)
    _, _, history = train(cfg, verbose=False)
    lrs = [r["lr"] for r in history]
    sigmas = [r["sigma"] for r in history]
    assert jnp.allclose(jnp.array(lrs), jnp.array([0.1, 0.05, 0.025, 0.0125]))
    assert jnp.allclose(jnp.array(sigmas), jnp.array([0.2, 0.1, 0.05, 0.04]))


def test_sweep_smoke(tmp_path):
    from swarm_es.sweep import run_sweep

    rows = run_sweep("pendulum", pop=8, n_eval=1, gens=10, lrs=[0.01, 0.02],
                     sigmas=[0.1], lr_decays=[1.0], score_window=5,
                     out=tmp_path, verbose=False)
    assert len(rows) == 2
    assert rows[0]["score"] >= rows[1]["score"]
    assert (tmp_path / "sweep.csv").exists() and (tmp_path / "best.json").exists()


def test_train_smoke_drone():
    cfg = Config(env="drone", pop=8, n_eval=2, gens=2, log_every=2, hidden=(8,))
    _, _, history = train(cfg, verbose=False)
    assert "mean_policy_hits" in history[0]

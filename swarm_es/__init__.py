"""swarm-es: evolution strategies over thousands of vmapped JAX environments."""
from . import es
from .envs import Drone, Pendulum, make_env
from .policy import MLPPolicy
from .rollout import rollout, rollout_trajectory

__version__ = "0.1.0"
__all__ = ["es", "Drone", "Pendulum", "make_env", "MLPPolicy", "rollout", "rollout_trajectory"]

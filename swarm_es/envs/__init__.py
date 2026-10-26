from .drone import Drone, DroneState
from .pendulum import Pendulum, PendulumState

ENVS = {"drone": Drone, "pendulum": Pendulum}


def make_env(name: str, **kwargs):
    if name not in ENVS:
        raise ValueError(f"unknown env {name!r}, choose from {sorted(ENVS)}")
    return ENVS[name](**kwargs)


__all__ = ["Drone", "DroneState", "Pendulum", "PendulumState", "ENVS", "make_env"]

#!/usr/bin/env python3
"""Utilities."""
import pickle

from jax import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402


def load_dataset(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    initial_states = jnp.array([d["x0"] for d in data])
    controls = jnp.array([d["ctrl_traj"] for d in data])
    return initial_states, controls


def rollout(initial_state, controls, dynamics, dynamics_params) -> jnp.ndarray:
    horizon = controls.shape[0]
    dt = dynamics_params["discretization_resolution"]

    states = jnp.zeros((horizon, len(initial_state)))
    states = states.at[0].set(initial_state)
    for t in range(horizon - 1):
        next_state = states[t] + dt * dynamics.state_dot(
            states[t], controls[t], dynamics_params
        )
        states = states.at[t + 1].set(next_state)
    return states


def initial_weights_penalization_reference_state_trajectory():
    # generated using np.random.uniform(low=0.5, high=2.5, size=(20,4))
    weights = jnp.array(
        [
            [2.31993711, 2.10973335, 1.27643918, 2.40638906],
            [2.16305628, 0.92978235, 0.8425473, 2.05522213],
            [0.54511473, 2.38979126, 1.82534442, 1.79607804],
            [1.52602309, 0.70958412, 2.28500543, 1.79144837],
            [1.47695067, 2.37700071, 1.10622824, 1.66791643],
            [1.77999985, 2.43335516, 2.13276163, 2.25013024],
            [1.08672352, 1.89562108, 2.44455541, 0.61613992],
            [0.91772376, 0.69279309, 1.3126017, 1.75713809],
            [2.35407733, 2.33764708, 0.71646967, 0.86020509],
            [1.98209716, 1.58648594, 1.59129719, 1.04037999],
            [1.6906858, 1.59305244, 2.1575869, 1.14175977],
            [2.25656044, 1.08790578, 1.8911752, 2.28738797],
            [1.71570255, 1.32924091, 0.6220067, 1.7408397],
            [0.95953363, 1.53826163, 1.24846342, 1.79400116],
            [1.12411295, 0.93086439, 2.40280909, 2.35137107],
            [1.99076212, 1.63976281, 1.6997142, 1.12886317],
            [2.20034932, 0.60009316, 0.79487412, 2.17139842],
            [2.07377772, 1.74829048, 1.48310797, 1.67964389],
            [0.82834224, 1.09670807, 1.86061252, 0.90516967],
            [0.75138763, 1.01916688, 0.80239714, 2.49789644],
        ]
    )
    return weights

"""Checks the dynamics functions."""
import copy

import jax.numpy as jnp
from jax import config

from diffmpc.dynamics.linear_dynamics import (
    LinearDynamics,
    default_parameters,
    default_state_dot_parameters,
)

config.update("jax_enable_x64", True)


def test_linear_state_dot_dimensions():
    """ "
    Tests state and control dimensions of Keisuke dynamics with path.
    """
    model = LinearDynamics(default_parameters)
    x = jnp.ones(model.num_states)
    u = jnp.ones(model.num_controls)

    dynamics_state_dot_params = copy.deepcopy(default_state_dot_parameters)
    x_dot = model.state_dot(x, u, dynamics_state_dot_params)
    assert len(x_dot) == model.num_states

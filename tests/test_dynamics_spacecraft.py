"""Checks the dynamics functions."""
import copy

import jax.numpy as jnp
from jax import config

from diffmpc.dynamics.spacecraft_dynamics import (
    SpacecraftDynamics,
    spacecraft_parameters,
    spacecraft_state_dot_parameters,
)

config.update("jax_enable_x64", True)


def test_spacecraft_state_dot_dimensions():
    """ "
    Tests state and control dimensions of Keisuke dynamics with path.
    """
    model = SpacecraftDynamics(spacecraft_parameters)
    x = jnp.ones(model.num_states)
    u = jnp.ones(model.num_controls)

    dynamics_state_dot_params = copy.deepcopy(spacecraft_state_dot_parameters)
    x_dot = model.state_dot(x, u, dynamics_state_dot_params)
    assert len(x_dot) == model.num_states

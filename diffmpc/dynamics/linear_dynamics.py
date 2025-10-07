"""Linear dynamics class"""

from typing import Any, Dict

import jax.numpy as jnp

from diffmpc.dynamics.base_dynamics import Dynamics

default_parameters: Dict[str, Any] = {
    "num_states": 4,
    "num_controls": 2,
    "names_states": ["pos_x", "pos_y", "vel_x", "vel_y"],
    "names_controls": ["force_x", "force_x"],
}
default_state_dot_parameters: Dict[str, Any] = {
    "A": jnp.array(
        [
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    ),
    "B": jnp.array([[0, 0], [0, 0], [1, 0], [0, 1]]),
    "b": jnp.array([0, 0, 0, 0]),
}


class LinearDynamics(Dynamics):
    """
    Linear dynamics class

    state_dot = A @ state + B @ control
    """

    def __init__(self, parameters: Dict[str, Any] = default_parameters):
        """
        Initializes the class.

        Args:
            parameters:  parameters of the class.
                (str, Any) dictionary
        """
        super().__init__(parameters)

    def state_dot(
        self,
        state: jnp.array,
        control: jnp.array,
        params: Dict[str, Any] = default_state_dot_parameters,
    ) -> jnp.array:
        """
        Computes the time derivative of the state of the system.

        Returns x_dot = f(x, u) where f describes the dynamics of the system.

        Args:
            state: state of the system (see names_states)
                (_num_states, ) array
            control: control input applied to the system (see names_controls)
                (_num_controls, ) array
            params: parameters of the state_dot function of the dynamics.
                (str, Any) dictionary

        Returns:
            state_dot: time derivative of the state
                (_num_states, ) array
        """
        state_dot = params["A"] @ state + params["B"] @ control + params["b"]
        return state_dot

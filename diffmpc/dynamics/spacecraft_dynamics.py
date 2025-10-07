"""Spacecraft dynamics class."""

from typing import Any, Dict

import jax.numpy as jnp

from diffmpc.dynamics.base_dynamics import Dynamics

spacecraft_parameters: Dict[str, Any] = {
    "num_states": 3,
    "num_controls": 3,
    "names_states": ["omega_x", "omega_y", "omega_z"],
    "names_controls": ["torque_x", "torque_y", "torque_z"],
}
spacecraft_state_dot_parameters = {"inertia_vector": jnp.array([5.0, 2.0, 1.0])}


class SpacecraftDynamics(Dynamics):
    """Spacecraft dynamics class."""

    def __init__(self, parameters: Dict[str, Any] = spacecraft_parameters):
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
        state_dot_params: Dict[str, Any] = spacecraft_state_dot_parameters,
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
        inertia = state_dot_params["inertia_vector"]
        inertia_inverse = 1.0 / inertia
        inertia = jnp.diag(inertia)
        inertia_inverse = jnp.diag(inertia_inverse)

        omega = state
        ox, oy, oz = omega
        omega_cross = jnp.array([[0, -oz, oy], [oz, 0, -ox], [-oy, ox, 0]])

        omega_dot = inertia_inverse @ (control - omega_cross @ inertia @ omega)

        state_dot = omega_dot
        return state_dot

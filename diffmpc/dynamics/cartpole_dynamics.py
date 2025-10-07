"""Nonlinear CartPole dynamics (continuous time)."""

from typing import Any, Dict

import jax.numpy as jnp

from diffmpc.dynamics.base_dynamics import Dynamics

default_parameters: Dict[str, Any] = {
    "num_states": 4,  # [x, x_dot, theta, theta_dot]
    "num_controls": 1,  # [force]
    "names_states": ["x", "x_dot", "theta", "theta_dot"],
    "names_controls": ["force"],
}

default_state_dot_parameters: Dict[str, Any] = {
    # Physical parameters
    "masscart": 1.0,
    "masspole": 0.1,
    "length": 0.5,  # half pole length (distance to center of mass)
    "gravity": 9.81,
}


class CartpoleDynamics(Dynamics):
    """
    Continuous-time cart-pole dynamics using the standard formulation.

    State:  [x, x_dot, theta, theta_dot]
    Control: [force]
    """

    def __init__(self, parameters: Dict[str, Any] = default_parameters):
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
        x, x_dot, theta, theta_dot = state
        force = control[0]

        m_c = params.get("masscart", 1.0)
        m_p = params.get("masspole", 0.1)
        length = params.get("length", 0.5)
        g = params.get("gravity", 9.81)

        total_mass = m_c + m_p
        polemass_length = m_p * length

        sin_t = jnp.sin(theta)
        cos_t = jnp.cos(theta)

        # Standard continuous-time dynamics (as in Gym's CartPoleContinuous)
        temp = (force + polemass_length * theta_dot**2 * sin_t) / total_mass
        theta_acc = (g * sin_t - cos_t * temp) / (
            length * (4.0 / 3.0 - (m_p * cos_t**2) / total_mass)
        )
        x_acc = temp - polemass_length * theta_acc * cos_t / total_mass

        return jnp.array([x_dot, x_acc, theta_dot, theta_acc])

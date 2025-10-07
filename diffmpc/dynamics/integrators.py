"""Numerical integration functions."""

import enum
from typing import Dict

import jax.numpy as jnp
from jax.lax import scan

from diffmpc.dynamics.base_dynamics import Dynamics


class DiscretizationScheme(enum.IntEnum):
    """Choice of discretization scheme."""

    EULER = 0
    MIDPOINT = 1
    RUNGEKUTTA4 = 2


def predict_next_state(
    dynamics: Dynamics,
    dt: float,
    discretization_scheme: DiscretizationScheme,
    dynamics_state_dot_params: Dict[str, jnp.array],
    state: jnp.array,
    control: jnp.array,
) -> jnp.array:
    """
    Predicts the next state from the current state in a single integration step.

    Args:
        dynamics: dynamics class
            Dynamics class
        dt: discretization resolution
            (float)
        discretization_scheme: choice of discretization scheme
            (DiscretizationScheme)
        dynamics_state_dot_params: parameters for the state_dot function of the dynamics
            (key=string, value=jnp.array(parameter_size))
            where each key is an argument of the function state_dot of the dynamics
        state: state variables
            (_num_state_variables, ) array
        control: control input variable
            (_num_control_variables, ) array

    Returns:
        next_state: prediction for the next state
            (_num_state_variables, ) array
    """
    # if discretization_scheme == DiscretizationScheme.EULER:
    state_next = state + dt * dynamics.state_dot(
        state, control, dynamics_state_dot_params
    )
    if discretization_scheme == DiscretizationScheme.MIDPOINT:
        state_mid = state + 0.5 * dt * dynamics.state_dot(
            state, control, dynamics_state_dot_params
        )
        state_next = state + dt * dynamics.state_dot(
            state_mid, control, dynamics_state_dot_params
        )
    elif discretization_scheme == DiscretizationScheme.RUNGEKUTTA4:
        k1 = dynamics.state_dot(state, control, dynamics_state_dot_params)
        k2 = dynamics.state_dot(
            state + 0.5 * dt * k1, control, dynamics_state_dot_params
        )
        k3 = dynamics.state_dot(
            state + 0.5 * dt * k2, control, dynamics_state_dot_params
        )
        k4 = dynamics.state_dot(state + dt * k3, control, dynamics_state_dot_params)
        state_next = state + (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4) * dt
    return state_next


def get_state_trajectory(
    dynamics: Dynamics,
    dt: float,
    discretization_scheme: DiscretizationScheme,
    discretization_num_substeps: int,
    dynamics_state_dot_params: Dict[str, jnp.array],
    initial_state: jnp.array,
    control_matrix: jnp.array,
):
    """
    Predicts the next state from the current state using multiple integration steps

    Args:
        dynamics: dynamics class
            Dynamics class
        dt: discretization resolution
            (float)
        discretization_scheme: choice of discretization scheme
            (DiscretizationScheme)
        discretization_num_substeps: number of integration substeps
            (int)
        dynamics_state_dot_params: parameters for the state_dot function of the dynamics
            (key=string, value=jnp.array(horizon, parameter_size))
            where each key is an argument of the function state_dot of the dynamics
        initial state: initial state variables
            (_num_state_variables, ) array
        control_matrix: control input trajectory
            (horizon, _num_control_variables, ) array

    Returns:
        state_matrix: prediction for the state trajectory
            (horizon+1, _num_state_variables, ) array
    """

    def next_state_scan(state, everything):
        state_next = predict_next_state(
            dynamics,
            dt,
            discretization_scheme,
            discretization_num_substeps,
            everything["dynamics_state_dot_params"],
            state,
            everything["control"],
        )
        return state_next, state_next

    everythings = {
        "control": control_matrix[:-1],
        "dynamics_state_dot_params": dynamics_state_dot_params,
    }
    _, state_matrix = scan(next_state_scan, initial_state, everythings)
    state_matrix = jnp.concatenate(
        [initial_state[jnp.newaxis, :], state_matrix], axis=0
    )
    return state_matrix

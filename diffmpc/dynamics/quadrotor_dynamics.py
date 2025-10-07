"""Quadrotor dynamics class. With rotation defined using a right-handed quaternion (not JPL convention)"""
from typing import Any, Dict

import jax.numpy as jnp

from diffmpc.dynamics.base_dynamics import Dynamics

quadrotor_parameters: Dict[str, Any] = {
    "num_states": 13,
    "num_controls": 4,
    "names_states": [
        "pos_x",
        "pos_y",
        "pos_z",
        "vel_x",
        "vel_y",
        "vel_z",
        "q_0",
        "q_1",
        "q_2",
        "q_3",
        "omega_x",
        "omega_y",
        "omega_z",
    ],
    "names_controls": ["thrust", "torque_x", "torque_y", "torque_z"],
}
quadrotor_state_dot_parameters = {
    "mass": 0.1,
    "inertia": jnp.array([0.1, 0.01, 0.01, 0.01, 0.1, 0.01, 0.01, 0.01, 0.1]),
}


# Skew symmetric map (lie algebra of SO(3), solves a x b = S(a)b)
def S(u):
    return jnp.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])


# Left quaternion product (used in the quaternion dynamics)
def q_left(q):
    qw = q[0]
    qv = q[1:]
    qL_B = jnp.concatenate((-qv.reshape(1, -1), S(qv)), axis=0)
    qL_A = jnp.concatenate((jnp.array([[0]]), qv.reshape(-1, 1)), axis=0)
    qL = jnp.concatenate((qL_A, qL_B), axis=1)
    for ii in range(4):
        qL = qL.at[ii, ii].set(qw)
    return qL


# Left quaternion product (used in the quaternion dynamics)
def q_conj(q):
    return jnp.concatenate((q[:1], -q[1:]))


# Convert the quaternion into a rotation
def get_rotation(q):
    qw, qx, qy, qz = q
    qw2 = qw * qw
    qx2 = qx * qx
    qy2 = qy * qy
    qz2 = qz * qz
    return jnp.array(
        [
            [qw2 + qx2 - qy2 - qz2, 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), qw2 - qx2 + qy2 - qz2, 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), qw2 - qx2 - qy2 + qz2],
        ]
    )


class QuadrotorDynamics(Dynamics):
    """Spacecraft dynamics class."""

    def __init__(self, parameters: Dict[str, Any] = quadrotor_parameters):
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
        state_dot_params: Dict[str, Any] = quadrotor_state_dot_parameters,
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
        inertia = state_dot_params["inertia"].reshape((3, 3))
        inertia_inverse = jnp.linalg.inv(
            inertia
        )  # We might want to do this computation elsewhere.
        m = state_dot_params["mass"]
        g = 9.81

        # Extract states (no checks are done here)
        v = jnp.array(state[3:6])
        q = jnp.array(state[6:10])
        w = jnp.array(state[10:13])
        f = control[0]
        tau = control[1:4]

        # Normalize the quaternion (this should be done post integration, this is a hack)
        q = q / jnp.linalg.norm(q)

        # Express dynamics on R9 x H, with velocity in global cooridnates (convention of scaramuzza)
        e3 = jnp.array([0, 0, 1])
        R = get_rotation(q)
        p_dot = v
        v_dot = -g * e3 + (f / m) * R @ e3
        q_dot = 0.5 * q_left(q) @ jnp.concatenate((jnp.array([0.0]), w))
        w_dot = inertia_inverse @ (S(inertia @ w) @ w + tau)

        state_dot = jnp.concatenate((p_dot, v_dot, q_dot, w_dot))
        return state_dot

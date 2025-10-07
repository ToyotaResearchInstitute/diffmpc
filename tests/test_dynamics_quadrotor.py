"""Checks the dynamics functions."""
import copy

import jax.numpy as jnp
import numpy as np
from jax import config

from diffmpc.dynamics.quadrotor_dynamics import (
    QuadrotorDynamics,
    S,
    get_rotation,
    q_conj,
    q_left,
    quadrotor_parameters,
    quadrotor_state_dot_parameters,
)

np.random.seed(0)
config.update("jax_enable_x64", True)

# Constants for testing
rtol = 1e-5
atol = 1e-8


# Test that the skwe operator does what we expect
def test_skew():
    a = jnp.array(
        np.random.randn(
            3,
        )
    )
    b = jnp.array(
        np.random.randn(
            3,
        )
    )
    assert jnp.allclose(
        jnp.cross(a, b), S(a) @ b, rtol=rtol, atol=atol
    ), "Skew result mismatch"


# Test that the rotation is orthogonal
def test_rotation_orthogonality():
    for ii in range(10):
        q_np = np.random.randn(
            4,
        )
        q = jnp.array(q_np / np.linalg.norm(q_np))
        R = get_rotation(q)
        assert jnp.allclose(
            jnp.eye(3), R.T @ R, rtol=rtol, atol=atol
        ), "Orthogonality result error"


# Test that the detemrinant is positive
def test_rotation_determinant():
    for ii in range(10):
        q_np = np.random.randn(
            4,
        )
        q = jnp.array(q_np / np.linalg.norm(q_np))
        R = get_rotation(q)
        assert jnp.allclose(
            1.0, np.linalg.det(np.array(R)), rtol=rtol, atol=atol
        ), "Determinant result error"


def test_q_conj():
    q_np = np.random.randn(
        4,
    )
    q = jnp.array(q_np / np.linalg.norm(q_np))
    qi = q_conj(q)
    assert jnp.allclose(q[0], +qi[0], rtol=rtol, atol=atol)
    assert jnp.allclose(q[1], -qi[1], rtol=rtol, atol=atol)
    assert jnp.allclose(q[2], -qi[2], rtol=rtol, atol=atol)
    assert jnp.allclose(q[3], -qi[3], rtol=rtol, atol=atol)


# # Test that we have the qL map implemented correctly
def test_rotation_v_qL():
    for ii in range(10):
        u = np.random.randn(
            3,
        )
        q_np = np.random.randn(
            4,
        )
        q = jnp.array(q_np / np.linalg.norm(q_np))
        R = get_rotation(q)

        a = R @ u
        b = q_left(q_left(q) @ jnp.concatenate((jnp.array([0]), u))) @ q_conj(q)
        assert jnp.allclose(a, b[1:], rtol=rtol, atol=atol), "Rotation map result error"


def test_quadrotor_state_dot_dimensions():
    """ "
    Tests state and control dimensions of Keisuke dynamics with path.
    """
    model = QuadrotorDynamics(quadrotor_parameters)
    x = jnp.ones(model.num_states)
    u = jnp.ones(model.num_controls)

    dynamics_state_dot_params = copy.deepcopy(quadrotor_state_dot_parameters)
    x_dot = model.state_dot(x, u, dynamics_state_dot_params)
    assert len(x_dot) == model.num_states

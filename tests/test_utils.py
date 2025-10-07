"""Tests for the solver classes."""

import jax.numpy as jnp
import numpy as np
import pytest
from jax import jacfwd

from diffmpc.utils.jax_utils import (
    jax_has_gpu,
    project_matrix_onto_positive_semidefinite_cone,
    value_and_jacfwd,
    value_and_jacrev,
)


def check_two_arrays_are_close(f1, f2):
    """Checks if f1 and f2 and their Jacobians are close."""
    assert len(f2) == len(f1)
    assert f2.shape == f1.shape
    assert jnp.all(jnp.linalg.norm(f1 - f2) < 1e-9)
    return True


def test_value_and_jacfwd():
    """Test value_and_jacfwd."""
    x = jnp.array([1.0, 10.5, -5.2])

    for i in range(2):
        if i == 0:
            # If f returns a scalar, the output should be a jax array
            def f(x):
                return (5.2 * x[1] ** 2 + x[2]) * jnp.ones(1)

        else:
            # f returns a vector
            def f(x):
                return jnp.array([x[0] ** 2, -x[1] ** 3])

        def f_dx(x):
            return jacfwd(f)(x)

        f1 = f(x)
        f1dx = f_dx(x)
        f2, f2dx = value_and_jacfwd(f, x)

        check_two_arrays_are_close(f1, f2)
        check_two_arrays_are_close(f1dx, f2dx)


def test_value_and_jacrev():
    """Test value_and_jacrev."""
    x = jnp.array([1.0, 10.5, -5.2])

    for i in range(2):
        if i == 0:
            # If f returns a scalar, the output should be a jax array
            def f(x):
                return (5.2 * x[1] ** 2 + x[2]) * jnp.ones(1)

        else:
            # f returns a vector
            def f(x):
                return jnp.array([x[0] ** 2, -x[1] ** 3])

        def f_dx(x):
            return jacfwd(f)(x)

        f1 = f(x)
        f1dx = f_dx(x)
        f2, f2dx = value_and_jacrev(f, x)

        check_two_arrays_are_close(f1, f2)
        check_two_arrays_are_close(f1dx, f2dx)


def test_jax_has_gpu():
    """Tests jax_has_gpu."""
    jax_has_gpu()


@pytest.mark.parametrize("n", [2, 3, 5])
def test_project_matrix_onto_positive_semidefinite_cone(n):
    """Tests the PSD projection."""
    rng = np.random.default_rng(0)
    A = rng.normal(size=(n, n))
    A = (A + A.T) / 2  # symmetrize

    projected = project_matrix_onto_positive_semidefinite_cone(jnp.array(A))
    eigvals = np.linalg.eigvalsh(np.array(projected))

    # All eigenvalues should be >= 0
    assert np.all(eigvals >= -1e-8)
    assert np.allclose(projected, projected.T, atol=1e-8)

"""Helper functions for jax such as autodifferentiation."""

from functools import partial
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from jax import jvp, vmap
from jax._src.api_util import check_callable


def value_and_jacfwd(
    f: Callable[[jnp.ndarray], jnp.ndarray], x: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute the value and the Jacobian of a function.

    Source: https://github.com/google/jax/pull/762

    Args:
        f: function taking an array and returning an array
            Callable[[jnp.ndarray], jnp.ndarray]
        x: input of the function f
            (num_variables) array

    Returns:
        f(x): value of the function f at x
            (num_outputs) array
        f_dx(x): value of the Jacobian of the function f at x
            (num_outputs, num_variables) array
    """
    check_callable(f)
    pushfwd = partial(jvp, f, (x,))
    basis = jnp.eye(x.size, dtype=x.dtype)
    y, jac = vmap(pushfwd, out_axes=(None, 1))((basis,))
    return y, jac


def value_and_jacrev(
    f: Callable[[jnp.ndarray], jnp.ndarray], x: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute the value and the Jacobian of a function.

    Source: https://github.com/google/jax/pull/762

    Args:
        f: function taking an array and returning an array
            Callable[[jnp.ndarray], jnp.ndarray]
        x: input of the function f
            (num_variables) array

    Returns:
        f(x): value of the function f at x
            (num_outputs) array
        f_dx(x): value of the Jacobian of the function f at x
            (num_outputs, num_variables) array
    """
    check_callable(f)
    y, pullback = jax.vjp(f, x)
    basis = jnp.eye(y.size, dtype=y.dtype)
    jac = jax.vmap(pullback)(basis)[0]
    return y, jac


def jax_has_gpu() -> bool:
    """Checks if Jax can use a GPU."""
    try:
        _ = jax.device_put(jnp.ones(1), device=jax.devices("gpu")[0])
        return True
    except RuntimeError:
        return False


def project_matrix_onto_positive_semidefinite_cone(
    matrix: jnp.ndarray, minimum_eigenvalue: float = 0.0
) -> jnp.ndarray:
    """Projects the matrix onto the positive semi-definite cone.

    See also https://github.com/google/trajax/blob/main/trajax/optimizers.py.

    Args:
        matrix: symmetric matrix,
            (n, n) array
        minimum_eigenvalue: minimum eigenvalue of the projected matrix,
            (float)

    Returns:
        projected_matrix: matrix with eigenvalues larger than  minimum_eigenvalue,
            (n, n) array
    """
    S, V = jnp.linalg.eigh(matrix)
    S = jnp.maximum(S, minimum_eigenvalue)
    proj_matrix = jnp.matmul(V, jnp.matmul(jnp.diag(S), V.T))
    return 0.5 * (proj_matrix + proj_matrix.T)

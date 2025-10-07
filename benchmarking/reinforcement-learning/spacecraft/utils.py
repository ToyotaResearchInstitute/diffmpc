"""Utilities"""
import jax.numpy as jnp
import numpy as np

# Problem dimensions
N_STATE = 3  # state dimension
N_CTRL = 3  # control dimension


# Solver settings
# diffmpc
DIFFMPC_SCP_ITER = 1  # diffmpc SQP iterations
# trajax
TRAJAX_LQR_ITER = 1  # trajax iLQR iterations
TRAJAX_ALPHA_MIN = 0.99  # trajax iLQR alpha min


def generate_problem_data(num_batch, seed):
    """Generate spacecraft problem data for benchmarking"""
    np.random.seed(seed)

    min_inertia = 1.0
    max_inertia = 10.0
    inertia_vector = min_inertia + np.random.rand(3) * (max_inertia - min_inertia)

    min_state = -0.1
    max_state = 0.1
    initial_states = min_state + np.random.rand(num_batch, 3) * (max_state - min_state)

    return jnp.array(inertia_vector), jnp.array(initial_states)


def reward(state, control):
    r = -(1e1 * jnp.sum(state**2) + jnp.sum(control**2))
    return r

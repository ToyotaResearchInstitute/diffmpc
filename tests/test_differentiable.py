"""Tests for the SQP solver."""

import copy

import numpy as np
from jax import config

config.update("jax_enable_x64", True)  # use double precision

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from diffmpc.dynamics.spacecraft_dynamics import SpacecraftDynamics  # noqa: E402
from diffmpc.problems.optimal_control_problem import OptimalControlProblem  # noqa: E402
from diffmpc.solvers.sqp import SQPSolver  # noqa: E402
from diffmpc.utils.load_params import (  # noqa: E402
    load_problem_params,
    load_solver_params,
)


def test_sqp_solve_differentiable():
    """Test spacecraft solve for various horizons and batchsizes"""
    np.random.seed(0)
    horizon = 10

    # Problem parameters
    problem_params = load_problem_params("spacecraft.yaml")
    diffmpc_horizon = horizon - 1  # diffmpc horizon is N+1
    min_state = -0.1
    max_state = 0.1
    problem_params["initial_state"] = min_state + np.random.rand(3) * (
        max_state - min_state
    )
    problem_params["horizon"] = diffmpc_horizon
    problem_params["reference_state_trajectory"] = jnp.zeros((diffmpc_horizon + 1, 3))
    problem_params["reference_control_trajectory"] = jnp.zeros((diffmpc_horizon + 1, 3))
    problem_params["weights_penalization_final_state"] = jnp.zeros(3)

    # Solver parameters
    solver_params = load_solver_params("sqp.yaml")
    solver_params["tol_convergence"] = 1.0e-12
    solver_params["num_scp_iteration_max"] = 30
    solver_params["pcg"]["tol_epsilon"] = 1.0e-24
    solver_params["linesearch"] = False
    solver_params["warm_start_backward"] = True

    dynamics = SpacecraftDynamics()
    problem = OptimalControlProblem(dynamics=dynamics, params=problem_params)
    solver = SQPSolver(program=problem, params=solver_params)

    weights = {
        k: problem_params[k]
        for k in [
            "weights_penalization_reference_state_trajectory",
            "weights_penalization_control_squared",
        ]
    }

    def objective(weights):
        solution = solver.solve(
            solver.initial_guess(), problem_params=problem_params, weights=weights
        )
        return jnp.linalg.norm(solution.states) + jnp.linalg.norm(solution.controls)

    def auto_grad(weights):
        grad = jax.grad(objective)(weights)
        return grad

    def finite_diff_grad(weights, eps=1e-12):
        # Compute finite difference gradients
        grad_fd = {}
        for key, value in weights.items():
            grad_fd[key] = jnp.zeros_like(value)
            for i in range(value.size):
                weights_plus = copy.deepcopy(weights)
                weights_plus[key] = weights_plus[key].at[i].add(eps)
                val_plus = objective(weights_plus)
                weights_minus = copy.deepcopy(weights)
                weights_minus[key] = weights_minus[key].at[i].add(-eps)
                val_minus = objective(weights_minus)
                grad_fd[key] = (
                    grad_fd[key].at[i].set((val_plus - val_minus) / (2 * eps))
                )
        return grad_fd

    grad = auto_grad(weights)
    grad_fd = finite_diff_grad(weights)

    for key in weights.keys():
        print("key =", key, grad_fd[key] - grad[key])
        assert jnp.allclose(grad_fd[key] - grad[key], 0.0, atol=1e-3, rtol=1e-3)

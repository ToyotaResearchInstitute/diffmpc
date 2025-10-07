"""Tests for the SQP solver."""
import numpy as np
import pytest
from jax import config

config.update("jax_enable_x64", True)  # use double precision

import jax.numpy as jnp  # noqa: E402

from diffmpc.dynamics.spacecraft_dynamics import SpacecraftDynamics  # noqa: E402
from diffmpc.problems.optimal_control_problem import OptimalControlProblem  # noqa: E402
from diffmpc.solvers.sqp import SolverReturnStatus, SQPSolver  # noqa: E402
from diffmpc.utils.load_params import (  # noqa: E402
    load_problem_params,
    load_solver_params,
)


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


def sqp_solve_spacecraft(num_batch, horizon, use_linesearch, warm_start_backward):
    # Problem parameters
    problem_params = load_problem_params("spacecraft.yaml")
    inertia_vector, initial_states = generate_problem_data(num_batch, seed=0)
    print(f"device: {inertia_vector.device}")
    diffmpc_horizon = horizon - 1  # diffmpc horizon is N+1
    problem_params["horizon"] = diffmpc_horizon
    problem_params["inertia_vector"] = inertia_vector
    problem_params["reference_state_trajectory"] = jnp.zeros((diffmpc_horizon + 1, 3))
    problem_params["reference_control_trajectory"] = jnp.zeros((diffmpc_horizon + 1, 3))
    problem_params["weights_penalization_final_state"] = jnp.zeros(3)

    # Solver parameters
    solver_params = load_solver_params("sqp.yaml")
    solver_params["tol_convergence"] = 1.0e-6
    solver_params["num_scp_iteration_max"] = 20
    solver_params["pcg"]["tol_epsilon"] = 1.0e-12
    solver_params["linesearch"] = use_linesearch
    solver_params["warm_start_backward"] = warm_start_backward
    solver_params["linesearch_alphas"] = [1.0]

    dynamics = SpacecraftDynamics()
    problem = OptimalControlProblem(dynamics=dynamics, params=problem_params)
    solver = SQPSolver(program=problem, params=solver_params)

    sqp_guess = solver.initial_guess()

    def solver_initial_guess(initial_state):
        params = {**problem_params, "initial_state": initial_state}
        return solver.initial_guess(params)

    weights = {
        k: problem_params[k]
        for k in [
            "weights_penalization_reference_state_trajectory",
            "weights_penalization_control_squared",
        ]
    }
    params = {**problem_params, "inertia_vector": inertia_vector}

    solution = solver.solve(solver_initial_guess(initial_states[0]), params, weights)
    return solver, sqp_guess, solution, dynamics


@pytest.mark.parametrize("num_batch", [1, 2])
@pytest.mark.parametrize("horizon", [10, 20])
@pytest.mark.parametrize("use_linesearch", [True, False])
@pytest.mark.parametrize("warm_start_backward", [True, False])
def test_sqp_solve(num_batch, horizon, use_linesearch, warm_start_backward):
    """Test spacecraft solve for various horizons and batchsizes"""

    solver, sqp_guess, solution, dynamics = sqp_solve_spacecraft(
        num_batch, horizon, use_linesearch, warm_start_backward
    )

    assert solver.name == "SQPSolver"
    assert solver.pcg is not None
    assert sqp_guess.states.shape == (horizon, dynamics.num_states)
    assert sqp_guess.controls.shape == (horizon, dynamics.num_controls)
    assert sqp_guess.dual.shape == (horizon, dynamics.num_states)
    assert solution.status == SolverReturnStatus.SUCCESS
    assert jnp.isfinite(solution.convergence_error)


def test_sqp_solve_warmstart():
    """Test that the warmstarting works as intended"""
    num_batch, horizon, use_linesearch = 1, 20, True
    iter = {}
    for warm_start_backward in [True, False]:
        _, _, solution, _ = sqp_solve_spacecraft(
            num_batch, horizon, use_linesearch, warm_start_backward
        )
        assert solution.status == SolverReturnStatus.SUCCESS
        assert jnp.isfinite(solution.convergence_error)
        if warm_start_backward:
            iter["warm"] = solution.num_iter
        else:
            iter["cold"] = solution.num_iter
    print(iter)
    assert iter["warm"] <= iter["cold"]

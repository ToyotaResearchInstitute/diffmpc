"""Tests of the optimal control features."""

import pytest
from jax import config

from diffmpc.dynamics.spacecraft_dynamics import SpacecraftDynamics
from diffmpc.problems.optimal_control_problem import OptimalControlProblem
from diffmpc.solvers.sqp import SolverReturnStatus, SQPSolver
from diffmpc.utils.load_params import load_problem_params, load_solver_params

config.update("jax_enable_x64", True)


@pytest.fixture
def solver_parameters():
    """Returns the solver parameters."""
    solver_parameters = load_solver_params("sqp.yaml")
    solver_parameters["pcg"]["tol_epsilon"] = 1e-15
    return solver_parameters


@pytest.fixture
def spacecraft_problem_params():
    """Returns optimal control problem parameters for the spacecraft problem."""
    problem_params = load_problem_params("spacecraft.yaml")
    return problem_params


def test_spacecraft_with_sqp_solver(spacecraft_problem_params, solver_parameters):
    """
    Test solving optimal control problem for
    the spacecraft system using the SQP solver.
    """
    dynamics = SpacecraftDynamics()
    problem = OptimalControlProblem(dynamics=dynamics, params=spacecraft_problem_params)
    solver = SQPSolver(program=problem, params=solver_parameters)
    sol = solver.solve(solver.initial_guess(), problem.params)
    convergence_error, status = sol.convergence_error, sol.status
    assert convergence_error < 2e-5
    assert SolverReturnStatus(status) is SolverReturnStatus.SUCCESS

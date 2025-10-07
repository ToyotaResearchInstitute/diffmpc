#!/usr/bin/env python3
"""Expert data collection"""
import pickle

import numpy as np
from jax import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from diffmpc.dynamics.cartpole_dynamics import CartpoleDynamics  # noqa: E402
from diffmpc.problems.optimal_control_problem import OptimalControlProblem  # noqa: E402
from diffmpc.solvers.sqp import SQPSolver  # noqa: E402
from diffmpc.utils.load_params import (  # noqa: E402
    load_problem_params,
    load_solver_params,
)


def collect_expert_data(num_trajs):
    dynamics = CartpoleDynamics()
    problem_params = load_problem_params("cartpole.yaml")

    problem = OptimalControlProblem(dynamics=dynamics, params=problem_params)
    solver_params = load_solver_params("sqp.yaml")
    solver_params["num_scp_iteration_max"] = 501
    solver_params["linesearch"] = True
    solver_params["verbose"] = solver_params["pcg"]["verbose"] = False
    solver_params["pcg"]["tol_epsilon"] = 1e-24
    solver = SQPSolver(program=problem, params=solver_params)

    all_data = []
    while len(all_data) < num_trajs:
        x_init = np.random.uniform(low=-0.5, high=0.5)
        xdot_init = np.random.uniform(low=-0.5, high=0.5)
        theta_init = np.random.uniform(low=-3.14, high=3.14)
        thetadot_init = np.random.uniform(low=-1.0, high=1.0)
        x_current = jnp.array([x_init, xdot_init, theta_init, thetadot_init])

        params_step = {**problem_params, "initial_state": x_current}
        initial_guess = solver.initial_guess(params_step)
        solution = solver.solve(initial_guess, params_step, {})
        print(
            f"convergence error = {solution.convergence_error}, num iter ="
            f" {solution.num_iter}"
        )
        all_data.append({"x0": x_current, "ctrl_traj": solution.controls})

        if len(all_data) % 10 == 0:
            print(len(all_data))

    return all_data


if __name__ == "__main__":
    np.random.seed(123)

    trajs = collect_expert_data(num_trajs=32)

    with open("demonstrations.pkl", "wb") as file:
        pickle.dump(trajs, file)

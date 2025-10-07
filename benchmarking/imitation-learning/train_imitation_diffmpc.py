#!/usr/bin/env python3
"""Imitation learning"""

import pickle
import time

import numpy as np
from jax import config

config.update("jax_enable_x64", True)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax import jit  # noqa: E402
from utils import (  # noqa: E402
    initial_weights_penalization_reference_state_trajectory,
    load_dataset,
    rollout,
)

from diffmpc.dynamics.cartpole_dynamics import CartpoleDynamics  # noqa: E402
from diffmpc.problems.optimal_control_problem import OptimalControlProblem  # noqa: E402
from diffmpc.solvers.sqp import SQPSolver  # noqa: E402
from diffmpc.utils.load_params import (  # noqa: E402
    load_problem_params,
    load_solver_params,
)

# -------------------------
# Hyperparameters
# -------------------------
# BATCH_SIZE = 32 # train on all of the imitation data
NUM_STEPS = 200
LEARNING_RATE = 1e-2


def main():
    x0_data, ctrl_data = load_dataset("demonstrations.pkl")
    problem_params = load_problem_params("cartpole.yaml")
    true_weights = problem_params["weights_penalization_reference_state_trajectory"]

    print(f"using device: {x0_data.device}")

    dynamics = CartpoleDynamics()
    problem = OptimalControlProblem(dynamics=dynamics, params=problem_params)

    solver_params = load_solver_params("sqp.yaml")
    solver_params["tol_convergence"] = 1e-1
    solver_params["num_scp_iteration_max"] = 5
    solver_params["pcg"]["tol_epsilon"] = 1e-14
    solver_params["linesearch"] = True
    solver_params["verbose"] = solver_params["pcg"]["verbose"] = False
    solver = SQPSolver(program=problem, params=solver_params)

    def initial_guess_from_data(x0, us):
        # use demonstration control trajectory as initial guess
        params = {**problem_params, "initial_state": x0}
        guess = solver.initial_guess(params)
        guess = guess._replace(
            states=rollout(x0, us, dynamics, problem_params), controls=us
        )
        return guess

    def loss_per_sample(w_xref, x0, us):
        params = {**problem_params, "initial_state": x0}
        initial_guess = initial_guess_from_data(x0, us)
        weights = {"weights_penalization_reference_state_trajectory": w_xref}
        sol = solver.solve(initial_guess, params, weights)
        controls = sol.controls
        loss = 0.5 * jnp.mean((controls - us) ** 2)
        return loss

    def batch_mean_loss(w_xref, x0s, us):
        losses = jax.vmap(loss_per_sample, in_axes=(None, 0, 0))(w_xref, x0s, us)
        return jnp.mean(losses)

    @jit
    def grad_step_and_imitation_learning_loss(w_xref, x0s, us):
        loss, grad = jax.value_and_grad(batch_mean_loss)(w_xref, x0s, us)
        loss, grad = jnp.nan_to_num(loss), jnp.nan_to_num(grad)
        w_xref = jnp.maximum(1e-10, w_xref - LEARNING_RATE * grad)
        return w_xref, loss

    @jit
    def model_loss(w_xref):
        return jnp.linalg.norm(w_xref - true_weights)

    initial_weights = initial_weights_penalization_reference_state_trajectory()
    all_imitation_losses = []
    all_model_losses = []
    all_times = []

    # jit
    grad_step_and_imitation_learning_loss(initial_weights[0], x0_data, ctrl_data)
    model_loss(initial_weights[0])

    for i, w_xref in enumerate(initial_weights):
        imitation_losses = [
            grad_step_and_imitation_learning_loss(w_xref, x0_data, ctrl_data)[1]
        ]
        model_losses = [model_loss(w_xref)]
        times = [0.0]

        s = time.monotonic()
        for step in range(NUM_STEPS):
            m_loss = model_loss(w_xref)
            model_losses.append(m_loss)
            w_xref, loss = grad_step_and_imitation_learning_loss(
                w_xref, x0_data, ctrl_data
            )

            imitation_losses.append(loss)
            ts = time.monotonic() - s
            times.append(ts)
            print(
                f"Run {1+i}/{len(initial_weights)}, Step {step}/{NUM_STEPS}: IL loss:"
                f" {loss}, model loss: {m_loss}, time: {ts}"
            )

        print()
        all_imitation_losses.append(np.array(imitation_losses))
        all_model_losses.append(np.array(model_losses))
        all_times.append(np.array(times))

    with open("train_il_diffmpc.pkl", "wb") as file:
        data = {
            "model_losses": np.array(all_model_losses),
            "imitation_losses": np.array(all_imitation_losses),
            "times": np.array(all_times),
        }
        pickle.dump(data, file)


if __name__ == "__main__":
    main()

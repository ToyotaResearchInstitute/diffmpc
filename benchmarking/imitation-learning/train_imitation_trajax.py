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
from trajax import optimizers  # noqa: E402
from utils import (  # noqa: E402
    initial_weights_penalization_reference_state_trajectory,
    load_dataset,
)

from diffmpc.dynamics.cartpole_dynamics import CartpoleDynamics  # noqa: E402
from diffmpc.dynamics.integrators import DiscretizationScheme  # noqa: E402
from diffmpc.utils.load_params import load_problem_params  # noqa: E402

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

    print(f"using device: {x0_data.device()}")

    dynamics = CartpoleDynamics()
    dt = float(problem_params["discretization_resolution"])
    scheme_int = int(problem_params.get("discretization_scheme", 0))
    assert DiscretizationScheme(scheme_int) == DiscretizationScheme.EULER

    def dynamics_step(x, u, t):
        return x + dt * dynamics.state_dot(x, u, problem_params)

    def cost(x, u, t, w_xref):
        state_cost = jnp.sum((x**2) * w_xref)
        control_cost = jnp.sum(
            (u**2) * problem_params["weights_penalization_control_squared"]
        )
        return state_cost + control_cost

    def loss_per_sample(w_xref, x0, us):
        X, controls, obj, gradient, adjoints, lqr, iteration = optimizers.ilqr(
            lambda x, u, t: cost(x, u, t, w_xref),
            dynamics_step,
            x0,
            us,  # use demonstration expert control trajectory as initial guess
            maxiter=5,
            alpha_min=0.06,
            vjp_method="tvlqr",
        )
        return 0.5 * jnp.mean((controls - us) ** 2)

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

    with open("train_il_trajax_on_trajaxdata.pkl", "wb") as file:
        data = {
            "model_losses": np.array(all_model_losses),
            "imitation_losses": np.array(all_imitation_losses),
            "times": np.array(all_times),
        }
        pickle.dump(data, file)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Plots demonstration data.
"""

import matplotlib.pyplot as plt
from jax import config

config.update("jax_enable_x64", True)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from utils import load_dataset, rollout  # noqa: E402

from diffmpc.dynamics.cartpole_dynamics import CartpoleDynamics  # noqa: E402
from diffmpc.problems.optimal_control_problem import OptimalControlProblem  # noqa: E402
from diffmpc.utils.load_params import load_problem_params  # noqa: E402


def main():
    print("> Loading demonstration data.")
    x0s, us = load_dataset("demonstrations.pkl")
    batch_size, horizon = us.shape[0], us.shape[1]
    ts = jnp.repeat(jnp.arange(horizon)[None], axis=0, repeats=batch_size)

    print("> Initializing dynamics and problem.")
    dynamics = CartpoleDynamics()
    problem_params = load_problem_params("cartpole.yaml")
    problem = OptimalControlProblem(dynamics=dynamics, params=problem_params)

    print("> Dynamics rollouts.")
    xs = jax.vmap(rollout, in_axes=(0, 0, None, None))(
        x0s, us, dynamics, problem_params
    )

    print("----------------------------------")
    print("Demonstration data sizes:")
    print("ts =", ts.shape)
    print("x0s =", x0s.shape)
    print("us =", us.shape)
    print("xs =", xs.shape)

    print("----------------------------------")
    print("Demonstration data cost:")
    print(
        "diffmpc median cost =",
        jnp.median(
            jax.vmap(problem.cost, in_axes=(0, 0, None))(xs, us, problem_params)
        ),
    )

    print("----------------------------------")
    print("Plotting demonstration data.")
    fig, axs = plt.subplots(1 + xs.shape[-1] + 1)
    fig.suptitle("DiffMPC")
    axs[0].scatter(x0s[:, 0], x0s[:, 1])
    for i in range(xs.shape[-1]):
        axs[1 + i].plot(ts.T, xs[:, :, i].T)
        axs[1 + i].set_ylabel(dynamics.names_states[i])
    axs[-1].plot(ts.T, us[:, :, 0].T)
    axs[-1].set_ylabel(dynamics.names_controls[0])
    for ax in axs:
        ax.grid()
    plt.show()
    print("----------------------------------")


if __name__ == "__main__":
    main()

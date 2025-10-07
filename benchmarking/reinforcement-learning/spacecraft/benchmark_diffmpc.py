"""Benchmark"""
import argparse
import os
import time

import numpy as np

# DiffMPC imports
from jax import config

config.update("jax_enable_x64", True)  # use double precision

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax import jit, vmap  # noqa: E402
from utils import generate_problem_data, reward  # noqa: E402

from diffmpc.dynamics.spacecraft_dynamics import SpacecraftDynamics  # noqa: E402
from diffmpc.problems.optimal_control_problem import OptimalControlProblem  # noqa: E402
from diffmpc.solvers.sqp import SQPSolver  # noqa: E402
from diffmpc.utils.load_params import (  # noqa: E402
    load_problem_params,
    load_solver_params,
)


def benchmark_diffmpc_rollout(num_batch, num_sim_steps, horizon):
    print("=== DiffMPC Rollout ===")

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
    solver_params["num_scp_iteration_max"] = 1
    solver_params["pcg"]["tol_epsilon"] = 1.0e-12
    solver_params["linesearch"] = True
    solver_params["linesearch_alphas"] = [1.0]

    dynamics = SpacecraftDynamics()
    problem = OptimalControlProblem(dynamics=dynamics, params=problem_params)
    solver = SQPSolver(program=problem, params=solver_params)

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
    all_times_fwd = []
    all_times_bwd = []
    for seed in range(10):
        inertia_vector, initial_states = generate_problem_data(num_batch, seed)

        def rollout(state, weights):
            initial_carry = (state, solver_initial_guess(state), 0.0)

            def rollout_step(carry, _):
                state, solution, running_reward = carry

                # Solve MPC problem
                params = {
                    **problem_params,
                    "inertia_vector": inertia_vector,
                    "initial_state": state,
                }
                solution = solver.solve(solution, params, weights)
                control = solution.controls[0]

                # Simulate forward
                dt = params["discretization_resolution"]
                next_state = state + dt * dynamics.state_dot(state, control, params)
                running_reward = running_reward + reward(next_state, control)

                return (next_state, solution, running_reward), None

            final_carry, _ = jax.lax.scan(
                rollout_step, initial_carry, None, length=num_sim_steps
            )
            _, _, final_reward = final_carry
            return final_reward

        rollout_batch = jit(vmap(rollout, in_axes=(0, None)))

        def rollout_for_grad(weights):
            return jnp.sum(rollout_batch(initial_states, weights))

        rollout_grad = jit(jax.grad(rollout_for_grad))

        # jit compile forward
        out = rollout_batch(jnp.zeros_like(initial_states), weights)
        jax.block_until_ready(out)

        # time forward
        start_time = time.monotonic()
        final_costs = rollout_batch(initial_states, weights)
        jax.block_until_ready(final_costs)
        forward_time = time.monotonic() - start_time

        # jit compile backward
        r = rollout_grad(weights)
        jax.block_until_ready(r)

        # time backward
        start_time = time.monotonic()
        gradients = rollout_grad(weights)
        jax.block_until_ready(gradients)
        grad_time = time.monotonic() - start_time

        print(
            f"Forward time: {forward_time*1000:.1f} ms"
            f" ({forward_time*1000/(num_batch*num_sim_steps):.3f} ms/step/problem)"
        )
        print(f"Aggregate cost: {final_costs.sum():.4f}")
        print(
            f"Grad time: {grad_time*1000:.1f} ms"
            f" ({grad_time*1000/(num_batch*num_sim_steps):.3f} ms/step/problem)"
        )

        all_times_fwd.append(forward_time)
        all_times_bwd.append(grad_time)

    return np.array(all_times_fwd), np.array(all_times_bwd)


def main():
    parser = argparse.ArgumentParser(description="MPC Benchmark")
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for benchmark"
    )
    parser.add_argument("--horizon", type=int, default=40, help="MPC horizon")
    parser.add_argument(
        "--num_sim_steps", type=int, default=20, help="Number of simulation steps"
    )

    args = parser.parse_args()

    print(f"\n{'=' * 70}")
    print(f"DiffMPC Rollout: {args.horizon} horizon")
    print(f"Simulation: {args.num_sim_steps} steps, {args.batch_size} batch")

    forward_times, backward_times = benchmark_diffmpc_rollout(
        args.batch_size, args.num_sim_steps, args.horizon
    )

    dirname = f"spacecraft_timing_{args.batch_size}_{args.horizon}"
    device = jnp.zeros(1).device
    os.makedirs(f"timing_results/{dirname}", exist_ok=True)
    np.save(f"timing_results/{dirname}/diffmpc_{device}_fwd", forward_times)
    np.save(f"timing_results/{dirname}/diffmpc_{device}_bwd", backward_times)


if __name__ == "__main__":
    main()

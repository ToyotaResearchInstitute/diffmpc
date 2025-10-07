"""Benchmarking script."""
import argparse
import os
import time

import numpy as np
from jax import config

config.update("jax_enable_x64", True)  # use double precision

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax import jit, vmap  # noqa: E402
from utils import (  # noqa: E402
    DIFFMPC_SCP_ITER,
    MPC_LQR_ITER,
    MPC_TOL,
    N_CTRL,
    N_STATE,
    generate_problem_data,
)

from diffmpc.dynamics.linear_dynamics import LinearDynamics  # noqa: E402
from diffmpc.problems.optimal_control_problem import OptimalControlProblem  # noqa: E402
from diffmpc.solvers.sqp import SQPSolver  # noqa: E402
from diffmpc.utils.load_params import (  # noqa: E402
    load_problem_params,
    load_solver_params,
)


def reward(state, control):
    r = -(jnp.sum(state**2) + jnp.sum(control**2))
    return r


def benchmark_diffmpc_rollout(
    N_BATCH, num_sim_steps, HORIZON, pcg_tol_epsilon, warm_start, num_repeats
):
    print("=== DiffMPC Rollout ===")

    DIFFMPC_HORIZON = HORIZON - 1  # diffmpc horizon is N+1

    # Problem parameters
    problem_params = load_problem_params("linear.yaml")
    Q, R, A_matrix, B_matrix, b_vector, initial_states = generate_problem_data(
        N_BATCH, seed=0
    )
    A_matrix, B_matrix, b_vector, initial_states = (
        A_matrix[0],
        B_matrix[0],
        b_vector[0],
        initial_states[0],
    )
    # Convert discrete-time dynamics to continuous-time for diffmpc
    # mpc.pytorch: x[k+1] = A_discrete*x[k] + B_discrete*u[k] + b_discrete
    # diffmpc with dt=1, Euler: x[k+1] = x[k] + dt*(A_cont*x[k] + B_cont*u[k] + b_cont)
    #                                     = (I + A_cont)*x[k] + B_cont*u[k] + b_cont
    # So: A_cont = A_discrete - I, B_cont = B_discrete, b_cont = b_discrete
    A_continuous = A_matrix - jnp.eye(N_STATE)
    B_continuous = B_matrix
    b_continuous = b_vector
    print(f"device: {A_continuous.device}")
    problem_params["horizon"] = DIFFMPC_HORIZON
    problem_params["discretization_resolution"] = 1.0
    problem_params["initial_state"] = jnp.zeros(N_STATE)
    problem_params["final_state"] = jnp.zeros(N_STATE)
    problem_params["reference_state_trajectory"] = jnp.zeros(
        (DIFFMPC_HORIZON + 1, N_STATE)
    )
    problem_params["reference_control_trajectory"] = jnp.zeros(
        (DIFFMPC_HORIZON + 1, N_CTRL)
    )
    problem_params["weights_penalization_reference_state_trajectory"] = jnp.diag(
        jnp.array(Q)
    )
    problem_params["weights_penalization_control_squared"] = jnp.diag(jnp.array(R))
    problem_params["weights_penalization_final_state"] = jnp.zeros(N_STATE)
    problem_params["A"] = A_continuous
    problem_params["B"] = B_continuous
    problem_params["b"] = b_continuous

    # Solver parameters
    solver_params = load_solver_params("sqp.yaml")
    solver_params["num_scp_iteration_max"] = DIFFMPC_SCP_ITER
    solver_params["pcg"]["tol_epsilon"] = pcg_tol_epsilon
    solver_params["warm_start_backward"] = warm_start
    solver_params["linesearch"] = True
    solver_params["linesearch_alphas"] = [1.0]
    solver_params["verbose"] = solver_params["pcg"]["verbose"] = False

    # Create dynamics
    dynamics_params = {
        "verbose": False,
        "num_states": N_STATE,
        "num_controls": N_CTRL,
        "names_states": [f"x{i}" for i in range(N_STATE)],
        "names_controls": [f"u{i}" for i in range(N_CTRL)],
    }
    dynamics = LinearDynamics(dynamics_params)
    problem = OptimalControlProblem(dynamics=dynamics, params=problem_params)
    solver = SQPSolver(program=problem, params=solver_params)

    def solver_initial_guess(x0):
        params = {**problem_params, "initial_state": x0}
        return solver.initial_guess(params)

    weights = {
        k: problem_params[k]
        for k in [
            "weights_penalization_reference_state_trajectory",
        ]
    }

    all_times_fwd = []
    all_times_bwd = []
    for seed in range(num_repeats):
        Q, R, A_matrix, B_matrix, b_vector, initial_states = generate_problem_data(
            N_BATCH, seed
        )
        A_matrix = jnp.array(A_matrix)
        B_matrix = jnp.array(B_matrix)
        b_vector = jnp.array(b_vector)
        initial_states = jnp.array(initial_states)
        Q = jnp.array(Q)
        R = jnp.array(R)

        A_continuous = A_matrix - jnp.eye(N_STATE)
        B_continuous = B_matrix
        b_continuous = b_vector
        problem_params["weights_penalization_reference_state_trajectory"] = jnp.diag(
            jnp.array(Q)
        )
        problem_params["weights_penalization_control_squared"] = jnp.diag(jnp.array(R))
        problem_params["weights_penalization_final_state"] = jnp.zeros(N_STATE)
        problem_params["A"] = A_continuous
        problem_params["B"] = B_continuous
        problem_params["b"] = b_continuous

        def rollout(state, weights):
            initial_carry = (state, solver_initial_guess(state), 0.0)

            def rollout_step(carry, _):
                current_state, current_solution, running_cost = carry

                # Solve MPC problem
                params = {**problem_params, "initial_state": current_state}
                solution = solver.solve(current_solution, params, weights)
                u_applied = solution.controls[0]

                # Simulate forward
                new_state = (
                    A_matrix @ current_state.T + B_matrix @ u_applied.T + b_vector
                ).T

                # Update running cost
                new_running_cost = running_cost - reward(new_state, u_applied)

                if warm_start:
                    solution = solution
                else:
                    solution = current_solution
                return (
                    new_state,
                    jax.lax.stop_gradient(solution),
                    new_running_cost,
                ), None

            final_carry, _ = jax.lax.scan(
                rollout_step, initial_carry, None, length=num_sim_steps
            )
            _, _, final_cost = final_carry
            return final_cost

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
        total_time = time.monotonic() - start_time

        # jit compile backward
        r = rollout_grad(weights)
        jax.block_until_ready(r)

        # time backward
        start_time = time.monotonic()
        gradients = rollout_grad(weights)
        jax.block_until_ready(gradients)
        grad_time = time.monotonic() - start_time

        print(
            f"Time: {total_time*1000:.1f} ms"
            f" ({total_time*1000/(N_BATCH*num_sim_steps):.3f} ms/step/problem)"
        )
        print(f"Aggregate cost: {final_costs.sum():.4f}")
        print(
            f"grad time: {grad_time*1000:.1f} ms"
            f" ({grad_time*1000/(N_BATCH*num_sim_steps):.3f} ms/step/problem)"
        )

        all_times_fwd.append(total_time)
        all_times_bwd.append(grad_time)

    return np.array(all_times_fwd), np.array(all_times_bwd)


def main():
    parser = argparse.ArgumentParser(description="MPC Benchmark")
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for benchmark"
    )
    parser.add_argument("--horizon", type=int, default=40, help="MPC horizon")
    parser.add_argument(
        "--num_sim_steps", type=int, default=50, help="Number of simulation steps"
    )
    parser.add_argument(
        "--pcg_eps",
        type=float,
        default=1e-12,
        help="PCG linear system solver tolerance",
    )
    parser.add_argument(
        "--cold_start", action="store_true", help="Do not warm start the solver."
    )
    parser.add_argument(
        "--num_repeats",
        type=int,
        default=10,
        help="Number of repeats of the experimrents",
    )

    args = parser.parse_args()
    warm_start = not (args.cold_start)
    print(f"\n{'=' * 70}")
    print(f"Num repeats: {args.num_repeats}")
    print(
        f"DiffMPC Rollout: {N_STATE} states, {N_CTRL} controls, {args.horizon} horizon"
    )
    print(f"Simulation: {args.num_sim_steps} steps, {args.batch_size} batch")
    print(
        f"Settings: MPC({MPC_LQR_ITER} iter, tol={MPC_TOL}, PCG tol={args.pcg_eps},"
        f" warm start={warm_start})"
    )

    forward_times, backward_times = benchmark_diffmpc_rollout(
        args.batch_size,
        args.num_sim_steps,
        args.horizon,
        args.pcg_eps,
        warm_start,
        args.num_repeats,
    )

    dirname = f"timing_{args.batch_size}_{args.horizon}_{N_STATE+N_CTRL}_pcg={args.pcg_eps}_warmstart={warm_start}"
    device = jnp.zeros(1).device
    os.makedirs(f"timing_results/{dirname}", exist_ok=True)
    np.save(f"timing_results/{dirname}/diffmpc_{device}_fwd", forward_times)
    np.save(f"timing_results/{dirname}/diffmpc_{device}_bwd", backward_times)


if __name__ == "__main__":
    main()

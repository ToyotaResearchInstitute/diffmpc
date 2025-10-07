"""Benchmarking script. Running in trajax docker environment is recommended."""
import argparse
import os
import time

import numpy as np
from jax import config

config.update("jax_enable_x64", True)  # use double precision

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax import jit, vmap  # noqa: E402
from trajax import optimizers  # noqa: E402
from utils import (  # noqa: E402
    MPC_LQR_ITER,
    MPC_TOL,
    N_CTRL,
    N_STATE,
    TRAJAX_ALPHA_MIN,
    TRAJAX_LQR_ITER,
    generate_problem_data,
)


def reward(state, control):
    r = -(jnp.sum(state**2) + jnp.sum(control**2))
    return r


def benchmark_trajax_rollout(N_BATCH, sim_steps, HORIZON):
    """Benchmark trajax ilqr in rollout mode"""
    print("=== Trajax iLQR Rollout ===")

    all_times_fwd = []
    all_times_bwd = []
    for seed in range(10):
        Q, R, A_matrix, B_matrix, b_vector, initial_states = generate_problem_data(
            N_BATCH, seed
        )

        A_matrix = jnp.array(A_matrix)
        B_matrix = jnp.array(B_matrix)
        b_vector = jnp.array(b_vector)
        initial_states = jnp.array(initial_states)
        Q = jnp.array(Q)
        R = jnp.array(R)
        print(f"device: {A_matrix.device()}")
        Q_weights = jnp.diag(Q)

        def dynamics_fn(x, u, t):
            return A_matrix @ x + B_matrix @ u + b_vector

        def make_cost_fn(Q_weights):
            def cost(x, u, t):
                return jnp.sum(x**2 * Q_weights) + jnp.sum(u**2 * R)

            return cost

        def single_ilqr(cost_fn, dynamics_fn, initial_states, U_in):
            X, U_opt, obj, grad_val, adjoints, lqr, it = optimizers.ilqr(
                cost_fn,
                dynamics_fn,
                initial_states,
                U_in,
                maxiter=TRAJAX_LQR_ITER,
                alpha_min=TRAJAX_ALPHA_MIN,
            )
            return U_opt

        def rollout(initial_states, Q_weights):
            cost_fn = make_cost_fn(Q_weights)
            initial_carry = (initial_states, 0.0, jnp.zeros((HORIZON, N_CTRL)))

            def rollout_step(carry, _):
                current_state, running_cost, last_control = carry

                # Solve MPC problem
                U_opt = single_ilqr(cost_fn, dynamics_fn, current_state, last_control)
                u_applied = U_opt[0]

                # Simulate forward
                new_state = A_matrix @ current_state + B_matrix @ u_applied + b_vector

                # Update running cost
                new_running_cost = running_cost - reward(new_state, u_applied)

                return (new_state, new_running_cost, U_opt), None

            final_carry, _ = jax.lax.scan(
                rollout_step, initial_carry, None, length=sim_steps
            )
            _, final_cost, _ = final_carry
            return final_cost

        rollout_batch = jit(vmap(rollout, in_axes=(0, None)))

        def rollout_for_grad(Q_weights):
            return jnp.sum(rollout_batch(initial_states, Q_weights))

        rollout_grad = jit(jax.grad(rollout_for_grad))

        # jit compile forward
        out = rollout_batch(jnp.zeros_like(initial_states), Q_weights)
        jax.block_until_ready(out)

        # time forward
        start_time = time.monotonic()
        final_costs = rollout_batch(initial_states, Q_weights)
        jax.block_until_ready(final_costs)
        total_time = time.monotonic() - start_time

        # jit compile backward
        r = rollout_grad(Q_weights)
        jax.block_until_ready(r)

        # time backward
        start_time = time.monotonic()
        gradients = rollout_grad(Q_weights)
        jax.block_until_ready(gradients)
        grad_time = time.monotonic() - start_time

        print(
            f"Time: {total_time*1000:.1f} ms"
            f" ({total_time*1000/(N_BATCH*sim_steps):.3f} ms/step/problem)"
        )
        print(f"Aggregate cost: {final_costs.sum():.4f}")
        print(
            f"grad time: {grad_time*1000:.1f} ms"
            f" ({grad_time*1000/(N_BATCH*sim_steps):.3f} ms/step/problem)"
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
        "--sim_steps", type=int, default=50, help="Number of simulation steps"
    )

    args = parser.parse_args()

    print(f"\n{'=' * 70}")
    print(f"MPC Rollout: {N_STATE} states, {N_CTRL} controls, {args.horizon} horizon")
    print(f"Simulation: {args.sim_steps} steps, {args.batch_size} batch")
    print(f"Settings: MPC({MPC_LQR_ITER} iter, tol={MPC_TOL})")

    forward_times, backward_times = benchmark_trajax_rollout(
        args.batch_size, args.sim_steps, args.horizon
    )

    dirname = f"timing_{args.batch_size}_{args.horizon}_{N_STATE+N_CTRL}"
    device = jnp.zeros(1).device()
    os.makedirs(f"timing_results/{dirname}", exist_ok=True)
    np.save(f"timing_results/{dirname}/trajax_{device}_fwd", forward_times)
    np.save(f"timing_results/{dirname}/trajax_{device}_bwd", backward_times)


if __name__ == "__main__":
    main()

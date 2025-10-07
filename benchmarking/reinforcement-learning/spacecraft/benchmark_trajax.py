"""Benchmark. Running in trajax docker environment is recommended."""
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
    N_CTRL,
    TRAJAX_ALPHA_MIN,
    TRAJAX_LQR_ITER,
    generate_problem_data,
    reward,
)

from diffmpc.dynamics.spacecraft_dynamics import SpacecraftDynamics  # noqa: E402
from diffmpc.utils.load_params import load_problem_params  # noqa: E402


def benchmark_trajax_rollout(num_batch, num_sim_steps, horizon):
    """Benchmark trajax ilqr in rollout mode"""
    print("=== Trajax iLQR Rollout ===")
    print(f"device: {jnp.ones(3).device()}")

    problem_params = load_problem_params("spacecraft.yaml")
    dynamics = SpacecraftDynamics()

    all_times_fwd = []
    all_times_bwd = []
    for seed in range(10):
        inertia_vector, initial_states = generate_problem_data(num_batch, seed)
        inertia_vector = jnp.array(inertia_vector)
        initial_states = jnp.array(initial_states)
        problem_params = {
            **problem_params,
            "inertia_vector": inertia_vector,
        }
        Q = jnp.array(problem_params["weights_penalization_reference_state_trajectory"])
        R = jnp.array(problem_params["weights_penalization_control_squared"])
        Q_weights = jnp.diag(Q)
        R_weights = jnp.diag(R)

        def dynamics_fn(x, u, t):
            dt = problem_params["discretization_resolution"]
            next_state = x + dt * dynamics.state_dot(x, u, problem_params)
            return next_state

        def make_cost_fn(Q_weights, R_weights):
            def cost(x, u, t):
                return jnp.sum(x**2 * Q_weights) + jnp.sum(u**2 * R_weights)

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

        def rollout(initial_states, QR_weights):
            Q_weights, R_weights = QR_weights[:-N_CTRL], QR_weights[-N_CTRL:]
            cost_fn = make_cost_fn(Q_weights, R_weights)
            initial_carry = (initial_states, 0.0, jnp.zeros((horizon, N_CTRL)))

            def rollout_step(carry, _):
                current_state, running_reward, last_control = carry

                # Solve MPC problem
                U_opt = single_ilqr(cost_fn, dynamics_fn, current_state, last_control)
                u_applied = U_opt[0]

                # Simulate forward
                new_state = dynamics_fn(current_state, u_applied, t=0)

                # Update running cost
                running_reward = running_reward + reward(new_state, u_applied)

                return (new_state, running_reward, U_opt), None

            final_carry, _ = jax.lax.scan(
                rollout_step, initial_carry, None, length=num_sim_steps
            )
            _, final_reward, _ = final_carry
            return final_reward

        rollout_batch = jit(vmap(rollout, in_axes=(0, None)))

        def rollout_for_grad(QR_weights):
            return jnp.sum(rollout_batch(initial_states, QR_weights))

        rollout_grad = jit(jax.grad(rollout_for_grad))

        QR_weights = jnp.concatenate([Q_weights, R_weights])
        jax.block_until_ready(QR_weights)

        # jit compile forward
        out = rollout_batch(
            jnp.zeros_like(initial_states), jnp.concatenate([Q_weights, R_weights])
        )
        jax.block_until_ready(out)

        # time forward
        start_time = time.monotonic()
        final_costs = rollout_batch(initial_states, QR_weights)
        jax.block_until_ready(final_costs)
        total_time = time.monotonic() - start_time

        # jit compile backward
        r = rollout_grad(QR_weights)
        jax.block_until_ready(r)

        # time backward
        start_time = time.monotonic()
        gradients = rollout_grad(QR_weights)
        jax.block_until_ready(gradients)
        grad_time = time.monotonic() - start_time

        print(
            f"Time: {total_time*1000:.1f} ms"
            f" ({total_time*1000/(num_batch*num_sim_steps):.3f} ms/step/problem)"
        )
        print(f"Aggregate cost: {final_costs.sum():.4f}")
        print(
            f"grad time: {grad_time*1000:.1f} ms"
            f" ({grad_time*1000/(num_batch*num_sim_steps):.3f} ms/step/problem)"
        )

        all_times_fwd.append(total_time)
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
    print(f"MPC Rollout: {args.horizon} horizon")
    print(f"Simulation: {args.num_sim_steps} steps, {args.batch_size} batch")

    forward_times, backward_times = benchmark_trajax_rollout(
        args.batch_size, args.num_sim_steps, args.horizon
    )

    dirname = f"spacecraft_timing_{args.batch_size}_{args.horizon}"
    device = jnp.zeros(1).device()
    os.makedirs(f"timing_results/{dirname}", exist_ok=True)
    np.save(f"timing_results/{dirname}/trajax_{device}_fwd", forward_times)
    np.save(f"timing_results/{dirname}/trajax_{device}_bwd", backward_times)


if __name__ == "__main__":
    main()

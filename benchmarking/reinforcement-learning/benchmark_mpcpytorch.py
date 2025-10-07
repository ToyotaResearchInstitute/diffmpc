"""Benchmarking script."""
import argparse
import os
import time

import numpy as np
import torch

# MPC.pytorch imports
from mpc import mpc
from mpc.dynamics import AffineDynamics
from mpc.mpc import QuadCost
from utils import (
    MPC_LQR_ITER,
    MPC_MAX_LINES,
    MPC_TOL,
    N_CTRL,
    N_STATE,
    generate_problem_data,
)


def benchmark_mpc_pytorch_rollout(device, N_BATCH, sim_steps, HORIZON):
    """Benchmark mpc.pytorch"""
    print("=== MPC.pytorch ===")

    device = torch.device(device)
    print(f"Using device: {device}")

    all_times_fwd = []
    all_times_bwd = []
    for seed in range(10):
        Q, R, A_base, B_base, b_base, x0 = generate_problem_data(N_BATCH, seed)

        # Move to GPU
        Q, R, A_base, B_base, b_base, x0 = [
            torch.from_numpy(x).to(device) for x in [Q, R, A_base, B_base, b_base, x0]
        ]

        # Create cost
        c_batch = torch.zeros(
            HORIZON, N_BATCH, N_STATE + N_CTRL, dtype=torch.double, device=device
        )

        dynamics = AffineDynamics(A_base, B_base, b_base)

        # Create solver
        mpc_solver = mpc.MPC(
            n_state=N_STATE,
            n_ctrl=N_CTRL,
            T=HORIZON,
            u_lower=None,
            u_upper=None,
            lqr_iter=MPC_LQR_ITER,
            eps=MPC_TOL,
            n_batch=N_BATCH,
            max_linesearch_iter=MPC_MAX_LINES,
            verbose=0,
            backprop=True,
            grad_method=mpc.GradMethods.ANALYTIC,
            exit_unconverged=False,
            detach_unconverged=False,
        )

        def rollout_pytorch(Q_weights, x_init):
            current_states = x_init.clone()
            total_costs = 0.0
            for step in range(sim_steps):
                QR_weighted = torch.block_diag(torch.diag(Q_weights), R)
                H_weighted = (
                    QR_weighted.unsqueeze(0)
                    .unsqueeze(0)
                    .expand(HORIZON, N_BATCH, -1, -1)
                )
                quad_cost_weighted = QuadCost(H_weighted, c_batch)

                x_sol, u_sol, _ = mpc_solver(
                    current_states, quad_cost_weighted, dynamics
                )
                # print(f"u_sol: {u_sol.shape}")
                u_applied = u_sol[0]
                new_states = (
                    A_base @ current_states.T
                    + B_base @ u_applied.T
                    + b_base.unsqueeze(1)
                ).T
                current_states = new_states
                total_costs += torch.sum(new_states**2) + torch.sum(u_applied**2)
            return total_costs

        def rollout_for_grad(Q_weights):
            return rollout_pytorch(Q_weights, x0).sum()

        # Warmup forward
        _ = rollout_pytorch(
            torch.ones(N_STATE, dtype=torch.double, device=device), torch.zeros_like(x0)
        )

        # Time forward
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start_time = time.monotonic()
        final_cost = rollout_pytorch(torch.diag(Q), x0)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_time = time.monotonic() - start_time

        print(
            f"Time: {total_time*1000:.1f} ms"
            f" ({total_time*1000/(N_BATCH*sim_steps):.3f} ms/step/problem)"
        )
        print(f"Aggregate cost: {final_cost.sum():.4f}")

        # Warmup backward
        Q_weights = torch.ones(
            N_STATE, dtype=torch.double, device=device, requires_grad=True
        )
        rollout_for_grad(Q_weights).backward()
        Q_weights.grad.zero_()

        # Time backward
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        grad_start_time = time.monotonic()
        rollout_for_grad(Q_weights).backward()
        # print(f"Q_weights.grad: {Q_weights.grad}")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        grad_time = time.monotonic() - grad_start_time

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
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")

    args = parser.parse_args()

    print(f"\n{'=' * 70}")
    print(f"MPC Rollout: {N_STATE} states, {N_CTRL} controls, {args.horizon} horizon")
    print(f"Simulation: {args.sim_steps} steps, {args.batch_size} batch")
    print(f"Settings: MPC({MPC_LQR_ITER} iter, tol={MPC_TOL})")

    forward_times, backward_times = benchmark_mpc_pytorch_rollout(
        args.device, args.batch_size, args.sim_steps, args.horizon
    )

    dirname = f"timing_{args.batch_size}_{args.horizon}_{N_STATE+N_CTRL}"
    device = args.device
    os.makedirs(f"timing_results/{dirname}", exist_ok=True)
    np.save(f"timing_results/{dirname}/mpcpytorch_{device}_fwd", forward_times)
    np.save(f"timing_results/{dirname}/mpcpytorch_{device}_bwd", backward_times)


if __name__ == "__main__":
    main()

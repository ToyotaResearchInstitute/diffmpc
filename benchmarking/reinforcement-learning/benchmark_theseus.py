"""Benchmarking script."""
import argparse
import os
import time

import numpy as np

# Theseus imports
import theseus as th
import torch
from utils import (
    MPC_LQR_ITER,
    MPC_TOL,
    N_CTRL,
    N_STATE,
    THESEUS_MAX_ITER,
    generate_problem_data,
)


def benchmark_theseus_rollout(device, N_BATCH, sim_steps, HORIZON):
    """Benchmark Theseus in MPC rollout mode using least-squares formulation"""
    print("=== Theseus Rollout ===")

    device = torch.device(device)
    print(f"Using device: {device}")
    all_times_fwd = []
    all_times_bwd = []
    for seed in range(10):
        Q, R, A_base, B_base, b_base, x0 = generate_problem_data(N_BATCH, seed)

        # Move to device
        Q, R, A, B, b, x0 = [
            torch.from_numpy(x).to(device) for x in [Q, R, A_base, B_base, b_base, x0]
        ]
        sqrtR = torch.sqrt(torch.diag(R))  # if R diagonal; else use chol below

        def build_theseus_layer():
            """Build Theseus optimization layer with Q weights"""
            n, m = A.shape[0], B.shape[1]
            dtype, device = A.dtype, A.device

            # Set PyTorch default dtype for Theseus consistency
            original_dtype = torch.get_default_dtype()
            torch.set_default_dtype(dtype)

            def mpc_cost_fn(optim_vars, aux_vars):
                """MPC cost function - returns scalar cost per batch element"""
                u_tensor = optim_vars[0].tensor  # (B, H*m)
                x0_batch = aux_vars[0].tensor  # (B, n)
                Qw = aux_vars[1].tensor  # (B, n)  diag weights per state
                Bsz = u_tensor.shape[0]
                Uv = u_tensor.view(Bsz, HORIZON, m)

                # Roll out dynamics and accumulate cost
                x = x0_batch
                res_chunks = []

                for t in range(HORIZON):
                    u_t = Uv[:, t, :]  # (B, m)
                    # --- dynamics (choose one and use it everywhere) ---
                    x = (A @ x.T + B @ u_t.T + b.unsqueeze(1)).T  # (B, n)
                    # x = x + dt * (A @ x.T + B @ u_t.T + b.unsqueeze(1)).T  # if Euler

                    # residuals: sqrt(Q) x_t, sqrt(R) u_t
                    sqrtQ = torch.sqrt(Qw.clamp_min(0))  # (B, n)
                    res_chunks.append(sqrtQ * x)  # (B, n)

                    # If R is diagonal:
                    res_chunks.append(u_t * sqrtR)  # (B, m)

                    # If R full PD:
                    # res_chunks.append((u_t @ LR.T))   # (B, m)

                residual = torch.cat(res_chunks, dim=1)  # (B, H*(n+m))
                return residual

            # Create Theseus variables
            u_vars = th.Vector(
                tensor=torch.zeros(N_BATCH, HORIZON * m, dtype=dtype, device=device),
                name="controls",
            )
            x0_var = th.Variable(
                tensor=torch.zeros(N_BATCH, n, dtype=dtype, device=device),
                name="initial_state",
            )
            Q_var = th.Variable(
                tensor=torch.ones(N_BATCH, n, dtype=dtype, device=device),
                name="Q_weights",
            )

            # Create cost function
            cost = th.AutoDiffCostFunction(
                [u_vars],
                mpc_cost_fn,
                dim=HORIZON * (N_STATE + N_CTRL),
                aux_vars=[x0_var, Q_var],
                name="mpc_cost",
            )

            objective = th.Objective(dtype=dtype)
            objective.add(cost)

            optimizer = th.LevenbergMarquardt(
                objective, th.CholeskyDenseSolver, max_iterations=THESEUS_MAX_ITER
            )
            layer = th.TheseusLayer(optimizer)
            layer.to(device=device, dtype=dtype)

            # Restore original dtype
            torch.set_default_dtype(original_dtype)

            return layer

        # Build layer once
        layer = build_theseus_layer()

        def rollout_theseus(Q_weights):
            """Perform MPC rollout with Theseus optimization - based on tdmpc.py approach"""
            current_states = x0.clone()
            total_cost = 0.0

            # Initialize control sequence
            controls = torch.zeros(
                N_BATCH, HORIZON * N_CTRL, dtype=torch.double, device=device
            )

            for step in range(sim_steps):
                # Prepare Theseus inputs
                theseus_inputs = {
                    "controls": controls,
                    "initial_state": current_states,
                    "Q_weights": Q_weights.unsqueeze(0).expand(N_BATCH, -1),
                }

                sol, info = layer.forward(
                    theseus_inputs, optimizer_kwargs={"backward_mode": "implicit"}
                )
                # print(f"info: {info}")

                # Extract control solution (use main solution for gradients)
                controls = sol["controls"]

                controls = controls.to(device)
                controls_reshaped = controls.view(N_BATCH, HORIZON, N_CTRL)

                # Apply first control action (MPC receding horizon)
                u_applied = controls_reshaped[
                    :, 0, :
                ]  # First control for each batch element

                # Simulate dynamics forward
                new_states = (A @ current_states.T + B @ u_applied.T + b.unsqueeze(1)).T
                new_states = new_states.to(device)
                current_states = new_states

                total_cost += torch.sum(new_states**2) + torch.sum(u_applied**2)

            return total_cost

        def rollout_for_grad(Q_weights):
            return rollout_theseus(Q_weights)

        # Warmup forward
        _ = rollout_theseus(torch.ones(N_STATE, dtype=torch.double, device=device))

        # Time forward
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start_time = time.monotonic()
        final_cost = rollout_theseus(torch.diag(Q))

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_time = time.monotonic() - start_time

        print(
            f"Time: {total_time*1000:.1f} ms"
            f" ({total_time*1000/(N_BATCH*sim_steps):.3f} ms/step/problem)"
        )
        print(f"Aggregate cost: {final_cost:.4f}")

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
    parser = argparse.ArgumentParser(description="Theseus Benchmark")
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
    print(
        f"Theseus Rollout: {N_STATE} states, {N_CTRL} controls, {args.horizon} horizon"
    )
    print(f"Simulation: {args.sim_steps} steps, {args.batch_size} batch")
    print(f"Settings: MPC({MPC_LQR_ITER} iter, tol={MPC_TOL})")

    forward_times, backward_times = benchmark_theseus_rollout(
        args.device, args.batch_size, args.sim_steps, args.horizon
    )

    dirname = f"timing_{args.batch_size}_{args.horizon}_{N_STATE+N_CTRL}"
    device = args.device
    os.makedirs(f"timing_results/{dirname}", exist_ok=True)
    np.save(f"timing_results/{dirname}/theseus_{device}_fwd", forward_times)
    np.save(f"timing_results/{dirname}/theseus_{device}_bwd", backward_times)


if __name__ == "__main__":
    main()

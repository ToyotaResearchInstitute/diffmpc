"""Utilities"""
import numpy as np

# Problem dimensions
N_STATE = 8  # state dimension
N_CTRL = 4  # control dimension

# Randomness
INITIAL_STATE_STDEV = 5.0

# Solver settings
# mpc.pytorch
MPC_LQR_ITER = 1  # mpc.pytorch iterations
MPC_TOL = 1e-3  # mpc.pytorch tolerance
MPC_MAX_LINES = 1  # mpc.pytorch max linesearch iterations
# diffmpc
DIFFMPC_SCP_ITER = 1  # diffmpc SQP iterations
# trajax
TRAJAX_LQR_ITER = 1  # trajax iLQR iterations
TRAJAX_ALPHA_MIN = 0.99  # trajax iLQR alpha min
# theseus
THESEUS_MAX_ITER = 1  # theseus max iterations


def generate_problem_data(N_BATCH, seed):
    """Generate problem data (matrices and initial states) for benchmarking"""
    # Set seeds for reproducibility
    np.random.seed(seed)

    # Cost matrices (identity)
    Q = np.eye(N_STATE, dtype=np.double)
    R = np.eye(N_CTRL, dtype=np.double)

    # Create a stable A matrix with eigenvalues strictly inside unit circle
    A_base = np.eye(N_STATE) + 0.1 * np.random.randn(N_STATE, N_STATE)

    # Restrict eigenvalues of A to be less than 1. Based on
    # https://github.com/osqp/osqp_benchmarks/blob/master/problem_classes/control.py.
    lambda_values, V = np.linalg.eig(A_base)
    abs_lambda_values = np.abs(lambda_values)
    for i in range(len(lambda_values)):
        lambda_values[i] = (
            lambda_values[i]
            if abs_lambda_values[i] < 1 - 1e-02
            else lambda_values[i] / (abs_lambda_values[i] + 1e-02)
        )
    # Reconstruct A = V * Lambda * V^{-1}
    A_base = (V @ np.diag(lambda_values) @ np.linalg.inv(V)).real

    B_base = np.random.randn(N_STATE, N_CTRL)
    b_base = 0.01 * np.random.randn(N_STATE)

    # Initial states (batch)
    x0 = np.random.randn(N_BATCH, N_STATE) * INITIAL_STATE_STDEV

    return Q, R, A_base, B_base, b_base, x0

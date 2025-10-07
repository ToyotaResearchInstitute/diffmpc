"""Tests for the PCG solver."""

import jax.numpy as jnp
import numpy as np
import pytest
from jax import config

config.update("jax_enable_x64", True)  # use double precision

from diffmpc.solvers.pcg import (  # noqa: E402
    PCGOptimalControl,
    QPCostParameters,
    QPDynamicsParameters,
    QPParameters,
)


def test_constructor_and_properties():
    """Test solver constructor."""
    solver_params = {"max_iter": 50, "tol_epsilon": 1e-5}
    oc = PCGOptimalControl(10, 4, 2, solver_params)

    assert oc.name == "PCGOptimalControl"
    assert oc.horizon == 10
    assert oc.num_states == 4
    assert oc.num_controls == 2
    assert oc.params == solver_params


def test_zero_qp_parameters_shapes():
    """Test that dimensions of all QP matrices are correctly defined."""
    oc = PCGOptimalControl(5, 3, 2)
    qp_params = oc.zero_qp_parameters()

    assert qp_params.dynamics.As_next.shape == (5, 3, 3)
    assert qp_params.dynamics.As.shape == (5, 3, 3)
    assert qp_params.dynamics.Bs.shape == (5, 3, 2)
    assert qp_params.dynamics.Cs.shape == (6, 3)
    assert qp_params.cost.Qinv.shape == (6, 3, 3)
    assert qp_params.cost.qvec.shape == (6, 3)
    assert qp_params.cost.Rinv.shape == (5, 2, 2)
    assert qp_params.cost.rvec.shape == (5, 2)


def test_zero_schur_and_dynamics_cost_shapes():
    """Test that dimensions of the cost matrices are correctly defined."""
    oc = PCGOptimalControl(4, 2, 1)
    dyn_cost = oc.zero_dynamics_cost_premultiplied_matrices()
    schur = oc.zero_schur_complement_matrices()

    assert dyn_cost.As_x_Qinv.shape == (4, 2, 2)
    assert dyn_cost.Asnext_x_Qinvnext.shape == (4, 2, 2)
    assert dyn_cost.Bs_x_Rinv.shape == (4, 2, 1)
    assert schur.S.shape == (5, 2, 6)
    assert schur.preconditioner_Phiinv.shape == (5, 2, 6)


@pytest.mark.parametrize("T", [2, 5])
@pytest.mark.parametrize("nx", [1, 5])
@pytest.mark.parametrize("nu", [1, 5])
def test_compute_S_Phiinv_identity_dynamics(T, nx, nu):
    """Test computation of the preconditioner."""

    oc = PCGOptimalControl(T, nx, nu, {"max_iter": 10, "tol_epsilon": 1e-8})

    qp_params = QPParameters(
        QPDynamicsParameters(
            As=jnp.broadcast_to(jnp.eye(nx), (T, nx, nx)),
            As_next=jnp.broadcast_to(jnp.eye(nx), (T, nx, nx)),
            Bs=jnp.ones((T, nx, nu)),
            Cs=jnp.ones((T, nx)),
        ),
        QPCostParameters(
            Qinv=jnp.tile(jnp.eye(nx), (T + 1, 1, 1)),
            qvec=jnp.zeros((T + 1, nx)),
            Rinv=jnp.tile(jnp.eye(nu), (T, 1, 1)),
            rvec=jnp.zeros((T, nu)),
        ),
    )

    schur = oc.compute_S_Phiinv(qp_params)
    print(schur.S.shape)
    assert schur.S.shape == (T + 1, nx, nx * 3)
    assert schur.preconditioner_Phiinv.shape[0] == T + 1
    assert not jnp.isnan(schur.S).any()


@pytest.mark.parametrize("T", [2, 5])
@pytest.mark.parametrize("nx", [1, 5])
@pytest.mark.parametrize("nu", [1, 5])
def test_compute_gamma_zeros(T, nx, nu):
    """Test computation of the RHS gamma."""

    oc = PCGOptimalControl(T, nx, nu)
    dyn_cost = oc.zero_dynamics_cost_premultiplied_matrices()
    Cs = jnp.zeros((T + 1, nx))
    Q0inv = jnp.eye(nx)
    qs = jnp.zeros((T + 1, nx))
    rs = jnp.zeros((T, nu))

    gamma = oc.compute_gamma(dyn_cost, Cs, Q0inv, qs, rs)
    assert gamma.shape == (T + 1, nx)
    assert jnp.allclose(gamma, 0.0)


@pytest.mark.parametrize("T", [2, 5])
@pytest.mark.parametrize("nx", [1, 5])
@pytest.mark.parametrize("nu", [1, 5])
def test_get_states_controls_from_zero_multipliers(T, nx, nu):
    """Test retriving the solution."""
    oc = PCGOptimalControl(T, nx, nu)
    og_params = oc.zero_qp_parameters()

    qp_params = QPParameters(
        QPDynamicsParameters(
            As_next=og_params.dynamics.As_next,
            As=og_params.dynamics.As,
            Bs=og_params.dynamics.Bs,
            Cs=og_params.dynamics.Cs,
        ),
        QPCostParameters(
            Qinv=jnp.tile(jnp.eye(nx), (T + 1, 1, 1)),
            qvec=jnp.zeros((T + 1, 1)),
            Rinv=jnp.tile(jnp.eye(nu), (T, 1, 1)),
            rvec=jnp.zeros((T, 1)),
        ),
    )

    lambdas = jnp.zeros((T + 1, nx))
    sol = oc.get_states_controls_from_kkt_multipliers(lambdas, qp_params)

    assert sol.states.shape == (T + 1, nx)
    assert sol.controls.shape == (T + 1, nu)
    assert jnp.allclose(sol.states, 0.0)
    assert jnp.allclose(sol.controls, 0.0)


@pytest.mark.parametrize("T", [2, 3])
@pytest.mark.parametrize("nx", [1, 2])
@pytest.mark.parametrize("nu", [1, 2])
def test_solve_KKT_Schur_trivial(T, nx, nu):
    """Test solving a trivial KKT system."""

    oc = PCGOptimalControl(T, nx, nu, {"max_iter": 10, "tol_epsilon": 1e-8})

    qp_params = QPParameters(
        QPDynamicsParameters(
            As=jnp.broadcast_to(jnp.eye(nx), (T, nx, nx)),
            As_next=jnp.broadcast_to(jnp.eye(nx), (T, nx, nx)),
            Bs=jnp.ones((T, nx, nu)),
            Cs=jnp.ones((T, nx)),
        ),
        QPCostParameters(
            Qinv=jnp.tile(jnp.eye(nx), (T + 1, 1, 1)),  # shape = (6, 3, 3)
            qvec=jnp.zeros((T + 1, nx)),
            Rinv=jnp.tile(jnp.eye(nu), (T, 1, 1)),  # shape = (5, 2, 2)
            rvec=jnp.zeros((T, nu)),
        ),
    )

    schur = oc.compute_S_Phiinv(qp_params)

    dyn_cost = oc.zero_dynamics_cost_premultiplied_matrices()
    Q0inv = jnp.eye(nx)
    qs = jnp.zeros((T + 1, nx))
    rs = jnp.zeros((T, nu))
    # Cs length must match T+1
    gammas = oc.compute_gamma(dyn_cost, jnp.ones((T + 1, nx)), Q0inv, qs, rs)
    lambdas_guess = jnp.zeros((T + 1, nx))

    sol, debug = oc.solve_KKT_Schur(qp_params, schur, gammas, lambdas_guess)

    assert sol.states.shape == (T + 1, nx)
    assert sol.controls.shape == (T + 1, nu)
    assert debug.num_iterations <= 10
    assert debug.convergence_eta >= 0.0


def test_solve_KKT_Schur_warmstart():
    """Test warm-starting PCG."""
    (T, nx, nu) = (3, 4, 5)

    oc = PCGOptimalControl(T, nx, nu, {"max_iter": 20, "tol_epsilon": 1e-13})

    qp_params = QPParameters(
        QPDynamicsParameters(
            As=jnp.broadcast_to(jnp.diag(np.random.rand(nx)), (T, nx, nx)),
            As_next=jnp.broadcast_to(jnp.diag(np.random.rand(nx)), (T, nx, nx)),
            Bs=jnp.ones((T, nx, nu)),
            Cs=jnp.ones((T, nx)),
        ),
        QPCostParameters(
            Qinv=jnp.tile(
                jnp.diag(np.random.rand(nx)), (T + 1, 1, 1)
            ),  # shape = (6, 3, 3)
            qvec=jnp.zeros((T + 1, nx)),
            Rinv=jnp.tile(jnp.eye(nu), (T, 1, 1)),  # shape = (5, 2, 2)
            rvec=jnp.zeros((T, nu)),
        ),
    )
    schur = oc.compute_S_Phiinv(qp_params)
    dyn_cost = oc.zero_dynamics_cost_premultiplied_matrices()
    Q0inv = jnp.eye(nx)
    qs = jnp.zeros((T + 1, nx))
    rs = jnp.zeros((T, nu))
    # Cs length must match T+1
    gammas = oc.compute_gamma(dyn_cost, jnp.ones((T + 1, nx)), Q0inv, qs, rs)
    lambdas_guess = jnp.zeros((T + 1, nx))

    sol, debug = oc.solve_KKT_Schur(qp_params, schur, gammas, lambdas_guess)

    assert sol.states.shape == (T + 1, nx)
    assert sol.controls.shape == (T + 1, nu)
    assert debug.num_iterations <= 10
    assert debug.convergence_eta >= 0.0

    sol_warm, debug_warm = oc.solve_KKT_Schur(
        qp_params, schur, gammas, sol.kkt_multipliers
    )
    assert sol_warm.states.shape == (T + 1, nx)
    assert sol_warm.controls.shape == (T + 1, nu)
    assert debug_warm.num_iterations <= 10
    assert debug_warm.convergence_eta >= 0.0
    assert jnp.allclose(sol.states - sol_warm.states, 0.0)
    assert jnp.allclose(sol.controls - sol_warm.controls, 0.0)
    assert jnp.allclose(sol.kkt_multipliers - sol_warm.kkt_multipliers, 0.0)
    assert debug_warm.num_iterations < debug.num_iterations

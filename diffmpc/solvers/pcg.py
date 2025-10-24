"""Implements PCG method."""
from typing import Any, Dict, NamedTuple, Tuple

import jax.numpy as jnp
from jax import vmap
from jax.lax import while_loop

from diffmpc.utils.load_params import load_solver_params

DEFAULT_PCG_SOLVER_PARAMS = load_solver_params("sqp.yaml")["pcg"]


class QPDynamicsParameters(NamedTuple):
    """Dynamics parameters of the quadratic program (QP)."""

    # (N, nx, nu) = (horizon, num_states, num_controls)
    As_next: jnp.ndarray  # (N, nx, nx)
    As: jnp.ndarray  # (N, nx, nx)
    Bs: jnp.ndarray  # (N, nx, nu)
    Cs: jnp.ndarray  # (N+1, nx) (initial state, dynamics) constraints


class QPCostParameters(NamedTuple):
    """Cost parameters of the quadratic program (QP)."""

    # (N, nx, nu) = (horizon, num_states, num_controls)
    Qinv: jnp.ndarray  # (N+1, nx, nx)
    qvec: jnp.ndarray  # (N+1, nx)
    Rinv: jnp.ndarray  # (N, nu, nu)
    rvec: jnp.ndarray  # (N, nu)


class QPParameters(NamedTuple):
    """Parameters of the quadratic program (QP)."""

    dynamics: QPDynamicsParameters
    cost: QPCostParameters


class QPSolution(NamedTuple):
    """Solution of the QP."""

    # (N, nx, nu) = (horizon, num_states, num_controls)
    states: jnp.ndarray  # (N+1, nx)
    controls: jnp.ndarray  # (N, nu)
    kkt_multipliers: jnp.ndarray  # (N+1, nx)


class QPDynamicsCostPremultipliedMatrices(NamedTuple):
    As_x_Qinv: jnp.ndarray  # (N, nx, nx)
    Asnext_x_Qinvnext: jnp.ndarray  # (N, nx, nx)
    Bs_x_Rinv: jnp.ndarray  # (N, nx, nu)


class SchurComplementMatrices(NamedTuple):
    """Schur complement matrix parameters of the QP."""

    # (N, nx, nu) = (horizon, num_states, num_controls)
    S: jnp.ndarray  # (N+1, nx, 3 * nx)
    preconditioner_Phiinv: jnp.ndarray  # (N+1, nx, 3 * nx)


class PCGDebugOutput(NamedTuple):
    """Debug output after running PCG."""

    num_iterations: int
    convergence_eta: float


class PCGOptimalControl:
    """
    Preconditioned conjugate gradient (PCG) method for solving quadratic programs (QPs).

    Used to solve trajectory optimization problems of the form
    min_{x, u}  sum_{t=0}^N       0.5 x_t^T Q_t x_t + q_t x_t
                + sum_{t=0}^{N-1} 0.5 u_t^T R_t u_t + r_t u_t
    such that   A_next_t @ x_{t+1} + A_t @ x_t + B_t @ u_t = F_t
                x_0 = initial_state.

    The algorithm is described in https://arxiv.org/abs/2309.08079:
    Emre Adabag, Miloni Atal, William Gerard, Brian Plancher,
    MPCGPU: Real-Time Nonlinear Model Predictive Control through
    Preconditioned Conjugate Gradient on the GPU,
    ICRA 2024.

    This class gives a slight extension to handle the term A_next_t @ x_{t+1}
    to support implicit integrators for the dynamics equality constraints.
    """

    def __init__(
        self,
        problem_horizon: int,
        problem_num_states: int,
        problem_num_controls: int,
        solver_params: Dict[str, Any] = DEFAULT_PCG_SOLVER_PARAMS,
    ):
        """Initializes the class."""
        self._params = solver_params
        self._name = "PCGOptimalControl"
        self._problem_horizon = problem_horizon
        self._problem_num_states = problem_num_states
        self._problem_num_controls = problem_num_controls

    @property
    def name(self) -> str:
        """Returns the name of the class."""
        return self._name

    @property
    def params(self) -> Dict:
        """Returns the parameters of the class."""
        return self._params

    @property
    def horizon(self) -> int:
        """Returns the problem horizon."""
        return self._problem_horizon

    @property
    def num_states(self) -> int:
        """Returns the number of state variables."""
        return self._problem_num_states

    @property
    def num_controls(self) -> int:
        """Returns the number of control variables."""
        return self._problem_num_controls

    def zero_qp_parameters(self):
        N, nx, nu = self.horizon, self.num_states, self.num_controls
        return QPParameters(
            QPDynamicsParameters(
                jnp.zeros((N, nx, nx)),
                jnp.zeros((N, nx, nx)),
                jnp.zeros((N, nx, nu)),
                jnp.zeros((N + 1, nx)),
            ),
            QPCostParameters(
                jnp.zeros((N + 1, nx, nx)),
                jnp.zeros((N + 1, nx)),
                jnp.zeros((N, nu, nu)),
                jnp.zeros((N, nu)),
            ),
        )

    def zero_dynamics_cost_premultiplied_matrices(self):
        N, nx, nu = self.horizon, self.num_states, self.num_controls
        return QPDynamicsCostPremultipliedMatrices(
            jnp.zeros((N, nx, nx)),
            jnp.zeros((N, nx, nx)),
            jnp.zeros((N, nx, nu)),
        )

    def zero_schur_complement_matrices(self):
        N, nx = self.horizon, self.num_states
        return SchurComplementMatrices(
            jnp.zeros((N + 1, nx, 3 * nx)), jnp.zeros((N + 1, nx, 3 * nx))
        )

    def compute_S_Phiinv(self, qp_parameters: QPParameters) -> SchurComplementMatrices:
        """
        Computes the Schur complement matrix S and preconditioner matrix Phi^{-1}.

        Args:
            qp_parameters: parameters of the problem,
                (QPParameters)

        Returns:
            schur_complement_matrices: contains (S, Phi^{-1})
                (SchurComplementMatrices)
        """
        nx = self.num_states
        As_next, As, Bs, Rinv, Qinv = (
            qp_parameters.dynamics.As_next,
            qp_parameters.dynamics.As,
            qp_parameters.dynamics.Bs,
            qp_parameters.cost.Rinv,
            qp_parameters.cost.Qinv,
        )

        # 1) get theta, phi, zeta
        def get_theta_phi(Anext_prev, Anext, A, B, Qinv, Rinv, Qinv_next):
            A_x_Qinv, Anext_x_Qinvnext, B_x_Rinv = A @ Qinv, Anext @ Qinv_next, B @ Rinv
            theta = (
                A_x_Qinv @ A.T + B_x_Rinv @ B.T + Anext_x_Qinvnext @ Anext.T
            )  # (nx, nx)
            phi = A_x_Qinv @ Anext_prev.T  # (nx, nx)
            return theta, phi

        As_next_prev = jnp.concatenate([jnp.eye(nx)[jnp.newaxis], As_next[:-1]], axis=0)
        thetas, phis = vmap(get_theta_phi)(
            As_next_prev, As_next, As, Bs, Qinv[:-1], Rinv, Qinv[1:]
        )  # (horizon, nx, nx) and (horizon, nx, nx)
        # 2) form S, phiinv
        thetas = jnp.concatenate([Qinv[0][jnp.newaxis], thetas], axis=0)
        thetas_inv = vmap(jnp.linalg.inv)(thetas)

        def get_S(thetas, phis):
            def get_S_block(theta, phi, phi_next):
                block = -jnp.concatenate([phi, theta, phi_next.T], axis=1)
                return block

            phis_for_S = jnp.concatenate(
                [jnp.zeros((1, nx, nx)), phis, jnp.zeros((1, nx, nx))], axis=0
            )  # (1 + horizon + 1, nx, nx)
            S = vmap(get_S_block)(
                thetas, phis_for_S[:-1], phis_for_S[1:]
            )  # (horizon + 1, nx, 3 nx)
            return S

        S = get_S(thetas, phis)  # (horizon + 1, nx, 3*nx)

        def get_Phiinv(thetas_inv, phis):
            def get_Phiinv_block(
                theta_prev_inv, theta_inv, theta_next_inv, phi, phi_next
            ):
                prod_term_prev = -theta_inv @ phi @ theta_prev_inv
                prod_term_next = -theta_inv @ phi_next.T @ theta_next_inv
                block = jnp.concatenate(
                    [prod_term_prev, theta_inv, prod_term_next], axis=1
                )
                return block

            thetas_inv_padded = jnp.concatenate(
                [jnp.zeros((1, nx, nx)), thetas_inv, jnp.zeros((1, nx, nx))], axis=0
            )
            phis_padded = jnp.concatenate(
                [jnp.zeros((1, nx, nx)), phis, jnp.zeros((1, nx, nx))], axis=0
            )  # (1 + horizon + 1, nx, nx)
            Phiinv = vmap(get_Phiinv_block)(
                thetas_inv_padded[:-2],
                thetas_inv_padded[1:-1],
                thetas_inv_padded[2:],
                phis_padded[:-1],
                phis_padded[1:],
            )  # (horizon + 1, 3, 3 n)
            return Phiinv

        Phiinv = get_Phiinv(thetas_inv, phis)
        return SchurComplementMatrices(
            S=S,
            preconditioner_Phiinv=Phiinv,
        )

    def compute_gamma(
        self,
        dynamics_cost_premultiplied_matrices: QPDynamicsCostPremultipliedMatrices,
        Cs: jnp.ndarray,
        Q0inv: jnp.ndarray,
        qs: jnp.ndarray,
        rs: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Computes the vector gamma used to solve the QP via the Schur complement method.

        Dimensions: (N, nx, nu) = (horizon, num_states, num_controls)

        Args:
            dynamics_cost_premultiplied_matrices: premultiplied matrices,
                (QPDynamicsCostPremultipliedMatrices)
            Cs: dynamics constraints vectors
                (N + 1, nx) array
            Q0inv: inverse of the state cost matrix for the initial state,
                (nx, nx) array
            qs: state cost vectors,
                (N + 1, nx) array
            rs: control cost vectors,
                (N, nu) array

        Returns:
            gammas: vector gamma
                (N + 1, nx)
        """

        def get_zeta(q, r, qnext, A_x_Qinv, Anext_x_Qinvnext, B_x_Rinv):
            zeta = A_x_Qinv @ q + B_x_Rinv @ r + Anext_x_Qinvnext @ qnext  # (nx)
            return zeta

        zetas = vmap(get_zeta)(
            qs[:-1],
            rs,
            qs[1:],
            dynamics_cost_premultiplied_matrices.As_x_Qinv,
            dynamics_cost_premultiplied_matrices.Asnext_x_Qinvnext,
            dynamics_cost_premultiplied_matrices.Bs_x_Rinv,
        )
        gamma = Cs + jnp.concatenate([(Q0inv @ qs[0])[jnp.newaxis], zetas], axis=0)
        return gamma

    def get_states_controls_from_kkt_multipliers(
        self,
        kkt_multipliers: jnp.ndarray,
        qp_parameters: QPParameters,
    ) -> QPSolution:
        """
        Returns the optimal state-control-multipliers solution from the kkt multipliers lambda.

        Args:
            kkt_multipliers: kkt multipliers lambda,
                (horizon+1, num_states) array
            qp_parameters: parameters of the QP,
                (QPParameters)
        Returns:
            solution: solution to the QP,
                (QPSolution)
        """
        As_next = qp_parameters.dynamics.As_next
        As = qp_parameters.dynamics.As
        Bs = qp_parameters.dynamics.Bs
        Qinv = qp_parameters.cost.Qinv
        qvec = qp_parameters.cost.qvec
        Rinv = qp_parameters.cost.Rinv
        rvec = qp_parameters.cost.rvec
        nx, nu = self.num_states, self.num_controls

        def get_optimal_state_from_lambda(Qinv, q, Anext_prev, A, lambd, lambd_next):
            state = -Qinv @ (q + Anext_prev.T @ lambd + A.T @ lambd_next)
            return state

        def get_optimal_control_from_lambda(Rinv, r, B, lambd_next):
            control = -Rinv @ (r + B.T @ lambd_next)
            return control

        def get_optimal_final_state_from_lambda(QNinv, qN, ANprev, lambdaN):
            state = -QNinv @ (qN + ANprev.T @ lambdaN)
            return state

        As_next_shifted = jnp.concatenate(
            [jnp.eye(nx)[jnp.newaxis], As_next[:-1]], axis=0
        )
        states = vmap(get_optimal_state_from_lambda)(
            Qinv[:-1],
            qvec[:-1],
            As_next_shifted,
            As,
            kkt_multipliers[:-1],
            kkt_multipliers[1:],
        )  # (horizon, nx)
        controls = vmap(get_optimal_control_from_lambda)(
            Rinv, rvec, Bs, kkt_multipliers[1:]
        )  # (horizon, nu)
        state_last = get_optimal_final_state_from_lambda(
            Qinv[-1], qvec[-1], As_next[-1], kkt_multipliers[-1]
        )
        states = jnp.concatenate([states, state_last[jnp.newaxis]], axis=0)
        controls = jnp.concatenate([controls, jnp.zeros((1, nu))], axis=0)
        qp_solution = QPSolution(states, controls, kkt_multipliers)
        return qp_solution

    def solve_KKT_Schur(
        self,
        qp_parameters: QPParameters,
        schur_complement_matrices: SchurComplementMatrices,
        schur_complement_gammas: jnp.ndarray,
        kkt_multipliers_guess: jnp.ndarray,
    ) -> Tuple[QPSolution, PCGDebugOutput]:
        """
        Solves the QP using the Schur complement method.
        """
        S = schur_complement_matrices.S
        Phiinv = schur_complement_matrices.preconditioner_Phiinv
        gammas = schur_complement_gammas
        lambs = kkt_multipliers_guess
        # GBD-PCG
        pcg_max_iter = self.params["max_iter"]
        pcg_epsilon = self.params["tol_epsilon"]
        nx = self.num_states

        # 1) Initialize
        def pcg_get_r_init(gammas, S, lambs):
            def get_r_block(gamma, S, lamb_prev, lamb, lamb_next):
                block = gamma - S @ jnp.concatenate([lamb_prev, lamb, lamb_next])
                return block

            lambs_padded = jnp.concatenate(
                [jnp.zeros((1, nx)), lambs, jnp.zeros((1, nx))], axis=0
            )
            r = vmap(get_r_block)(
                gammas, S, lambs_padded[:-2], lambs_padded[1:-1], lambs_padded[2:]
            )
            return r

        r = pcg_get_r_init(gammas, S, lambs)  # (horizon + 1, 3)
        rs = jnp.concatenate(
            [
                jnp.concatenate([jnp.zeros((1, nx)), r[:-1]], axis=0),
                r,
                jnp.concatenate([r[1:], jnp.zeros((1, nx))], axis=0),
            ],
            axis=-1,
        )
        rtilde = jnp.matvec(Phiinv, rs)  # vmap(lambda A, v: A @ v)
        p = rtilde.copy()  # (horizon + 1, nx)
        eta = jnp.sum(r * rtilde)  # jnp.sum(vmap(jnp.dot)(r, rtilde))

        # 2) Solve linear system via PCG
        def cond_fun(val: Tuple):
            it, r, p, eta, lambs = val
            _continue = jnp.logical_and(
                jnp.abs(eta) > pcg_epsilon, it <= pcg_max_iter - 1
            )
            _continue = jnp.logical_or(_continue, it < 1)
            return _continue

        def pcg_iterate_fun(val):
            it, r, p, eta, lambs = val
            for _ in range(self.params["num_iter_check_termination"]):
                ps = jnp.concatenate(
                    [
                        jnp.concatenate([jnp.zeros((1, nx)), p[:-1]], axis=0),
                        p,
                        jnp.concatenate([p[1:], jnp.zeros((1, nx))], axis=0),
                    ],
                    axis=-1,
                )
                Upsilon = jnp.matvec(S, ps)  # vmap(lambda A, v: A @ v)
                v = jnp.sum(p * Upsilon)  # jnp.sum(vmap(jnp.dot)(p, Upsilon))
                alpha = eta / v
                lambs = lambs + alpha * p
                r = r - alpha * Upsilon
                rs = jnp.concatenate(
                    [
                        jnp.concatenate([jnp.zeros((1, nx)), r[:-1]], axis=0),
                        r,
                        jnp.concatenate([r[1:], jnp.zeros((1, nx))], axis=0),
                    ],
                    axis=-1,
                )
                rtilde = jnp.matvec(Phiinv, rs)  # vmap(lambda A, v: A @ v)
                etaprime = jnp.sum(r * rtilde)  # jnp.sum(vmap(jnp.dot)(r, rtilde))
                beta = etaprime / eta
                p = rtilde + beta * p
                eta = etaprime
                # jax.debug.print("eta = {eta}", eta=eta)
                it = it + 1
            val = (it, r, p, eta, lambs)
            return val

        val = while_loop(cond_fun, pcg_iterate_fun, init_val=(0, r, p, eta, lambs))
        # 3) get optimal solution (lambda* -> (x, u))
        qp_solution = self.get_states_controls_from_kkt_multipliers(
            kkt_multipliers=val[-1],
            qp_parameters=qp_parameters,
        )
        pcg_debug_output = PCGDebugOutput(
            num_iterations=val[0], convergence_eta=val[-2]
        )
        return qp_solution, pcg_debug_output

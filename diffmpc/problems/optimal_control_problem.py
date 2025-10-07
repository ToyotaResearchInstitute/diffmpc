"""Implements optimization problems."""

import copy
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from jax import vmap

from diffmpc.dynamics.base_dynamics import Dynamics
from diffmpc.dynamics.integrators import DiscretizationScheme, predict_next_state
from diffmpc.utils.jax_utils import value_and_jacfwd
from diffmpc.utils.load_params import check_parameters_dictionary_or_raise_errors


class OptimalControlProblem:
    """
    Optimal control problem class.
    """

    def __init__(
        self,
        dynamics: Dynamics,
        params: Dict[str, Any] = None,
        check_parameters_are_valid: bool = True,
    ):
        """Initializes the class."""
        self._dynamics = dynamics
        if check_parameters_are_valid:
            check_parameters_dictionary_or_raise_errors(params)
        params = copy.deepcopy(params)
        self._params = params
        self._discretization_scheme = DiscretizationScheme(
            params["discretization_scheme"]
        )
        self._num_variables = (
            self.num_control_variables + self.num_state_variables
        ) * (self.horizon + 1)
        self._num_constraints = self.num_state_variables * (self.horizon + 1)

    @property
    def dynamics(self) -> Dynamics:
        """Returns the dynamics of the class."""
        return self._dynamics

    @property
    def params(self) -> Dict:
        """Returns a dictionary of parameters of the program."""
        return self._params

    @property
    def horizon(self) -> int:
        """Returns the problem horizon."""
        return int(self.params["horizon"])

    @property
    def discretization_scheme(self) -> DiscretizationScheme:
        """Returns the discretization scheme."""
        return self._discretization_scheme

    @property
    def num_variables(self) -> int:
        """Returns the number of optimization variables."""
        return self._num_variables

    @property
    def num_constraints(self) -> int:
        """Returns the number of constraints."""
        num = self._num_constraints
        return num

    @property
    def num_state_variables(self) -> int:
        """Returns the number of state variables."""
        return self.dynamics.num_states

    @property
    def num_control_variables(self) -> int:
        """Returns the number of control variables."""
        return self.dynamics.num_controls

    def initial_guess(
        self, params: Dict[str, Any] = None
    ) -> Tuple[jnp.array, jnp.array]:
        """Returns an initial guess for the solution"""
        if params is None:
            params = self.params
        x_initial = params["initial_state"]
        x_final = params["final_state"]
        horizon = self.horizon

        # straight-line initial guess
        state_matrix = jnp.zeros((horizon + 1, self.num_state_variables))
        for t in range(horizon + 1):
            alpha1 = (horizon - t) / horizon
            alpha2 = t / horizon
            state_matrix = state_matrix.at[t].set(
                x_initial * alpha1 + x_final * alpha2 + 1e-6
            )
        # zero initial guess
        control_matrix = jnp.zeros((horizon + 1, self.num_control_variables)) + 1e-6
        return state_matrix, control_matrix

    def equality_constraints(
        self, states: jnp.array, controls: jnp.array, params: Dict[str, Any]
    ) -> jnp.array:
        """Returns equality constraints.

        Returns h(x) corresponding to the
        equality constraints h(x) = 0.

        Args:
            states: state trajectory,
                (N + 1, nx) array
            controls: control trajectory,
                (N + 1, nu) array
            params: dictionary of parameters of the optimal control problem,
                (key=string, value=Any)

        Returns:
            h_value: value of h(x),
                (_num_equality_constraints, ) array
        """
        horizon = self.horizon

        # initial state is fixed
        initial_state_constraints = states[0] - params["initial_state"]

        # dynamics constraints
        next_states = vmap(
            predict_next_state,
            in_axes=(None, None, None, None, 0, 0),
        )(
            self.dynamics,
            params["discretization_resolution"],
            self.discretization_scheme,
            params,
            states[:horizon],
            controls[:horizon],
        )
        dynamics_constraints = next_states - states[1:]

        # all equality constraints
        dynamics_constraints = dynamics_constraints.flatten()
        constraints = jnp.concatenate(
            [
                initial_state_constraints,
                dynamics_constraints,
            ]
        )
        return constraints

    def step_cost(
        self, state: jnp.array, control: jnp.array, params: Dict[str, Any]
    ) -> float:
        """Returns step cost to minimize.

        Args:
            state: state of the system
                (_num_states, ) array or
            control: control input applied to the system
                (_num_controls, ) array
            params: dictionary of parameters defining the optimal control problem,
                (key=string, value=Any)

        Returns:
            cost: value of the step cost at (state, control),
                (float)
        """
        nu = self.num_control_variables
        reference_state = params["reference_state_trajectory"]
        reference_control = params["reference_control_trajectory"]
        weights_x_ref = params["weights_penalization_reference_state_trajectory"]
        weights_u_norm = params["weights_penalization_control_squared"]

        if self.params["penalize_control_reference"]:
            reference = jnp.concatenate([reference_state, reference_control], axis=-1)
        else:
            reference = jnp.concatenate([reference_state, jnp.zeros(nu)], axis=-1)
        weights_ref = jnp.concatenate([weights_x_ref, weights_u_norm], axis=-1)

        state_control = jnp.concatenate([state, control], axis=-1)
        total_cost = weights_ref * (state_control - reference) ** 2
        total_cost = jnp.sum(total_cost)
        return total_cost

    def final_cost(self, state: jnp.array, params: Dict[str, Any]) -> float:
        """Returns final cost to minimize.

        Args:
            state: state of the system
                (_num_states, ) array or
            params: dictionary of parameters defining the optimal control problem,
                (key=string, value=Any)

        Returns:
            cost: value of the final cost at (state, control),
                (float)
        """
        weights_x_ref = params["weights_penalization_reference_state_trajectory"]
        weights_x_final = params["weights_penalization_final_state"]
        weights_ref = weights_x_ref + weights_x_final
        # state-control cost
        total_cost = weights_ref * (state - params["final_state"]) ** 2
        total_cost = jnp.sum(total_cost)
        return total_cost

    def cost(
        self, states: jnp.array, controls: jnp.array, params: Dict[str, Any]
    ) -> float:
        """Returns total cost to minimize.

        Args:
            states: state trajectory,
                (N + 1, nx) array
            controls: control trajectory,
                (N + 1, nu) array
            params: dictionary of parameters of the optimal control problem,
                (key=string, value=Any)

        Returns:
            cost: value of the step cost at (state, control),
                (float)
        """
        keys_reference = ["reference_state_trajectory", "reference_control_trajectory"]
        keys_weights = [
            "weights_penalization_reference_state_trajectory",
            "weights_penalization_control_squared",
            "weights_penalization_final_state",
            "final_state",
        ]
        step_cost_params = {
            **{key: params[key][:-1] for key in keys_reference},
            **{
                key: jnp.repeat(params[key][None], repeats=self.horizon, axis=0)
                for key in keys_weights
            },
        }
        final_cost_params = {
            **{key: params[key][-1] for key in keys_reference},
            **{key: params[key] for key in keys_weights},
        }
        step_costs = vmap(self.step_cost)(states[:-1], controls[:-1], step_cost_params)
        total_cost = jnp.sum(step_costs) + self.final_cost(
            states[-1], final_cost_params
        )
        return total_cost

    def get_cost_linearized_matrices(
        self,
        states: jnp.array,
        controls: jnp.array,
        params: Dict[str, Any],
    ) -> Tuple[jnp.ndarray]:
        """
        Returns cost terms (QN, qN, Q, q, R, r) corresponding to the cost
        cost = 0.5 x_N^T Q_N x_N + q_N x_N
               + sum_{t=0}^N 0.5 x_t^T Q_t x_t + q_t x_t
               + sum_{t=0}^{N-1} 0.5 u_t^T R_t u_t + r_t u_t.

        Args:
            states: state trajectory,
                (N + 1, nx) array
            controls: control trajectory,
                (N + 1, nu) array
            problem_params: dictionary of parameters of the optimal control problem,
                (key=string, value=Any)

        Returns:
            QN: terminal quadratic state cost matrices,
                (num_states, num_states) array
            qN: terminal vector state cost matrices,
                (num_states) array
            Q: quadratic state cost matrices,
                (horizon, num_states, num_states) array
            q: state cost vectors,
                (horizon, num_states) array
            R: quadratic control cost matrices,
                (horizon, num_controls, num_controls) array
            r: control cost vectors,
                (horizon, num_controls) array
        """
        keys_reference = ["reference_state_trajectory", "reference_control_trajectory"]
        keys_weights = [
            "weights_penalization_reference_state_trajectory",
            "weights_penalization_control_squared",
            "weights_penalization_final_state",
            "final_state",
        ]
        step_cost_params = {
            **{key: params[key][:-1] for key in keys_reference},
            **{
                key: jnp.repeat(params[key][None], repeats=self.horizon, axis=0)
                for key in keys_weights
            },
        }
        final_cost_params = {
            **{key: params[key][-1] for key in keys_reference},
            **{key: params[key] for key in keys_weights},
        }
        Jt_dx, Jt_du = vmap(jax.grad(self.step_cost, argnums=(0, 1)))(
            states[:-1], controls[:-1], step_cost_params
        )
        Ht_dxx = vmap(jax.hessian(self.step_cost, argnums=(0)))(
            states[:-1], controls[:-1], step_cost_params
        )
        Ht_duu = vmap(jax.hessian(self.step_cost, argnums=(1)))(
            states[:-1], controls[:-1], step_cost_params
        )
        JT_dx = jax.grad(self.final_cost, argnums=(0))(states[-1], final_cost_params)
        HT_dxx = jax.hessian(self.final_cost, argnums=(0))(
            states[-1], final_cost_params
        )

        # c(x) = c(xbar) + J(xbar) (x-xbar) + 1/2 (x-xbar)^T H(xbar) (x-xbar)
        #      = constants(xbar) + (J(xbar) - xbar^T H(xbar)) x + 1/2 x^T H(xbar) x
        QN = HT_dxx
        qN = JT_dx - jnp.dot(states[-1], HT_dxx)
        Q = Ht_dxx
        q = vmap(lambda H, J, x: J - jnp.dot(x, H))(Ht_dxx, Jt_dx, states[:-1])
        R = Ht_duu
        r = vmap(lambda H, J, u: J - jnp.dot(u, H))(Ht_duu, Jt_du, controls[:-1])
        return QN, qN, Q, q, R, r

    def get_dynamics_linearized_matrices_with_states_controls(
        self, states: jnp.array, controls: jnp.array, params: Dict[str, Any]
    ) -> Tuple[jnp.ndarray]:
        """
        Returns terms for the initial state and dynamics equality constraints
        states[0] = Cs[0]
        As_next[t]@states[t+1] + As[t]@states[t] + Bs[t]@controls[t] = Fs[t]
        where Cs = [Cs[0], Fs] and t = 0, ..., N-1
        corresponding to the linearization of dynamics constraints
            f_t(x_{t+1}, x_t, u_t) = 0
        around a (states, controls) trajectory.

        Dimensions: (N, nx, nu) = (horizon, num_states, num_controls)

        Args:
            states: state trajectory,
                (N + 1, nx) array
            controls: control trajectory,
                (N + 1, nu) array
            params: dictionary of parameters of the optimal control problem,
                (key=string, value=Any)

        Returns:
            As_next: dynamics matrices multiplying next states
                (N, nx, nx) array
            As: dynamics matrices multiplying states
                (N, nx, nx) array
            Bs: dynamics matrices multiplying controls
                (N, nx, nu) array
            Cs: initial state and dynamics vectors, Cs = (x0, Fs)
                (N+1, nx) array (initial state, dynamics) constraints
        """

        # x+ = f(x,u) ~= f(y,v) + ∇f(y,v) (x-y, u-v)
        # => -I x+ + ∇f(y,v)(x, u) = -(f(y,v) - ∇f(y,v) (y, v))
        def next_state(state_control):
            state = state_control[: self.num_state_variables]
            control = state_control[self.num_state_variables :]
            return predict_next_state(
                self.dynamics,
                params["discretization_resolution"],
                self.discretization_scheme,
                params,
                state,
                control,
            )

        def next_state_and_gradient_dstate_dcontrol(state, control):
            state_control = jnp.concatenate([state, control])
            return value_and_jacfwd(next_state, state_control)

        next_states, next_states_dstate_dcontrol = vmap(
            next_state_and_gradient_dstate_dcontrol
        )(
            states[: self.horizon],
            controls[: self.horizon],
        )
        As = next_states_dstate_dcontrol[:, :, : self.num_state_variables]
        Bs = next_states_dstate_dcontrol[:, :, self.num_state_variables :]
        Cs = jnp.concatenate(
            [
                params["initial_state"][jnp.newaxis],
                -next_states
                + vmap(lambda A, x: A @ x)(As, states[: self.horizon])
                + vmap(lambda A, x: A @ x)(Bs, controls[: self.horizon]),
            ],
            axis=0,
        )
        As_next = jnp.repeat(
            -jnp.eye(self.num_state_variables)[jnp.newaxis],
            repeats=self.horizon,
            axis=0,
        )
        return As_next, As, Bs, Cs

"""Base dynamics class."""

from typing import Any, Dict, List

import jax.numpy as jnp

default_parameters: Dict[str, Any] = {
    "num_states": 2,
    "num_controls": 2,
    "names_states": ["state_1", "state_2"],
    "names_controls": ["control_1", "control_2"],
}
default_state_dot_parameters: Dict[str, Any] = {}


class Dynamics:
    """Base dynamics class."""

    def __init__(self, parameters: Dict[str, Any] = default_parameters):
        """
        Initializes the class.

        Args:
            parameters:  parameters of the class.
                (str, Any) dictionary
        """
        self._params = parameters
        self._state_name_to_state_index_dict = {
            name: i for i, name in enumerate(self.names_states)
        }
        self._control_name_to_control_index_dict = {
            name: i for i, name in enumerate(self.names_controls)
        }

    @property
    def params(self) -> Dict[str, Any]:
        """Returns the parameters of the class."""
        return self._params

    @property
    def num_states(self) -> int:
        """Returns the number of state variables."""
        return self.params["num_states"]

    @property
    def num_controls(self) -> int:
        """Returns the number of control variables."""
        return self.params["num_controls"]

    @property
    def names_states(self) -> List[str]:
        """Returns the names of the state variables."""
        return self.params["names_states"]

    @property
    def names_controls(self) -> List[str]:
        """Returns the names of the control variables."""
        return self.params["names_controls"]

    @property
    def state_name_to_state_index_dict(self) -> Dict[str, Any]:
        """Returns dictionary state_name_to_state_index_dict"""
        return self._state_name_to_state_index_dict

    @property
    def control_name_to_control_index_dict(self) -> Dict[str, Any]:
        """Returns dictionary control_name_to_control_index_dict"""
        return self._control_name_to_control_index_dict

    def get_state_variable_at_state_name(
        self, state: jnp.array, state_name: str
    ) -> float:
        """
        Gets the variable named state_name in the state.

        Args:
            state: state of the system (see names_states)
                (_num_states, ) array
            state_name: name of the state variable (see names_states)
                (str)

        Returns:
            state_variable: variable named state_name in the state vector
                (float)
        """
        variable = state[self.state_name_to_state_index_dict[state_name]]
        return variable

    def get_control_variable_at_control_name(
        self, control: jnp.array, control_name: str
    ) -> float:
        """
        Gets the variable named control_name in the state.

        Args:
            control: control input of the system (see controls_states)
                (_num_controls, ) array
            control_name: name of the control variable (see controls_states)
                (str)

        Returns:
            control_variable: variable named control_name in the control vector
                (float)
        """
        variable = control[self.control_name_to_control_index_dict[control_name]]
        return variable

    def state_dot(
        self,
        state: jnp.array,
        control: jnp.array,
        params: Dict[str, Any] = default_state_dot_parameters,
    ) -> jnp.array:
        """
        Computes the time derivative of the state of the system.

        Returns x_dot = f(x, u) where f describes the dynamics of the system.

        Args:
            state: state of the system (see names_states)
                (_num_states, ) array
            control: control input applied to the system (see names_controls)
                (_num_controls, ) array
            params: parameters of the state_dot function of the dynamics.
                (str, Any) dictionary

        Returns:
            state_dot: time derivative of the state
                (_num_states, ) array
        """
        raise NotImplementedError

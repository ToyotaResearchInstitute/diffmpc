"""Helper functions to load parameter files."""

import os
from typing import Any, Dict

import jax.numpy as jnp
import yaml


def load_params(params_yaml_file_path: str) -> Dict[str, Any]:
    """Loads yaml file containing parameters.

    Args:
        params_yaml_file_path (str): absolute path to the yaml file
            (str)

    Returns:
        params: dictionary containing parameters in the yaml file
            (Dict[str, Any])
    """
    with open(params_yaml_file_path, "r") as file:
        params = yaml.safe_load(file)
    return params


def load_solver_params(params_yaml_filename: str) -> Dict[str, Any]:
    """Loads yaml file containing solver parameters.

    Args:
        params_yaml_filename (str): name path to the yaml file from folder
            containing optimization problems
            (str)

    Returns:
        params: dictionary containing parameters in the yaml file
            (Dict[str, Any])
    """
    # load parameters
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    problem_params_dir = os.path.join(parent_dir, "solvers/params/")
    file_path = os.path.join(problem_params_dir, params_yaml_filename)
    params = load_params(file_path)
    return params


def load_problem_params(params_yaml_filename: str) -> Dict[str, Any]:
    """Loads yaml file containing parameters.

    Args:
        params_yaml_filename (str): name path to the yaml file from folder
            containing optimization problems
            (str)

    Returns:
        params: dictionary containing parameters in the yaml file
            (Dict[str, Any])
    """
    # load parameters
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    problem_params_dir = os.path.join(parent_dir, "problems/params/")
    file_path = os.path.join(problem_params_dir, params_yaml_filename)
    params = load_params(file_path)

    # convert to jax arrays
    entries_to_not_jaxify = [
        "horizon",
        "discretization_scheme",
        "penalize_control_reference",
        "rescale_optimization_variables",
    ]
    for key in params:
        if key not in entries_to_not_jaxify:
            params[key] = jnp.array(params[key])

    # duplicated over time
    params["reference_state_trajectory"] = jnp.repeat(
        params["reference_state_trajectory"][None],
        repeats=params["horizon"] + 1,
        axis=0,
    )
    params["reference_control_trajectory"] = jnp.repeat(
        params["reference_control_trajectory"][None],
        repeats=params["horizon"] + 1,
        axis=0,
    )
    return params


def check_parameters_dictionary_or_raise_errors(params: Dict[str, Any]) -> bool:
    """
    Returns True if params contains all keys in
    DEFAULT_OPTIMAL_CONTROL_PROBLEM_PARAMETERS and raises an error otherwise.
    """
    if params is None:
        raise ValueError("[OptimalControlProblem] params should be a dictionary.")
    for key in load_problem_params("required_parameters.yaml"):
        if key not in params:
            raise KeyError(
                "[OptimalControlProblem]", key, "is not in params dictionary"
            )
    return True

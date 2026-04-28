"""
diverse_functions.py

This module contains various basic functions used in CBF based safety filter

Two basic functions are needed

1. extended class kappa function
2. the barrier function

"""

import numpy as np
import casadi as ca
# from casadi import SX, MX
# from typing import Union

# ------------- Extended class kappa function --------------
def ext_kappa(x, mode: str):
    """
    This is the extended class kappa function.

    Parameters:
    1. x
    2. mode: "linear", "arctan", "tanh"

    Output: y
    """
    if mode == "linear":
        return (1 - 1e-4) * x
    elif mode == "arctan":
        return np.arctan(x)
    elif mode == "tanh":
        return np.tanh(x)
    else:
        raise ValueError("Invalid input! Please use [linear], [arctan] or [tanh]")

def ext_kappa_pmsm(x, mode: str):
    """
    This is the extended class kappa function.

    Parameters:
    1. x
    2. mode: "linear", "arctan", "tanh"

    Output: y
    """
    if mode == "linear":
        return (1 - 5e-3) * x
    elif mode == "arctan":
        return np.arctan(x)
    elif mode == "tanh":
        return np.tanh(x)
    else:
        raise ValueError("Invalid input! Please use [linear], [arctan] or [tanh]")


# ------------- Control barrier function --------------
def cbf_acc_linear(x, mode = "SIM"):
    """
    This is the control barrier function for adaptive cruise control
    used in safety filter optimization problem.

    Here we use a quadratic barrier function.

    Parameters:
    1. x: state
        x[0]: velocity
        x[1]: distance
    2. mode: "SIM" or "NLP", default value is "SIM"
    """
    buffer = 0.5
    margin = x[1] - 1.8 * x[0]

    if mode == "SIM":
        return margin - buffer
    elif mode == "NLP": 
        return margin - buffer
    else:
        raise ValueError("Invalid input! Please use 'NLP' for optimizatoin or 'SIM' for simulation.")

# ------------- Control barrier function --------------
def cbf_pmsm_linear(x, mode = "SIM"):
    """
    This is the control barrier function for adaptive cruise control
    used in safety filter optimization problem.

    Here we use a quadratic barrier function.

    Parameters:
    1. x: state
        x[0]: velocity
        x[1]: distance
    2. mode: "SIM" or "NLP", default value is "SIM"
    """
    buffer = 0
    margin = 7.5 - x[1]

    if mode == "SIM":
        return margin - buffer
    elif mode == "NLP": 
        return margin - buffer
    else:
        raise ValueError("Invalid input! Please use 'NLP' for optimizatoin or 'SIM' for simulation.")


def cbf_acc_quadratic(x, mode = "SIM"):
    """
    This is the control barrier function for adaptive cruise control
    used in safety filter optimization problem.

    Here we use a quadratic barrier function.

    Parameters:
    1. x: state
        x[0]: velocity
        x[1]: distance
    2. mode: "SIM" or "NLP", default value is "SIM"
    """
    buffer = 0.5
    margin = x[1] - 1.8 * x[0]

    if mode == "SIM":
        return buffer ** 2 if margin >= buffer else 2 * buffer * margin - margin ** 2
    elif mode == "NLP": 
        return ca.if_else(margin >= buffer, buffer ** 2, 
                          2 * buffer * margin - margin ** 2)
    else:
        raise ValueError("Invalid input! Please use 'NLP' for optimizatoin or 'SIM' for simulation.")

# ------------- Compute the Lipschitz constant of the barrier function -------------
def estimate_lipschitz_constant(ratio: float, a = 0.5, x_range=(20, 35), y_range=(0, 100), resolution=300):
    """
    Estimate the Lipschitz constant of f(x, y) = (y - a*x) - (y - a*x)^2
    over the rectangular domain x in [x_range[0], x_range[1]],
    y in [y_range[0], y_range[1]].

    Parameters:
        a (float): Parameter in the function.
        x_range (tuple): Range of x values (default: (20, 35)).
        y_range (tuple): Range of y values (default: (0, 100)).
        resolution (int): Number of grid points per axis (default: 300).

    Returns:
        L_est (float): Estimated Lipschitz constant (maximum gradient norm).
    """
    x_min, x_max = x_range
    y_min, y_max = y_range

    def grad_norm(x, y):
        z = y - ratio * x
        df_dx = ratio * (2 * ratio * x - ratio * y) - ratio  # Equivalent to 2a(y - ax) - a
        df_dy = -2 * z + 1
        return np.sqrt(df_dx**2 + df_dy**2)

    x_vals = np.linspace(x_min, x_max, resolution)
    y_vals = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x_vals, y_vals)

    Z = grad_norm(X, Y)
    L_est = np.max(Z)
    return L_est


"""
This script contains several useful functions that helps to formulate
the RLS problem, e.g., matrix generation

It has the following functions:

1) uncertainty set matrix generator
2) uncertainty set vector generator
"""

import numpy as np

def interleave_diag(lb_vec, ub_vec):
    """
    This function generates the matrix H_w to describe the set
    {
    H_w * w <= vec(1) 
    }
    given that the disturbance is box constrained
    and the origin is in the iterior

    Input:
    1) lb_vec: lower bound vector for each entry of w [array]
    2) ub_vec: upper bound vector for each entry of w [array]

    Output:
    H_w [nparray]

    Build the 2n * n matrix
        [ 1/u1  0   …      0 ]
        [ 1/v1  0   …      0 ]
        [ 0  1/u2   …      0 ]
        [ 0  1/v2   …      0 ]
        [ ⋱  ⋱  ⋱      ⋱ ]
        [ 0   0   …    1/un ]
        [ 0   0   …    1/vn ]
    given equal-length 1-D arrays u and v.

    """
    u = np.asarray(ub_vec)
    v = np.asarray(lb_vec)
    if u.shape != v.shape or u.ndim != 1:
        raise ValueError("u and v must be 1-D arrays of the same length")

    n = u.size
    H_w = np.zeros((2 * n, n), dtype=float)

    rows = np.arange(n) * 2          # [0, 2, 4, …, 2n-2]
    H_w[rows,     rows // 2] =  1 / u      # even rows:  1/u
    H_w[rows + 1, rows // 2] =  1 / v      # odd  rows:  1/v
  
    return H_w

def interleave_vec(lb_vec, ub_vec):
    """
    This function generates the vector h_theta to describe the set
    {
    H_theta * theta <= h_theta 
    }
    given that the disturbance is box constrained
    and the origin is in the iterior

    Input:
    1) lb_vec: lower bound vector for each entry of theta [array]
    2) ub_vec: upper bound vector for each entry of theta [array]

    Output:
    h_theta [array]

    Build the 2n * 1 vector
        [ u1 ]
        [-v1 ]
        [u2  ]
        [-v2 ]
        [ ⋱ ]
        [ un ]
        [ -vn ]
    given equal-length 1-D arrays u and v.
    """
    u = np.asarray(ub_vec)
    v = np.asarray(lb_vec)
    if u.shape != v.shape or u.ndim != 1:
        raise ValueError("u and v must be 1-D arrays of the same length")

    n = u.size
    h_theta = np.zeros(2 * n, dtype=float)

    rows = np.arange(n) * 2          # [0, 2, 4, …, 2n-2]
    h_theta[rows] =  u      # even rows:  u
    h_theta[rows + 1] = -v      # odd  rows:  -v
 
    return h_theta


def pmsm_recover(hat_theta, notarray = True):
    """
    This function recovers the PMSM parameters (J, B and phi) from the estimated parameters (theta_1, theta_2, theta_3)

        :param hat_theta: the estimated parameters (theta_1, theta_2, theta_3) from the RLS estimator
        
        :return: the recovered PMSM parameters (J, B and phi)
    """
    phi_est = hat_theta[2]
    J_est = phi_est / hat_theta[1]
    B_est = hat_theta[0] * J_est

    if notarray:
        return {'phif': phi_est, 'J': J_est, 'B': B_est}
    else:
        return np.array([phi_est, J_est, B_est])

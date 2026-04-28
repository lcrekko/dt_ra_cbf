"""
This script is the main file for RLS (recursive least squares) algorithm.

Additional projection operation imposed by set-membership identification (SMI) is also implemented.
"""

import numpy as np
from optimization_utils.metric import ( 
    project_onto_feasible_set, 
    max_l1_deviation_value, 
    polytope_inclusion
)


class RLSProjection:
    """
    This class is the RLS estimator with projection operation after point estimate

    It has 4 functions:
    1. Initialization
    2. Prior point update
    3. SMID set update
    4. projection
    """
    def __init__(self, n_theta, dim_x, f, my_kernel, dt, H_w):
        """
        This is the initialization of the RLS class with a constant covariance

        List of parameters:
        1 n_theta: dimension of the unknown parameter
        2 dim_x: state dimension
        3. f: nominal function in x^+ = f(x) + g(x)u
        4. my_kernel: the kernel function
        5. dt: sampling time
        6. H_w: the matrix describing the disturbance polytope
        """

        # Assign the values and functions
        self.n_theta = n_theta
        self.dim_x = dim_x
        self.kernel = my_kernel
        self.f = f
        self.dt = dt
        self.H_w = H_w
    
    def update_para(self, x_now, x_pre, u_pre, theta_pre, cov_pre, t: int):
        """
        This is the parameter update function

        List of parameters:
        1. x_now: the current state (x_t)
        2. x_pre: the previous state (x_{t-1})
        3. u_pre: the previous input (u_{t-1})
        4. theta_pre: the previous parameter estimate (hat{theta}_{t-1})
        5. t: the running time [integer type]

        Output: dictionary contains the following entries

        1. "para", the updated parameter estimate
        2. "cov", modified covariance matrix
        """
        # compute the kernel value
        phi_t = self.kernel(x_pre, self.dt)

        # compute the gain
        if t == 0:
            K = 0 * cov_pre
        else:
            K = cov_pre @ phi_t @ np.linalg.inv(np.eye(self.dim_x) + phi_t.T @ cov_pre @ phi_t)

        # u_polish = u_pre.flatten()

        # compute the estimated next state
        hat_x_now = self.f(x_pre, u_pre, self.dt) - phi_t.T @ theta_pre

        # compute the parameter increments
        theta_add = K @ (x_now - hat_x_now) # x_now is the measured next state

        # update the parameter, and note the update is with a MINUS sign
        theta_now  = theta_pre - theta_add

        # update the covariance
        cov_post = cov_pre - K @ phi_t.T @ cov_pre

        return {"para": theta_now, "cov": cov_post}
    
    def update_paraset(self, x_now, x_pre, u_pre, H_theta_pre, h_theta_pre, t: int):
        """
        This is the function used to update the set-membership estimate

        List of parameters:
        1. x_now: the current state (x_t)
        2. x_pre: the previous state (x_{t-1})
        3. u_pre: the previous input (u_{t-1})
        4. H_theta_pre: the old parameter matrix (H_theta)
        5. h_theta_pre: the old parameter vector (h_theta)
        6. t: the running time [integer type]
        """
        if t == 0:
            H_theta_new, h_theta_new = H_theta_pre, h_theta_pre
        else:
            # compute the bias
            # u_polish = np.asarray(u_pre).flatten()

            bias_t = x_now - self.f(x_pre, u_pre, self.dt)

            # compute the added rows for the new matrix and vector
            H_theta_add = self.H_w @ self.kernel(x_pre, self.dt).T
            h_theta_add = np.ones(self.H_w.shape[0]) - self.H_w @ bias_t

            # append and get the new matrix
            H_theta_append = np.vstack((H_theta_pre, H_theta_add))
            h_theta_append = np.hstack((h_theta_pre, h_theta_add))

            # update the parameter set
            H_theta_new, h_theta_new = polytope_inclusion(H_theta_append, h_theta_append)

        return H_theta_new, h_theta_new
    
    def posterior(self, H_theta_new, h_theta_new, theta_prior, theta_pre):
        """
        This is a series of post-operations on the prior parameter estimate
        given a SMID estimate

        List of parameters:
        1. H_theta_new: the updated matrix H_theta
        2. h_theta_new: the updated vector h_theta
        3. theta_prior: the prior parameter estimate
        4. theta_pre: the posterior parameter estimate at the previous step

        Output: dictionary contains the following entries

        1. "para", the updated parameter estimate after projection
        2. "delta", 2-norm of the parameter update
        3. "bound", 1-norm of the maximized error bound
        """
        # compute the posterior parameter using projection
        theta_post = project_onto_feasible_set(H_theta_new, h_theta_new, theta_prior)

        # compute the norm of the difference
        norm_diff_para = np.linalg.norm(theta_post - theta_prior, ord=2)

        # compute the error bound using 1-norm (linear programming)
        error_bound = max_l1_deviation_value(H_theta_new, h_theta_new, theta_post)

        return {"para": theta_post, "delta": norm_diff_para, "bound": error_bound}


class RLSUpdate:
    """
    This class is the updated RLS estimator

    It has 4 functions:
    1. Initialization
    2. Prior point update
    3. SMID set update
    4. projection
    """
    def __init__(self, n_theta, dim_x, f, my_kernel, para_model, matrix_model, dt, H_w):
        """
        This is the initialization of the RLS class with a constant covariance

        List of parameters:
        1 n_theta: dimension of the unknown parameter
        2 dim_x: state dimension
        3. f: nominal function in x^+ = f(x) + g(x)u
        4. my_kernel: the kernel function
        5. para_model: the parameter of the model (PMSM)
        6. matrix_model: the matrix model for the system (PMSM)
        7. dt: sampling time
        8. H_w: the matrix describing the disturbance polytope
        """

        # Assign the values and functions
        self.n_theta = n_theta
        self.dim_x = dim_x
        self.kernel = my_kernel
        self.f = f
        self.dt = dt
        self.H_w = H_w
        self.para_model = para_model
        self.matrix_model = matrix_model
    
    def update_para(self, x_now, x_pre, u_pre, theta_pre, cov_pre, t: int):
        """
        This is the parameter update function

        List of parameters:
        1. x_now: the current state (x_t)
        2. x_pre: the previous state (x_{t-1})
        3. u_pre: the previous input (u_{t-1})
        4. theta_pre: the previous parameter estimate (hat{theta}_{t-1})
        5. t: the running time [integer type]

        Output: dictionary contains the following entries

        1. "para", the updated parameter estimate
        2. "cov", modified covariance matrix
        """
        # compute the kernel value
        f = self.f(x_pre, self.para_model, self.dt)
        phi_t = self.kernel(x_pre, self.para_model, self.dt)

        # compute the gain
        if t == 0:
            K = 0 * cov_pre
        else:
            K = cov_pre @ phi_t @ np.linalg.inv(np.eye(self.dim_x) + phi_t.T @ cov_pre @ phi_t)

        # u_polish = u_pre.flatten()

        # compute the estimated next state
        hat_x_now = f - phi_t.T @ theta_pre + self.dt * self.matrix_model['B'] @ u_pre

        # compute the parameter increments
        theta_add = K @ (x_now - hat_x_now) # x_now is the measured next state

        # update the parameter, and note the update is with a MINUS sign
        theta_now  = theta_pre - theta_add

        # update the covariance
        cov_post = cov_pre - K @ phi_t.T @ cov_pre

        return {"para": theta_now, "cov": cov_post}
    
    def update_paraset(self, x_now, x_pre, u_pre, H_theta_pre, h_theta_pre, t: int):
        """
        This is the function used to update the set-membership estimate

        List of parameters:
        1. x_now: the current state (x_t)
        2. x_pre: the previous state (x_{t-1})
        3. u_pre: the previous input (u_{t-1})
        4. H_theta_pre: the old parameter matrix (H_theta)
        5. h_theta_pre: the old parameter vector (h_theta)
        6. t: the running time [integer type]
        """
        if t == 0:
            H_theta_new, h_theta_new = H_theta_pre, h_theta_pre
        else:
            # compute the bias
            # u_polish = np.asarray(u_pre).flatten()

            bias_t = x_now - self.f(x_pre, self.para_model, self.dt) - self.dt * self.matrix_model['B'] @ u_pre

            # compute the added rows for the new matrix and vector
            H_theta_add = self.H_w @ self.kernel(x_pre, self.para_model, self.dt).T
            h_theta_add = np.ones(self.H_w.shape[0]) - self.H_w @ bias_t

            # append and get the new matrix
            H_theta_append = np.vstack((H_theta_pre, H_theta_add))
            h_theta_append = np.hstack((h_theta_pre, h_theta_add))

            # update the parameter set
            H_theta_new, h_theta_new = polytope_inclusion(H_theta_append, h_theta_append)

        return H_theta_new, h_theta_new
    
    def posterior(self, H_theta_new, h_theta_new, theta_prior, theta_pre):
        """
        This is a series of post-operations on the prior parameter estimate
        given a SMID estimate

        List of parameters:
        1. H_theta_new: the updated matrix H_theta
        2. h_theta_new: the updated vector h_theta
        3. theta_prior: the prior parameter estimate
        4. theta_pre: the posterior parameter estimate at the previous step

        Output: dictionary contains the following entries

        1. "para", the updated parameter estimate after projection
        2. "delta", 2-norm of the parameter update
        3. "bound", 1-norm of the maximized error bound
        """
        # compute the posterior parameter using projection
        theta_post = project_onto_feasible_set(H_theta_new, h_theta_new, theta_prior)

        # compute the norm of the difference
        norm_diff_para = np.linalg.norm(theta_post - theta_pre, ord=2)

        # compute the error bound using 1-norm (linear programming)
        error_bound = max_l1_deviation_value(H_theta_new, h_theta_new, theta_post)

        return {"para": theta_post, "delta": norm_diff_para, "bound": error_bound}




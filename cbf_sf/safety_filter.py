"""
safety_filter.py

This module contains
(1) the basic SafetyFilter class that
implements a safety filter that generates a safe input for discrete-time systems;

The filter uses CasADi for symbolic modeling and the optimizer within CasADi.
"""

import casadi as ca
import numpy as np

class AdaptiveSafetyFilter():
    """
    This is the adaptive safety filter class, it has two parts
    1. Initialization and defining the NLP optimization problem
    2. Solve the NLP and return the filtered safe control input
    """
    def __init__(self, dt, num_para, 
                 gamma, L_B, bar_w,
                 x_dim, u_dim,
                 u_min, u_max,
                 dynamics, kernel, cbf, kappa, mode) -> None:
        """
        Initializing the safety filter.

        Parameters:
        1. dt: sampling time
        2. num_para: [int] number of parameters in the dynamics model
        3. gamma: the minimum eigenvalue of the weighting matrix
        4. L_B: Lipschitz constant of the CBF
        5. bar_w: disturbance norm
        6. x_dim: [int] dimension of the state vector
        7. u_dim: [int] dimension of the control input
        8. u_min: input limit (lower bound)
        9. u_max: input limit (upper bound)
        10. dynamics: [function] parametric system model
        11. kernel: [function] kernel in the system model
        12. cbf: [function] control barrier function
        13. kappa: [function] the extended class-kappa function
        14. mode: [str] type of the extended class-kappa function
            - "linear"
            - "arctan"
            - "tanh"
        """
        # --------- passing functions and parameters -----------
        # 1. functions
        self.kernel = kernel
        self.cbf = cbf
        self.kappa = kappa
        self.mode = mode # the mode for kappa function

        # 2. parameters
        self.gamma = gamma
        self.L_B = L_B
        self.bar_w = bar_w
        self.dt = dt

        # --------- function wrapping -----------
        # 1. dynamics function
        x_f = ca.MX.sym("x", x_dim) # type: ignore
        u_f = ca.MX.sym("u", u_dim) # type: ignore
        theta_f = ca.MX.sym("theta", num_para) # type: ignore
        x_plus = dynamics(x_f, u_f, theta_f, dt, "NLP")
        f_opt = ca.Function("f_opt", [x_f, u_f, theta_f], [x_plus])

        # 2. control barrier function
        x_B = ca.MX.sym("x", x_dim) # type: ignore
        B_x = cbf(x_B, "NLP")
        B_opt = ca.Function("B_opt", [x_B], [B_x])

        # 3. composite function
        x_Bf = ca.MX.sym("x", x_dim) # type: ignore
        u_Bf = ca.MX.sym("u", u_dim) # type: ignore
        theta_Bf = ca.MX.sym("theta", num_para) # type: ignore
        Bfx = B_opt(f_opt(x_Bf, u_Bf, theta_Bf))
        self.Bf_opt = ca.Function("Bf_opt", [x_Bf, u_Bf, theta_Bf], [Bfx])

        # Create an Opti instance
        self.opti = ca.Opti()

        # Decision variables: control input
        self.usf = self.opti.variable(u_dim)

        # ---------- Parameters -----------
        # 1. nominal input in the objective function
        self.unom = self.opti.parameter(u_dim)
        # 2. system state
        self.x = self.opti.parameter(x_dim)
        # 3. parameter of the dynamics model
        self.para = self.opti.parameter(num_para)

        # ----------- Objective function ----------
        diff_u = self.usf - self.unom
        self.obj = ca.mtimes([diff_u.T, np.eye(u_dim), diff_u]) # type: ignore

        # ----------- Constraints ------------
        # 1. Basic input constraints
        self.opti.subject_to(self.usf <= u_max)
        self.opti.subject_to(-self.usf <= -u_min)

        # Set the objective
        self.opti.minimize(self.obj)

        # Configure the solver
        opts = {"print_time": False, "ipopt": {"print_level": 0}}
        self.opti.solver("ipopt", opts)
    
    def filter(self, x_val, para_val, unom_val, diff_para, bound_error):
        """
        This is the main filter function that generates the safe input.

        Parameters:
        1. x_val: state value
        2. para_val: parameter value
        3. unom_val: nominal control input
        4. diff_para: increments norm
        5. bound_error: error bound
        """
        # -------- Computation of E_{\theta,t}(x) -----------
        # compute the norm of the kernel
        kernel_val = self.kernel(x_val, self.dt)
        kernel_norm = np.linalg.norm(kernel_val, ord = 2)
        # compute the first term
        E_theta_1 = (self.L_B * kernel_norm + diff_para / self.gamma) * bound_error
        # compute the second term
        E_theta_2 = (diff_para ** 2) / (2 * self.gamma)
        # obtain the final E_{\theta,t}(x)
        E_theta = E_theta_1 + E_theta_2

        # -------- Computation of the class-kappa term \alpha(B(x) - error) ----------
        B_error = self.cbf(x_val) - bound_error ** 2 / (2 * self.gamma)
        alpha_B = self.kappa(B_error, self.mode)

        # -------- Computation of the joint comparison term
        joint_comparison = self.cbf(x_val) + self.L_B * self.bar_w + E_theta - alpha_B

        # -------- Add the barrier function constraint --------
        self.opti.subject_to(joint_comparison <= self.Bf_opt(self.x, self.usf, self.para))

        # -------- Assign the value to parameters --------
        # Set the nominal input to the objective
        self.opti.set_value(self.unom, unom_val)
        
        # Set the initial state parameter value
        self.opti.set_value(self.x, x_val)

        # Set the parameter value for the dynamics function
        self.opti.set_value(self.para, para_val)

        # Solve the optimization problem
        sol = self.opti.solve()

        # Extract the safe control input
        usf = sol.value(self.usf)
        return np.atleast_1d(usf), E_theta


         
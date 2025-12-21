"""
mpc_controller.py

This module contains
(1) the basic open-loop MPCController class that
implements a Model Predictive Controller (MPC) for discrete-time systems;

(2) a specific myopic robust adaptive MPC controller for adaptive cruise control 

The controller uses CasADi for symbolic modeling and the optimizer within CasADi.
"""

import casadi as ca
import numpy as np

class MPCController:
    """
    This is the MPC controller class, it has two parts
    1. Initialization and defining the NLP optimization problem
    2. Solve the NLP but only return the first input for closed-loop simulation and analysis
    """
    def __init__(self, horizon, d_t,
                 x_dim, u_dim,
                 x_ref, u_ref,
                 umin, umax,
                 Q, R, P,
                 dynamics, num_para):
        """
        Initialize the MPC controller.

        Parameters:
            horizon: int, prediction horizon
            d_t: the sampling time
            x_dim: int, dimension of the state vector
            u_dim: int, dimension of the control input
            x_ref: state reference point
            u_ref: input reference point
            umin: numpy array, lower bound for control inputs
            umax: numpy array, upper bound for control inputs
            Q: numpy array, state weighting matrix
            R: numpy array, input weighting matrix
            P: numpy array, final state weighting matrix
            dynamics: function, discrete-time dynamics function
            num_para: int, number of parameters in the dynamics function
        """
        self.N = horizon
        self.d_t = d_t
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.x_ref = x_ref
        self.u_ref = u_ref
        self.umin = umin
        self.umax = umax
        #self.Q = Q
        #self.R = R
        #self.P = P
        self.dynamics = dynamics

        # Create an Opti instance
        self.opti = ca.Opti()

        # Decision variables: states over the horizon and control inputs
        self.X = self.opti.variable(x_dim, self.N+1)
        self.U = self.opti.variable(u_dim, self.N)

        # Parameter for the initial state
        self.X0 = self.opti.parameter(x_dim)
        # Parameter for the dynamics function
        self.para = self.opti.parameter(num_para)
        # Initial condition constraint
        self.opti.subject_to(self.X[:,0] == self.X0)

        # Build the cost function and constraints over the horizon
        self.obj = 0  # Initialize objective function
        for k in range(self.N):
            # Stage cost
            diff_x = self.X[:, k] - x_ref
            diff_u = self.U[:, k] - u_ref
            self.obj += ca.mtimes([diff_x.T, Q, diff_x]) + ca.mtimes([diff_u.T, R, diff_u])

            # Dynamics constraint: x_{k+1} = f(x_k, u_k, para_k)
            x_next = self.dynamics(self.X[:, k], self.U[:, k], self.para, self.d_t, "NLP")
            self.opti.subject_to(self.X[:, k+1] == x_next)

            # Input constraints (elementwise)
            self.opti.subject_to(self.umin <= self.U[:, k])
            self.opti.subject_to(self.U[:, k] <= self.umax)

        # Terminal cost (the weight is set the same as the stage cost)
        diff_x_last = self.X[:, -1] - x_ref
        self.obj += ca.mtimes([diff_x_last.T, P, diff_x_last])

        # Set the objective
        self.opti.minimize(self.obj)

        # Configure the solver
        opts = {"print_time": False, "ipopt": {"print_level": 0}}
        self.opti.solver("ipopt", opts)


    def solve_closed(self, x0_val, para):
        """
        Solve the MPC problem for a given initial state
        and return only the first input for closed-loop integration

        Parameters:
            x0_val: numpy array, initial state value.
            para: the parameter vector for the dynamics function.

        Returns:
            u_0: numpy array, the first control action.
        """

        # Set the initial state parameter value
        self.opti.set_value(self.X0, x0_val)

        # Set the parameter value for the dynamics function
        self.opti.set_value(self.para, para)

        # Solve the optimization problem
        sol = self.opti.solve()
        # Extract the first control input
        u_0 = sol.value(self.U[:, 0])
        return np.atleast_1d(u_0)


class RAMPCACC:
    """
    This is the myopic robust adaptive MPC controller class with adaptive constraint 
    tightening for adaptive cruise control (ACC), it has two parts
    1. Initialization and defining the NLP optimization problem
    2. Solve the NLP but only return the first input for closed-loop simulation and analysis
    """
    def __init__(self, d_t, M, bar_w,
                 x_ref, u_ref,
                 umin, umax,
                 Q, R, P,
                 dynamics, num_para):
        """
        Initialize the MPC controller.

        Parameters:
            d_t: the sampling time
            M: mass of the vehicle
            bar_w: the disturbance bound
            x_ref: state reference point
            u_ref: input reference point
            umin: numpy array, lower bound for traction force
            umax: numpy array, upper bound for traction force
            Q: numpy array, state weighting matrix
            R: numpy array, input weighting matrix
            P: numpy array, final state weighting matrix
            dynamics: function, discrete-time dynamics function
            num_para: int, number of parameters in the dynamics function
        """
        self.d_t = d_t
        self.x_ref = x_ref
        self.u_ref = u_ref
        self.umin = umin
        self.umax = umax
        #self.Q = Q
        #self.R = R
        #self.P = P
        self.dynamics = dynamics

        # Create an Opti instance
        self.opti = ca.Opti()

        # Decision variables: states over the horizon and control inputs
        self.X = self.opti.variable(2, 2) # 2 states, 1 step prediction
        self.U = self.opti.variable(1, 1) # 1 input, 1 step prediction

        # ---------- Assignable parameters ------------- 
        # Parameter for the initial state
        self.X0 = self.opti.parameter(2)
        # Parameter for the dynamics function
        self.para = self.opti.parameter(num_para)
        # Parameter for the adaptive tightening term
        self.ebound = self.opti.parameter(1) # disturbance bound and parameter error bound 
        
        # Initial condition constraint
        self.opti.subject_to(self.X[:,0] == self.X0)

        # Build the cost function and constraints for one-step prediction
        self.obj = 0  # Initialize objective function
        # Stage cost
        diff_x = self.X[:, 0] - x_ref
        diff_u = self.U[:, 0] - u_ref
        self.obj += ca.mtimes([diff_x.T, Q, diff_x]) + ca.mtimes([diff_u.T, R, diff_u])

        # Dynamics constraint: x_{k+1} = f(x_k, u_k, para_k)
        x_next = self.dynamics(self.X[:, 0], self.U[:, 0], self.para, self.d_t, "NLP")
        self.opti.subject_to(self.X[:, 1] == x_next)

        # Input constraints (elementwise box)
        self.opti.subject_to(self.umin <= self.U[:, 0])
        self.opti.subject_to(self.U[:, 0] <= self.umax)

        # Safety constraints
        tight = self.d_t * (np.sqrt(2) * bar_w + ca.sqrt(1 + ca.power(self.X[0, 0], 4) / (M**2)) * self.ebound)
        self.opti.subject_to(self.X[1, 1] - 1.8 * self.X[0, 1] >= tight)

        # Terminal cost (the weight is set the same as the stage cost)
        diff_x_last = self.X[:, -1] - x_ref
        self.obj += ca.mtimes([diff_x_last.T, P, diff_x_last])

        # Set the objective
        self.opti.minimize(self.obj)

        # Configure the solver
        opts = {"print_time": False, "ipopt": {"print_level": 0}}
        self.opti.solver("ipopt", opts)


    def solve_closed(self, x0_val, para, ebound):
        """
        Solve the MPC problem for a given initial state
        and return only the first input for closed-loop integration

        Parameters:
            x0_val: numpy array, initial state value.
            para: the parameter vector for the dynamics function.

        Returns:
            u_0: numpy array, the first control action.
        """

        # Set the initial state parameter value
        self.opti.set_value(self.X0, x0_val)

        # Set the parameter value for the dynamics function
        self.opti.set_value(self.para, para)

        # Set the error bound to be the instantaneous error bound
        self.opti.set_value(self.ebound, ebound)

        # Solve the optimization problem
        sol = self.opti.solve()
        # Extract the first control input
        u_0 = sol.value(self.U[:, 0])
        return np.atleast_1d(u_0)

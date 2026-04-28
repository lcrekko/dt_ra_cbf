import numpy as np
# import control as ctrl
from nmpc.diverse_functions import pmsm_drift, pmsm_kernel, pmsm_gu, pmsm_gw

dt = 1e-4 # sampling interval [s]

# PMSM parameters
pmsm_params = {
    'R': 0.8, # resistance [Ohm]
    'L': 2.9e-3, # d-axis inductance [H]
    'T_load': 2.25, # load torque [N*m]
    'phif': 0.081, # flux linkage [Wb]
    'num_pole': 4, # number of pole pairs
    'J': 2.35e-4, # inertia [kg*m^2]
    'B': 7.4e-4, # friction coefficient [N*m*s/rad]
    'iq_lim': 4, # maximum q-axis current [A]
    'u_lim': 220 # maximum voltage [V]
}

linear_model_params = {
    'A': np.array([[-pmsm_params['B'] / pmsm_params['J'], pmsm_params['phif'] * pmsm_params['num_pole'] / pmsm_params['J']],
                   [-pmsm_params['phif'] * pmsm_params['num_pole'] / pmsm_params['L'], -pmsm_params['R'] / pmsm_params['L']]]),
    'B': np.array([[0], [1 / pmsm_params['L']]])
}

est_params = {
    'phif': pmsm_params['phif'],
    'B': pmsm_params['B'],
    'R': pmsm_params['R']
}

x_t = np.array([10, 70])

u_t = np.array([1])

# ------ Compute different terms of the dynamics ------
f_t = pmsm_drift(x_t, pmsm_params, dt=dt)
phi_t = pmsm_kernel(x_t, pmsm_params, dt=dt)
g_u_t = pmsm_gu(x_t, pmsm_params, dt=dt)
g_w_t = pmsm_gw(x_t, pmsm_params, dt=dt)

# print(g_u_t.shape)

# w_t = np.array([load_disturbance[t], id_disturbance[t]]) # combined disturbance
para_star = np.array([est_params['phif'], est_params['B'], est_params['R']])

# x_next = f_t - phi_t.T @ para_nom + g_u_t * u_t + g_w_t * w_t
# d_w = g_w_t * np.array([load_disturbance[t], id_disturbance[t]])
bias = np.array([-dt * pmsm_params['T_load'] / pmsm_params['J'], 0]) # bias term due to load torque
x_next_1  = (np.eye(2) + dt * linear_model_params['A']) @ x_t + dt * linear_model_params['B'] @ u_t + bias
x_next_2 = f_t - phi_t.T @ para_star + dt * linear_model_params['B'] @ u_t

print("x_next_1:", x_next_1)
print("x_next_2:", x_next_2)
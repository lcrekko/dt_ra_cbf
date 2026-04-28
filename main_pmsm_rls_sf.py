"""
This is the main script used to generate the simulation results
for the PMSM application of the robust adaptive safe control approach.

The simualtion is done using Monte Carlo with different disturbance realizations,
and the number of realizations can be tuned depending on the time budget.

The code is mainly build on test_pid_pmsm.py, for further improvement
and advanced settings, please refer to test_pid_pmsm.py and play around with the code.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
# import control as ctrl
from cbf_sf.safety_filter import AdaptiveSafetyFilterLinear
# from cbf_sf.diverse_functions import ext_kappa_pmsm, cbf_pmsm_linear
from nmpc.diverse_functions import pmsm_drift, pmsm_kernel, pmsm_kernel_opt
from nmpc.controller import PPD_PMSM_Controller
from rls.rls_main import RLSUpdate
from rls.rls_utils import interleave_vec, interleave_diag
from simulation_utils.generators import generate_step_reference, generate_noise_simulations, rpm_to_rads
from optimization_utils.metric import max_l1_deviation_value
from plotters.utils import plotter_kernel
plt.rcParams.update({
    "text.usetex": True,                  # Use LaTeX for text rendering
    "font.family": "serif",               # Use a serif font
    "font.serif": ["Computer Modern Roman"] # Set the font
})

## ---------- System Initialization ------------
# Time parameters
dt = 1e-3 # sampling interval [s]
T  = 4 # total time horizon [s]
N_t = int(T / dt) # total time steps
T_initial = 0 # time before disturbances start [s]

# state and input dimensions
x_dim = 2
u_dim = 1

# PMSM parameters
pmsm_params = {
    'R': 0.8, # resistance [Ohm]
    'L': 2.9e-3, # d-axis inductance [H]
    'T_load': 0, # load torque [N*m]
    'phif': 0.081, # flux linkage [Wb]
    'num_pole': 4, # number of pole pairs
    'J': 2.35e-4, # inertia [kg*m^2]
    'B': 7.4e-4, # friction coefficient [N*m*s/rad]
    'iq_lim': 2.85, # maximum q-axis current [A]
    'u_lim': 220 # maximum voltage [V]
}

# The range of maximum voltage [V]
u_lim = pmsm_params['u_lim']
# ----------------------------------------------

## ---------- Disturbance and Reference Initialization ------------
# disturbance parameters
load_max = 0.0225 # maximum load disturbance (torque) [N*m]
id_max = 0.04

# speed bound
omega_max = rpm_to_rads(4000)

# create generators with a specific seed (e.g., 42)
rng_load = np.random.default_rng(seed=60)
rng_id = np.random.default_rng(seed=6)

# number of disturbance realizations (for Monte Carlo simulation)
N_sim = 100

# generate disturbance trajectories
load_disturbance = 2 * generate_noise_simulations(N_t, N_sim, rng_load, t_initial=int(0/dt), noise_scale=load_max) - load_max # scale to [-load_max, load_max]
id_disturbance = 2 * generate_noise_simulations(N_t, N_sim, rng_id, t_initial=int(0/dt), noise_scale=id_max) - id_max # scale to [-id_max, id_max]
# generate reference trajectory
t_w_vec = [int(0/dt), int(0.5/dt), int(1/dt), int(2/dt), int(3/dt)]
omega_ref_vec = [rpm_to_rads(500), rpm_to_rads(-500), rpm_to_rads(3000), rpm_to_rads(-500), rpm_to_rads(0)] # reference angular velocity in rad/s
omega_ref_trajectory = generate_step_reference(N_t, t_w_vec, omega_ref_vec)

# disturbance limit
w_lim = dt * np.array([load_max / pmsm_params['J'], id_max * pmsm_params['num_pole'] * omega_max])

# disturbance matrix
H_w = interleave_diag(-w_lim, w_lim)


## --------- PID Controller Initialization ------------
linear_model_params = {
    'A': np.array([[-pmsm_params['B'] / pmsm_params['J'], pmsm_params['phif'] * pmsm_params['num_pole'] / pmsm_params['J']],
                   [-pmsm_params['phif'] * pmsm_params['num_pole'] / pmsm_params['L'], -pmsm_params['R'] / pmsm_params['L']]]),
    'B': np.array([[0], [1 / pmsm_params['L']]])
}

# load the nominal PID controller
controller = PPD_PMSM_Controller(linear_model_params['A'], linear_model_params['B'], 
                                 desired_poles=[-25 + 20j, -25 - 20j], int_omega_error_gain=1e-6)

## --------- RLS + SMID Estimator Initialization ------------
# Extract nominal estimated parameters from the PMSM parameters
est_params = {
    'phif': pmsm_params['phif'],
    'B': pmsm_params['B'],
    'R': pmsm_params['R']
}

# define parameter bounds
pmsm_bounds = {
    'bound_phif': (0.065, 0.095), # bounds for the flux linkage [Wb]
    'bound_B': (7e-4, 8e-4), # bounds for the friction coefficient [N*m*s/rad]
    'bound_R': (0.6, 0.9) # bounds for the resistance [Ohm]
}

num_para = len(est_params)
para_0 = np.array([pmsm_bounds[f'bound_{key}'][1] for key in est_params.keys()])
para_star = np.array([est_params[key] for key in est_params.keys()])

# parameter bound
LB_para = [pmsm_bounds[f'bound_{key}'][0] for key in est_params.keys()]
UB_para = [pmsm_bounds[f'bound_{key}'][1] for key in est_params.keys()]

# parameter matrix
H_para = interleave_diag(-np.ones(num_para), np.ones(num_para))
h_para = interleave_vec(LB_para, UB_para)

# initial error bound
bound_0 = max_l1_deviation_value(H_para, h_para, para_0)

# initial covaraince matrix
var_para = np.array([1, 1e-4, 10])
cov = np.diag(var_para)
# define the estimator
my_rls = RLSUpdate(num_para, x_dim, pmsm_drift, pmsm_kernel, pmsm_params, linear_model_params, dt, H_w)
# ----------------------------------------------------------

## --------- Safety Filter Initialization ------------
# the minimum eigenvalue
gamma = 1e7
I_max = pmsm_params['iq_lim'] - 0.05
alpha = 1 - 5e-3
# compute the Lipschitz constant (linear barrier function)
L_B = 1
# disturbance bound (2-norm)
bar_w  = w_lim[1]
print(f"disturbance bound: {bar_w}")

# define the safety filter
my_sf = AdaptiveSafetyFilterLinear(
    dt,
    gamma, L_B, bar_w,
    x_dim, u_dim,
    -u_lim, u_lim,
    pmsm_params, pmsm_kernel_opt, I_max, alpha
)
# ------------------------------------------------------

# ## ---------- Simulation Initialization ------------
# # initial state
# X0 = np.array([0.0, 0.0])   # [angular velocity, q-axis current]

# T_turning = int(-1/dt) # time to start filtering, set as -1s (always filter from the beginning)
# T_project = int(5/dt) # time to stop projection, set as 5s (so no stop at all)
# """
# n: nominal aPID control (no safety filter)
# r: PID + robust CBF control (no adaptive)
# nr: aPID + robust CBF control (CBF does not adaptive)
# ar: adaptive PID + robust adaptive CBF control (proposed approach)
# """

# # Initialize the output trajectory for nominal aPID control
# x_n_traj = np.zeros((N_sim, N_t + 1, x_dim)) # state
# u_n_traj = np.zeros((N_sim, N_t, u_dim)) # input
# para_n_traj = np.zeros((N_sim, N_t + 1, num_para)) # parameter
# bound_n_traj = np.zeros((N_sim, N_t + 1)) # parameter error bound

# # Initialize the output trajectory for PID + robust CBF control (no adaptive)
# x_r_traj = np.zeros((N_sim, N_t + 1, x_dim)) # state
# u_r_traj = np.zeros((N_sim, N_t, u_dim)) # input

# # Note: no adaptive, no need to record parameter estimation trajectory and error bound trajectory for the non-adaptive robust CBF control

# # Initialize the output trajectory for aPID+rCBF control (CBF does not adaptive)
# x_nr_traj = np.zeros((N_sim, N_t + 1, x_dim)) # state
# u_nr_traj = np.zeros((N_sim, N_t, u_dim)) # input
# para_nr_traj = np.zeros((N_sim, N_t + 1, num_para)) # parameter
# bound_nr_traj = np.zeros((N_sim, N_t + 1)) # parameter error bound

# # Initialize the output trajectory for aPID + robust adaptive CBF control (proposed approach)
# x_ar_traj = np.zeros((N_sim, N_t + 1, x_dim)) # state
# u_ar_traj = np.zeros((N_sim, N_t, u_dim)) # input
# para_ar_traj = np.zeros((N_sim, N_t + 1, num_para)) # parameter
# bound_ar_traj = np.zeros((N_sim, N_t + 1)) # parameter error bound
# # ---------------------------------------------------------------------

# ## -------------- Main Simulation Loops --------------
# # Simulation main loops
# for i in range(N_sim):
# # outer loop for different realizations
#     # Initialize the state
#     x_n_traj[i, 0, :] = X0
#     x_r_traj[i, 0, :] = X0
#     x_nr_traj[i, 0, :] = X0
#     x_ar_traj[i, 0, :] = X0
#     # initialize estimation for aPID control
#     para_n_traj[i, 0, :] = para_0
#     # initialize estimation for aPID+rCBF control
#     para_nr_traj[i, 0, :] = para_0
#     # initialize estimation for robust adaptive MPC
#     para_ar_traj[i, 0, :] = para_0
#     bound_ar_traj[i, 0] = bound_0

#     # Initialize the first parameter difference for adaptive safe control
#     diff_para_ar = 0

#     # Initialize the integral error for the PID controller
#     int_error_n = 0
#     int_error_r = 0
#     int_error_nr = 0
#     int_error_ar = 0

#     # initial covaraince matrix
#     var_para = np.array([1, 1e-4, 10])
#     cov_n = np.diag(var_para)
#     cov_nr = np.diag(var_para)
#     cov_ar = np.diag(var_para)

#     # sample a disturbance realization
#     w_load = load_disturbance[i, :]
#     w_id = id_disturbance[i, :]

#     # reset the parameter matrix
#     H_para_n = interleave_diag(-np.ones(num_para), np.ones(num_para)) # for adaptive safe control
#     h_para_n = interleave_vec(LB_para, UB_para) # for adaptive safe control

#     H_para_nr = interleave_diag(-np.ones(num_para), np.ones(num_para)) # for adaptive safe control
#     h_para_nr = interleave_vec(LB_para, UB_para) # for adaptive safe control

#     H_para_ar = interleave_diag(-np.ones(num_para), np.ones(num_para)) # for adaptive safe control
#     h_para_ar = interleave_vec(LB_para, UB_para) # for adaptive safe control

#     for t in range(N_t):
#         # extract the current state
#         x_n = x_n_traj[i, t, :]
#         x_r = x_r_traj[i, t, :]
#         x_nr = x_nr_traj[i, t, :]
#         x_ar = x_ar_traj[i, t, :]

#         # extract the current reference
#         omega_ref = omega_ref_trajectory[t]

#         # ------------ Control Input Computation ------------
#         # compute the aPID control input (nominal, no safety filter)
#         u_n, int_error_n = controller.compute_control(omega_ref, x_n, 
#                                                       pmsm_params, para_n_traj[i, t, :],
#                                                       int_error_n)
#         # compute the PID + robust CBF control input (no adaptive)
#         u_r, int_error_r = controller.compute_control(omega_ref, x_r,
#                                                       pmsm_params, para_0,
#                                                       int_error_r)
#         # compute the aPID + robust CBF control input (adaptive only for PID, but CBF does not adaptive)
#         u_nr, int_error_nr = controller.compute_control(omega_ref, x_nr,
#                                                         pmsm_params, para_nr_traj[i, t, :],
#                                                         int_error_nr)
#         # compute the aPID + robust adaptive CBF control input (proposed approach)
#         u_ar, int_error_ar = controller.compute_control(omega_ref, x_ar,
#                                                         pmsm_params, para_ar_traj[i, t, :],
#                                                         int_error_ar)
#         if t > T_turning:
#             u_n_safe = u_n # no safety filter for the nominal aPID control
#             u_r_safe = my_sf.filter(x_r, bound_0, 0, para_0, u_r)
#             u_nr_safe = my_sf.filter(x_nr, bound_0, 0, para_0, u_nr)
#             u_ar_safe = my_sf.filter(x_ar, bound_ar_traj[i, t], diff_para_ar, para_ar_traj[i, t, :], u_ar)
#         else:
#             u_n_safe = u_n
#             u_r_safe = u_r
#             u_nr_safe = u_nr
#             u_ar_safe = u_ar
        
#         # Store control input
#         u_n_traj[i, t, :] = u_n_safe
#         u_r_traj[i, t, :] = u_r_safe
#         u_nr_traj[i, t, :] = u_nr_safe
#         u_ar_traj[i, t, :] = u_ar_safe

#         # Compute the disturbance coupling term for the next state update
#         g_w_n = pmsm_gw(x_n, pmsm_params, dt=dt)
#         g_w_r = pmsm_gw(x_r, pmsm_params, dt=dt)
#         g_w_nr = pmsm_gw(x_nr, pmsm_params, dt=dt)
#         g_w_ar = pmsm_gw(x_ar, pmsm_params, dt=dt)

#         # Form the disturbance term
#         w_t = np.array([w_load[t], w_id[t]])

#         # ------------ State Update ------------
#         # update the state for nominal aPID control
#         x_n_traj[i, t+1, :]  = (np.eye(2) + dt * linear_model_params['A']) @ x_n + dt * linear_model_params['B'] @ u_n_safe + g_w_n * w_t
#         # update the state for PID + robust CBF control (no adaptive)
#         x_r_traj[i, t+1, :]  = (np.eye(2) + dt * linear_model_params['A']) @ x_r + dt * linear_model_params['B'] @ u_r_safe + g_w_r * w_t
#         # update the state for PID + robust CBF control (adaptive)
#         x_nr_traj[i, t+1, :]  = (np.eye(2) + dt * linear_model_params['A']) @ x_nr + dt * linear_model_params['B'] @ u_nr_safe + g_w_nr * w_t
#         # update the state for the proposed approach
#         x_ar_traj[i, t+1, :]  = (np.eye(2) + dt * linear_model_params['A']) @ x_ar + dt * linear_model_params['B'] @ u_ar_safe + g_w_ar * w_t

#         # ------------ Parameter Estimation Update ------------
#         # update the parameter estimation for aPID control
#         info_n = my_rls.update_para(x_n_traj[i, t+1, :], x_n_traj[i, t, :], u_n_traj[i, t, :], 
#                                     para_n_traj[i, t, :], cov_n, t+1)
#         info_nr = my_rls.update_para(x_nr_traj[i, t+1, :], x_nr_traj[i, t, :], u_nr_traj[i, t, :], 
#                                      para_nr_traj[i, t, :], cov_nr, t+1)
#         info_ar = my_rls.update_para(x_ar_traj[i, t+1, :], x_ar_traj[i, t, :], u_ar_traj[i, t, :], 
#                                      para_ar_traj[i, t, :], cov_ar, t+1)
        
#         # SMID projection for prior update
#         if t < T_project:
#             # for aPID control
#             H_para_n_p, h_para_n_p = my_rls.update_paraset(x_n_traj[i, t+1, :], x_n_traj[i, t, :], u_n_traj[i, t, :],
#                                                            H_para_n, h_para_n, t+1)
#             info_n_p = my_rls.posterior(H_para_n, h_para_n, info_n["para"], para_n_traj[i, t, :])
#             para_n_traj[i, t+1, :] = info_n_p["para"]
#             # for aPID+rCBF control
#             H_para_nr_p, h_para_nr_p = my_rls.update_paraset(x_nr_traj[i, t+1, :], x_nr_traj[i, t, :], u_nr_traj[i, t, :],
#                                                               H_para_nr, h_para_nr, t+1)
#             info_nr_p = my_rls.posterior(H_para_nr, h_para_nr, info_nr["para"], para_nr_traj[i, t, :])
#             para_nr_traj[i, t+1, :] = info_nr_p["para"]
#             # for the proposed approach
#             H_para_ar_p, h_para_ar_p = my_rls.update_paraset(x_ar_traj[i, t+1, :], x_ar_traj[i, t, :], u_ar_traj[i, t, :],
#                                                               H_para_ar, h_para_ar, t+1)
#             info_ar_p = my_rls.posterior(H_para_ar, h_para_ar, info_ar["para"], para_ar_traj[i, t, :])
#             para_ar_traj[i, t+1, :] = info_ar_p["para"]
#         else:
#             para_n_traj[i, t+1, :] = info_n["para"]
#             para_nr_traj[i, t+1, :] = info_nr["para"]
#             para_ar_traj[i, t+1, :] = info_ar["para"]
        
#         # For the proposed approach, update for terms for safe filter
#         diff_para_ar = np.linalg.norm(para_ar_traj[i, t+1, :] - para_ar_traj[i, t, :], ord=2)
#         # bound_ar_traj[i, t+1] = np.random.uniform(1, 3)*np.linalg.norm(para_ar_traj[i, t+1, :] - para_star, ord=2)
#         bound_ar_traj[i, t + 1] = info_ar_p["bound"]
        
#         # update the covariance matrix
#         cov_n = info_n["cov"]
#         cov_nr = info_nr["cov"]
#         cov_ar = info_ar["cov"]
    
#     print(f"Simulation {i+1}/{N_sim} completed.")

# # save data for reuse
# np.savez('data_pmsm_n.npz', state=x_n_traj, input=u_n_traj, para=para_n_traj)
# np.savez('data_pmsm_r.npz', state=x_r_traj, input=u_r_traj, para=para_0) # no parameter estimation for the non-adaptive robust CBF control, so just save the nominal parameter
# np.savez('data_pmsm_nr.npz', state=x_nr_traj, input=u_nr_traj, para=para_nr_traj)
# np.savez('data_pmsm_ar.npz', state=x_ar_traj, input=u_ar_traj, para=para_ar_traj, bound=bound_ar_traj)

# load data (for fast plotting and editing plots for aesthetics)
data_pmsm_n = np.load("data_pmsm_n.npz")
data_pmsm_r = np.load("data_pmsm_r.npz")
data_pmsm_nr = np.load("data_pmsm_nr.npz")
data_pmsm_ar = np.load("data_pmsm_ar.npz")

x_n_traj = data_pmsm_n["state"]
x_r_traj = data_pmsm_r["state"]
x_nr_traj = data_pmsm_nr["state"]
x_ar_traj = data_pmsm_ar["state"]

u_n_traj = data_pmsm_n["input"]
u_r_traj = data_pmsm_r["input"]
u_nr_traj = data_pmsm_nr["input"]
u_ar_traj = data_pmsm_ar["input"]

para_n_traj = data_pmsm_n["para"]
para_r_traj = para_0 # no parameter estimation for the non-adaptive robust
para_nr_traj = data_pmsm_nr["para"]
para_ar_traj = data_pmsm_ar["para"]

bound_ar_traj = data_pmsm_ar["bound"]

## ---------- Plotting Initialization ------------
# define time axis for plotting
time = dt * np.arange(0, N_t + 1) # from 0 to T with N_t+1 points (including the initial time)
time_s = dt * np.arange(0, N_t) # for plotting the control input and disturbances

# Plotting parameters
my_linewidth = 1.2
mygreen = (0.4157, 0.7490, 0.6588)
myblue = (0.4549, 0.4353, 0.6941)
myred = (0.8980, 0.5882, 0.3529)
mypurple = (0.8000, 0.7600, 0.4200)
mydarkblue = (0.4235, 0.7765, 0.8471)
mydarkblue_cop = (0.7725*0.5, 0.8000*0.5, 0.8588*0.5)
myleaveyellow = (0.9333, 0.4588, 0.3922)
myleaveyellow_cop = (0.9098*0.5, 0.8118*0.5, 0.5725*0.5)
mysolfyellow = (0.96, 0.84, 0.45)
mysoftyellow_cop = (0.96*0.5, 0.84*0.5, 0.45*0.5)
legend_1 = "aPID-raCBF"
legend_2 = "aPID-rCBF"
legend_3 = "aPID"
legend_4 = "PID-rCBF"
linestyle_1 = '-'
linestyle_2 = '-'
linestyle_3 = '-'
linestyle_4 = '-'
# ------------------------------------------------------

## ----------- Plotting -----------
# Plot the state and input trajectories
fig_sys, axes_sys = plt.subplots(3, 1, figsize=(4.5, 4))

# Plot 1: Angular velocity
plotter_kernel(axes_sys[0], time, x_ar_traj[:, :, 0],
               legend_1, my_linewidth, mygreen, linestyle_1)
plotter_kernel(axes_sys[0], time, x_nr_traj[:, :, 0],
               legend_2, my_linewidth, myblue, linestyle_2)
plotter_kernel(axes_sys[0], time, x_n_traj[:, :, 0],
               legend_3, my_linewidth, myred, linestyle_3)
plotter_kernel(axes_sys[0], time, x_r_traj[:, :, 0],
               legend_4, my_linewidth, mypurple, linestyle_4)
axes_sys[0].plot(time_s, omega_ref_trajectory, linestyle='--', color='black', linewidth=my_linewidth)
axes_sys[0].legend(loc='upper center',
    bbox_to_anchor=(0.5, 1.75),  # position relative to the whole figure
    ncol=2,                        # all items in one row
    frameon=True)
axes_sys[0].set_facecolor((0.95, 0.95, 0.95))
axes_sys[0].set_ylabel(r'$\omega$ [rad/s]')
axes_sys[0].grid(True, linestyle='--', color='white', linewidth=1)
# add zoomed-in plot later

axins_5 = zoomed_inset_axes(axes_sys[0], zoom = 2.5, loc='upper right')
plotter_kernel(axins_5, time, x_ar_traj[:, :, 0],
               legend_1, my_linewidth, mygreen, linestyle_1)
plotter_kernel(axins_5, time, x_nr_traj[:, :, 0],
               legend_2, my_linewidth, myblue, linestyle_2)
plotter_kernel(axins_5, time, x_n_traj[:, :, 0],
               legend_3, my_linewidth, myred, linestyle_3)
plotter_kernel(axins_5, time, x_r_traj[:, :, 0],
               legend_4, my_linewidth, mypurple, linestyle_4)
axins_5.plot(time_s, omega_ref_trajectory, linestyle='--', color='black', linewidth=my_linewidth)
axins_5.set_xticks([])
axins_5.yaxis.tick_left()
axins_5.set_xlim(1.05, 1.2)
axins_5.set_ylim(245, 340)
mark_inset(axes_sys[0], axins_5, loc1=2, loc2=4, fc="none", ec="gray")

axes_sys[0].set_xticklabels([])

# Plot 2: q-axis current
plotter_kernel(axes_sys[1], time, x_ar_traj[:, :, 1],
               legend_1, my_linewidth, mygreen, linestyle_1)
plotter_kernel(axes_sys[1], time, x_nr_traj[:, :, 1],
               legend_2, my_linewidth, myblue, linestyle_2)
plotter_kernel(axes_sys[1], time, x_n_traj[:, :, 1],
               legend_3, my_linewidth, myred, linestyle_3)
plotter_kernel(axes_sys[1], time, x_r_traj[:, :, 1],
               legend_4, my_linewidth, mypurple, linestyle_4)
axes_sys[1].axhline(-pmsm_params['iq_lim'], color='gray', linestyle='--', linewidth=my_linewidth)
axes_sys[1].axhline(pmsm_params['iq_lim'], color='gray', linestyle='--', linewidth=my_linewidth)
axes_sys[1].set_facecolor((0.95, 0.95, 0.95))
axes_sys[1].set_ylabel(r'$i_{\mathrm{q}}$ [A]')
axes_sys[1].grid(True, linestyle='--', color='white', linewidth=1)
# add zoomed-in plot later
axins_3 = zoomed_inset_axes(axes_sys[1], zoom = 3, loc='upper right')
plotter_kernel(axins_3, time, x_ar_traj[:, :, 1],
               legend_1, my_linewidth, mygreen, linestyle_1)
plotter_kernel(axins_3, time, x_nr_traj[:, :, 1],
               legend_2, my_linewidth, myblue, linestyle_2)
plotter_kernel(axins_3, time, x_n_traj[:, :, 1],
               legend_3, my_linewidth, myred, linestyle_3)
plotter_kernel(axins_3, time, x_r_traj[:, :, 1],
               legend_4, my_linewidth, mypurple, linestyle_4)
axins_3.axhline(-pmsm_params['iq_lim'], color='gray', linestyle='--', linewidth=my_linewidth)
axins_3.axhline(pmsm_params['iq_lim'], color='gray', linestyle='--', linewidth=my_linewidth)
axins_3.set_xticks([])
axins_3.yaxis.tick_left()
axins_3.set_xlim(0.95, 1.15)
axins_3.set_ylim(2.3, 3.2)
mark_inset(axes_sys[1], axins_3, loc1=2, loc2=4, fc="none", ec="gray")

axins_4 = zoomed_inset_axes(axes_sys[1], zoom = 2.5, loc='lower right')
plotter_kernel(axins_4, time, x_ar_traj[:, :, 1],
               legend_1, my_linewidth, mygreen, linestyle_1)
plotter_kernel(axins_4, time, x_nr_traj[:, :, 1],
               legend_2, my_linewidth, myblue, linestyle_2)
plotter_kernel(axins_4, time, x_n_traj[:, :, 1],
               legend_3, my_linewidth, myred, linestyle_3)
plotter_kernel(axins_4, time, x_r_traj[:, :, 1],
               legend_4, my_linewidth, mypurple, linestyle_4)
axins_4.axhline(-pmsm_params['iq_lim'], color='gray', linestyle='--', linewidth=my_linewidth)
axins_4.axhline(pmsm_params['iq_lim'], color='gray', linestyle='--', linewidth=my_linewidth)
axins_4.set_xticks([])
axins_4.yaxis.tick_left()
axins_4.set_xlim(1.98, 2.15)
axins_4.set_ylim(-3.2, -2)
mark_inset(axes_sys[1], axins_4, loc1=2, loc2=4, fc="none", ec="gray")


axes_sys[1].set_xticklabels([])

# Plot 3: control input (stator voltage)
plotter_kernel(axes_sys[2], time_s, u_ar_traj[:, :, 0],
               legend_1, my_linewidth, mygreen, linestyle_1)
plotter_kernel(axes_sys[2], time_s, u_nr_traj[:, :, 0],
               legend_2, my_linewidth, myblue, linestyle_2)
plotter_kernel(axes_sys[2], time_s, u_n_traj[:, :, 0],
               legend_3, my_linewidth, myred, linestyle_3)
plotter_kernel(axes_sys[2], time_s, u_r_traj[:, :, 0],
               legend_4, my_linewidth, mypurple, linestyle_4)
axes_sys[2].axhline(-pmsm_params['u_lim'], color='gray', linestyle='--', linewidth=my_linewidth)
axes_sys[2].axhline(pmsm_params['u_lim'], color='gray', linestyle='--', linewidth=my_linewidth)
axes_sys[2].set_facecolor((0.95, 0.95, 0.95))
axes_sys[2].set_ylabel(r'$u_{\mathrm{q}}$ [V]')
axes_sys[2].grid(True, linestyle='--', color='white', linewidth=1)
# add zoomed-in plot later
# Set x-axis label only on the last plot
axes_sys[2].set_xlabel('Time [s]')

fig_sys.tight_layout()
plt.subplots_adjust(hspace=0.3)
fig_sys.savefig('pmsm_control.pdf', format='pdf', bbox_inches='tight', dpi=300)

# plot the parameter estimation trajectory
fig_rls, axes_rls = plt.subplots(3, 1, figsize=(4.5, 2.8), layout='constrained')

# Plot 1: flux linkage estimation
plotter_kernel(axes_rls[0], time, para_ar_traj[:, :, 0],
               r'$\hat{\phi}_{\mathrm{f}}$', my_linewidth, mydarkblue, linestyle_1)
axes_rls[0].axhline(y=para_star[0], label = r'$\phi^\ast_{\mathrm{f}}$',
                    color=mydarkblue_cop, linestyle=':', linewidth=my_linewidth)
# axes_rls[0].set_title("RLS estimation with SMID")
axes_rls[0].legend(loc='upper right')
axes_rls[0].set_facecolor((0.95, 0.95, 0.95))
axes_rls[0].grid(True, linestyle='--', color='white', linewidth=1)

axins_0 = zoomed_inset_axes(axes_rls[0], zoom = 4, loc='upper center')
plotter_kernel(axins_0, time, para_ar_traj[:, :, 0],
               r'$\hat{\phi}_{\mathrm{f}}$', my_linewidth, mydarkblue, linestyle_1)
axins_0.axhline(y=para_star[0], label = r'$\phi^\ast_{\mathrm{f}}$',
                    color=mydarkblue_cop, linestyle=':', linewidth=my_linewidth)
axins_0.set_xticks([])
axins_0.yaxis.tick_right()
axins_0.set_xlim(-0.01, 0.075)
axins_0.set_ylim(0.0785, 0.083)
mark_inset(axes_rls[0], axins_0, loc1=2, loc2=4, fc="none", ec="gray")


axes_rls[0].set_xticklabels([])

# Plot 2: friction coefficient estimation
plotter_kernel(axes_rls[1], time, para_ar_traj[:, :, 1],
               r'$\hat{B}_{\mathrm{vis}}$', my_linewidth, myleaveyellow, linestyle_1)
axes_rls[1].axhline(y=para_star[1], label = r'$\hat{B}_{\mathrm{vis}}^\ast$',
                    color=myleaveyellow_cop, linestyle=':', linewidth=my_linewidth)
axes_rls[1].legend(loc='upper right')
axes_rls[1].set_facecolor((0.95, 0.95, 0.95))
axes_rls[1].grid(True, linestyle='--', color='white', linewidth=1)
axes_rls[1].set_xticklabels([])

formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_powerlimits((-4, 4))  # Force multiplier display if within 10^3 range
axes_rls[1].yaxis.set_major_formatter(formatter)

# Plot 3: resistance estimation
plotter_kernel(axes_rls[2], time, para_ar_traj[:, :, 2],
               r'$\hat{R}$', my_linewidth, mysolfyellow, linestyle_1)
axes_rls[2].axhline(y=para_star[2], label = r'$\hat{R}^\ast$',
                    color=mysoftyellow_cop, linestyle=':', linewidth=my_linewidth)
axes_rls[2].legend(loc='upper right')
axes_rls[2].set_facecolor((0.95, 0.95, 0.95))
axes_rls[2].grid(True, linestyle='--', color='white', linewidth=1)

axins_2 = zoomed_inset_axes(axes_rls[2], zoom = 4, loc='upper center')
plotter_kernel(axins_2, time, para_ar_traj[:, :, 2],
               r'$\hat{R}$', my_linewidth, mysolfyellow, linestyle_1)
axins_2.axhline(y=para_star[2], label = r'$\hat{R}^\ast$',
                    color=mysoftyellow_cop, linestyle=':', linewidth=my_linewidth)
axins_2.set_xticks([])
axins_2.yaxis.tick_right()
axins_2.set_xlim(-0.01, 0.075)
axins_2.set_ylim(0.795, 0.825)
mark_inset(axes_rls[2], axins_2, loc1=2, loc2=4, fc="none", ec="gray")

axes_rls[2].set_xlabel('Time [s]')

fig_rls.tight_layout()

plt.subplots_adjust(hspace=0.3)
fig_rls.savefig('pmsm_estimation.pdf', format='pdf', bbox_inches='tight', dpi=300)

plt.show()

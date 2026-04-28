import numpy as np
import matplotlib.pyplot as plt
# import control as ctrl
from cbf_sf.safety_filter import AdaptiveSafetyFilterLinear
from cbf_sf.diverse_functions import ext_kappa_pmsm, cbf_pmsm_linear
from nmpc.diverse_functions import pmsm_drift, pmsm_kernel, pmsm_gw, pmsm_dynamics, pmsm_kernel_opt
from nmpc.controller import PPD_PMSM_Controller
from rls.rls_main import RLSUpdate
from rls.rls_utils import interleave_vec, interleave_diag, pmsm_recover
from simulation_utils.generators import generate_step_reference, generate_noise_simulations, rpm_to_rads
from optimization_utils.metric import max_2norm_polytope, max_l1_deviation_value

## ---------- Initialization ------------

# Time parameters
dt = 1e-3 # sampling interval [s]
T  = 4 # total time horizon [s]
N_t = int(T / dt) # total time steps
T_initial = 0 # time before disturbances start [s]

# PMSM parameters
pmsm_params = {
    'R': 0.8, # resistance [Ohm]
    'L': 2.9e-3, # d-axis inductance [H]
    'T_load': 0, # load torque [N*m]
    'phif': 0.081, # flux linkage [Wb]
    'num_pole': 4, # number of pole pairs
    'J': 2.35e-4, # inertia [kg*m^2]
    'B': 7.4e-4, # friction coefficient [N*m*s/rad]
    'iq_lim': 2.8, # maximum q-axis current [A]
    'u_lim': 220 # maximum voltage [V]
}

# disturbance parameters
load_max = 0.0225 # maximum load disturbance (torque) [N*m]
id_max = 0.04

# speed bound
omega_max = rpm_to_rads(4000)

# Create a Generator with a specific seed (e.g., 42)
rng_load = np.random.default_rng(seed=42)
rng_id = np.random.default_rng(seed=24)

# generate disturbance trajectories
load_disturbance = 2 * generate_noise_simulations(N_t, 1, rng_load, t_initial=int(0/dt), noise_scale=load_max) - load_max # scale to [-load_max, load_max]
id_disturbance = 2 * generate_noise_simulations(N_t, 1, rng_id, t_initial=int(0/dt), noise_scale=id_max) - id_max # scale to [-id_max, id_max]
# generate reference trajectory
t_w_vec = [int(0/dt), int(0.5/dt), int(1/dt), int(2/dt), int(3/dt)]
w_ref_vec = [rpm_to_rads(500), rpm_to_rads(-500), rpm_to_rads(3000), rpm_to_rads(-500), rpm_to_rads(0)] # reference angular velocity in rad/s
w_ref_trajectory = generate_step_reference(N_t, t_w_vec, w_ref_vec)

# # generate load trajectory
# t_load_vec = [int(T_initial/dt)]
# load_ref_vec = [pmsm_params['T_load']] # reference load torque in N*m
# load_ref_trajectory = generate_step_reference(N_t, t_load_vec, load_ref_vec)

# sys_params = {
#     'J': pmsm_params['J'],
#     'B': pmsm_params['B'],
#     'phif': pmsm_params['phif'],
# }

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

# state and input dimensions
x_dim = 2
u_dim = 1

# The range of maximum voltage [V]
u_lim = pmsm_params['u_lim']

#  --------------- Disturbance information -----------------
# disturbance limit
w_lim = dt * np.array([load_max / pmsm_params['J'], id_max * pmsm_params['num_pole'] * omega_max])

# disturbance matrix
H_w = interleave_diag(-w_lim, w_lim)
# -----------------------------------------------------------

# ----------------- Parameter bounds for SMID ------------------
pmsm_bounds = {
    'bound_phif': (0.065, 0.095), # bounds for the flux linkage [Wb]
    'bound_B': (7e-4, 8e-4), # bounds for the friction coefficient [N*m*s/rad]
    'bound_R': (0.6, 0.9) # bounds for the resistance [Ohm]
}

num_para = 3
para_0 = np.array([pmsm_bounds['bound_phif'][1], pmsm_bounds['bound_B'][0], pmsm_bounds['bound_R'][0]])
para_star = np.array([est_params['phif'], est_params['B'], est_params['R']])

# parameter bound
LB_para = [pmsm_bounds['bound_phif'][0], pmsm_bounds['bound_B'][0], pmsm_bounds['bound_R'][0]]
UB_para = [pmsm_bounds['bound_phif'][1], pmsm_bounds['bound_B'][1], pmsm_bounds['bound_R'][1]]

# parameter matrix
H_para = interleave_diag(-np.ones(num_para), np.ones(num_para))
h_para = interleave_vec(LB_para, UB_para)

# initial error bound
bound_0 = max_l1_deviation_value(H_para, h_para, para_0)
# -----------------------------------------------------------

# ------------- Controller and safety filter ----------------
# load the nominal PID controller
controller = PPD_PMSM_Controller(linear_model_params['A'], linear_model_params['B'], 
                                 desired_poles=[-25 + 20j, -25 - 20j], int_omega_error_gain=1e-6)

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

# --------------------------------------------------


X0 = np.array([0.0, 0.0]) # initial state: [angular velocity, q-axis current]

x_traj = np.zeros((N_t + 1, 2)) # to store the state trajectory
u_traj = np.zeros((N_t, 1)) # to store the control input trajectory

x_traj[0, :] = X0 # set the initial state
int_omega_error = 0.0 # initialize integral of the angular velocity error

# ----------- initialize the RLS module ------------
# initial covaraince matrix
var_para = np.array([1, 1e-4, 10])
cov = np.diag(var_para)
# define the estimator
my_rls = RLSUpdate(num_para, x_dim, pmsm_drift, pmsm_kernel, pmsm_params, linear_model_params, dt, H_w)
# initialize the parameter trajectory
para_traj = np.ones((N_t + 1, num_para)) * para_0
# --------------------------------------------------
T_turning = int(5/dt) # time step when the load disturbance is added, also the time step when the safety filter starts to work
T_project = int(5/dt) # time step when the load disturbance is added, also the time step when the safety filter starts to work

# ----------- Initialize additional arrays for safety filter ------------
bound_traj = np.zeros(N_t + 1) 

diff_para = 0 # initialize the parameter increments
para_traj[0, :] = para_0 # assign the initial parameter estimate
bound_traj[0] = bound_0 # assign the initial error bound
# --------------------------------------------------

## ---------- Simulation Loop ------------
for t in range(N_t):
    # Get current state and reference
    x_t = x_traj[t, :]
    omega_ref = w_ref_trajectory[t]

    # Compute control input of three different controllers
    u_t, int_omega_error = controller.compute_control(omega_ref, x_t, 
                                                      pmsm_params, para_traj[0, :],
                                                      int_omega_error)
    # u_t, int_omega_error = controller.compute_control(omega_ref, x_t, pmsm_params, sys_params, int_omega_error)
    if t > T_turning:
        u_t_safe = my_sf.filter(x_t, bound_traj[0], 0, para_traj[0, :], u_t)
    else:
        u_t_safe = u_t

    # Store control input
    u_traj[t] = u_t_safe
    # u_traj[t] = u_t

    # print(f"current input: {u_t}")

    # ------ Compute different terms of the dynamics ------
    f_t = pmsm_drift(x_t, pmsm_params, dt=dt)
    phi_t = pmsm_kernel(x_t, pmsm_params, dt=dt)
    g_w_t = pmsm_gw(x_t, pmsm_params, dt=dt)

    # print(g_u_t.shape)
    # x_next = f_t - phi_t.T @ para_nom + g_u_t * u_t + g_w_t * w_t
    d_w = g_w_t * np.array([load_disturbance[t], id_disturbance[t]])
    # x_next  = (np.eye(2) + dt * linear_model_params['A']) @ x_t + dt * linear_model_params['B'] @ u_t + d_w
    x_next = f_t - phi_t.T @ para_star + dt * linear_model_params['B'] @ u_traj[t] + d_w

    # Store next state
    x_traj[t + 1, :] = x_next

    # para_info = my_rls.update_para(x_traj[t+1, :], x_traj[t, :], u_traj[t, :], para_traj[t, :], cov, t+1)

    # para_traj[t + 1, :] = para_info["para"]
    # if t < T_project:
    #     H_para_next, h_para_next = my_rls.update_paraset(x_traj[t+1, :], x_traj[t, :], u_traj[t, :],
    #                                                      H_para, h_para, t+1)
    #     para_post = my_rls.posterior(H_para, h_para, para_info["para"], para_traj[t, :])
    #     para_traj[t + 1, :] = para_post["para"]

    # # bound_traj[t + 1] = para_post["bound"]
    # diff_para = np.linalg.norm(para_traj[t + 1, :] - para_traj[t, :], ord=2)
    # bound_traj[t + 1] = np.random.uniform(1, 3)*np.linalg.norm(para_traj[t + 1, :] - para_star, ord=2)
    # cov = para_info["cov"]

    print(f"current step: {t}")


## ---------- Plotting ------------
# --------- State and input subplots (3 rows, 1 column)
fig_sys, axes_sys = plt.subplots(3, 1, figsize=(7, 7*5/6))

# Plot 1
axes_sys[0].plot(np.arange(N_t+1) * dt, x_traj[:, 0], label=r'$\omega$')
axes_sys[0].plot(np.arange(N_t) * dt, w_ref_trajectory, label=r'$\omega_{\mathrm{ref}}$ (rad/s)', linestyle='--')
#plt.xlabel('Time (s)')
axes_sys[0].set_ylabel('Speed (rad/s)')
axes_sys[0].legend()
axes_sys[0].set_title('PMSM Simulation Results')
axes_sys[0].set_facecolor((0.95, 0.95, 0.95))
axes_sys[0].grid(True, linestyle='--', color='white', linewidth=1)

# Plot 2
# axes[1].plot(time_state, D_traj_opt, label='distance (OPT)')
axes_sys[1].plot(np.arange(N_t+1) * dt, x_traj[:, 1], label=r'$i_q$')
axes_sys[1].axhline(-pmsm_params['iq_lim'], color='gray', linestyle='--', linewidth=0.5)
axes_sys[1].axhline(pmsm_params['iq_lim'], color='gray', linestyle='--', linewidth=0.5)
axes_sys[1].set_ylabel('Current (A)')
axes_sys[1].legend()
axes_sys[1].set_facecolor((0.95, 0.95, 0.95))
axes_sys[1].grid(True, linestyle='--', color='white', linewidth=1)

# Plot 3
axes_sys[2].plot(np.arange(N_t) * dt, u_traj[:, 0], label=r'$u_q$')
axes_sys[2].axhline(-pmsm_params['u_lim'], color='gray', linestyle='--', linewidth=0.5)
axes_sys[2].axhline(pmsm_params['u_lim'], color='gray', linestyle='--', linewidth=0.5)
axes_sys[2].set_xlabel('Time (s)')
axes_sys[2].set_ylabel('Voltage (V)')
axes_sys[2].legend()
axes_sys[2].set_facecolor((0.95, 0.95, 0.95))
axes_sys[2].grid(True, linestyle='--', color='white', linewidth=1)

# --------- RLS subplots (4 rows, 1 column) -----------
fig_rls, axes_rls = plt.subplots(4, 1, figsize=(6, 6))
# Plot 1
axes_rls[0].plot(np.arange(N_t+1) * dt, para_traj[:, 0], label=r'$\hat{\phi}_{\text{f}}$')
axes_rls[0].axhline(y=para_star[0], label = r'$\phi^\ast_{\text{f}}$', color='r', linestyle='--', linewidth=2)
axes_rls[0].set_title("RLS Estimation with SMID")
axes_rls[0].legend()
axes_rls[0].set_facecolor((0.95, 0.95, 0.95))
axes_rls[0].grid(True, linestyle='--', color='white', linewidth=1)

# Plot 2
axes_rls[1].plot(np.arange(N_t+1) * dt, para_traj[:, 1], label=r'$\hat{B}$')
axes_rls[1].axhline(y=para_star[1], label = r'$B^\ast$', color='r', linestyle='--', linewidth=2)
axes_rls[1].legend(loc='upper right')
axes_rls[1].set_facecolor((0.95, 0.95, 0.95))
axes_rls[1].grid(True, linestyle='--', color='white', linewidth=1)

# Plot 3
axes_rls[2].plot(np.arange(N_t+1) * dt, para_traj[:, 2], label=r'$\hat{R}$')
axes_rls[2].axhline(y=para_star[2], label = r'$R^\ast$', color='r', linestyle='--', linewidth=2)
axes_rls[2].set_xlabel('Time (s)')
axes_rls[2].legend()
axes_rls[2].set_facecolor((0.95, 0.95, 0.95))
axes_rls[2].grid(True, linestyle='--', color='white', linewidth=1)

# Plot 4
axes_rls[3].plot(np.arange(N_t+1) * dt, bound_traj, label=r'error bound')
# axes_rls[3].axhline(y=para_star[3], label = r'$\theta^\ast_4$', color='r', linestyle='--', linewidth=2)
axes_rls[3].set_xlabel('Time (s)')
axes_rls[3].legend()
axes_rls[3].set_facecolor((0.95, 0.95, 0.95))
axes_rls[3].grid(True, linestyle='--', color='white', linewidth=1)


plt.tight_layout()
plt.show()






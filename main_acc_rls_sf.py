"""
This is main script for ACC using MPC, RLS with SMID, and safety filter
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from rls.rls_main import RLSProjection
from rls.rls_utils import interleave_vec, interleave_diag
from nmpc.diverse_functions import sacc_dynamics, sacc_fg, sacc_kernel
from nmpc.controller import MPCController, RAMPCACC
from cbf_sf.diverse_functions import ext_kappa, cbf_acc_linear
from cbf_sf.safety_filter import AdaptiveSafetyFilter
from optimization_utils.metric import max_l1_deviation_value, max_2norm_polytope
from plotters.utils import plotter_kernel
plt.rcParams.update({
    "text.usetex": True,                  # Use LaTeX for text rendering
    "font.family": "serif",               # Use a serif font
    "font.serif": ["Computer Modern Roman"] # Set the font
})

# --------------- Basic setting ---------------
# state and input dimensions
dt = 0.1 # trivial sampling time
x_dim = 2
u_dim = 1

# The range of maximum traction force [Newton]
u_lim = 1e4

#  --------------- Disturbance information -----------------
w_lim = np.array([0.2, 1])

# disturbance matrix
H_w = interleave_diag(-w_lim, w_lim)

# disturbance bound (2-norm)
bar_w  = max_2norm_polytope(H_w, np.ones(2 * x_dim))

# ---------------- Parameter information -----------------
num_para = 2
para_0 = np.array([0.15, 30])
para_star = np.array([0.55, 22])

# parameter bound
LB_para = [0.1, 20]
UB_para = [0.6, 32]

# parameter matrix
H_para = interleave_diag(-np.ones(num_para), np.ones(num_para))
h_para = interleave_vec(LB_para, UB_para)

# initial error bound
bound_0 = max_l1_deviation_value(H_para, h_para, para_0)

# print("H_para:", H_para)
# print("h_para:", h_para)

# ----------- Test dynamics and one step estimation test --------------
# initial state
x_0 = np.array([22, 96])
# u_0 = 100 * np.random.randn(1)

# # sample a random disturbance
# w = dt * np.random.uniform(-w_lim, w_lim, size=x_dim)

# x_p = sacc_dynamics(x_0, u_0, para_star) + w
# # x_p2 = vacc_fg(x_0, u_0) - vacc_kernel(x_0).T @ para_star

# print("next state integrated:", x_p)
# print("next state separated:", x_p2)

# ----------- initialize the RLS module ------------
# define the estimator
my_rls = RLSProjection(num_para, x_dim, sacc_fg, sacc_kernel, dt, H_w)
# --------------------------------------------------

# ----------- initialize the MPC module -------------
# cost weights
Q = np.array([[1000, 0], [0, 1e-2]])
R = np.array([5 * 1e-4]) # low weights, we do not care about the input
P = 10 * Q
# state reference
v_ref = 30
d_ref = v_ref * 1.8
x_ref = np.array([v_ref, d_ref])

# input reference
u_static = 150
u_ref = np.array([u_static])

# prediction horizon
# T_prediction = 0.1
N_prediction = 1

# define the unconstrained nominal MPC controller
my_mpc = MPCController(N_prediction, dt,
                       x_dim, u_dim,
                       x_ref, u_ref,
                       -u_lim, u_lim,
                       Q, R, P,
                       sacc_dynamics, num_para)
# define the robust adaptive MPC controller with constraint tightening
my_rampc = RAMPCACC(dt, 1650.0, bar_w,
                    x_ref, u_ref,
                    -u_lim, u_lim,
                    Q, R, P,
                    sacc_dynamics, num_para)
# ----------------------------------------------------

# ----------- initialize the SF module ------------
# the minimum eigenvalue
gamma = 1e5

# compute the Lipschitz constant (linear barrier function)
L_B = np.linalg.norm(np.array([1, 1.8]))

# define the safety filter
my_sf = AdaptiveSafetyFilter(dt, num_para,
                             gamma, L_B, bar_w,
                             x_dim, u_dim,
                             -u_lim, u_lim,
                             sacc_dynamics, sacc_kernel, cbf_acc_linear,
                             ext_kappa, "linear")

# --------------------------------------------------

# ------------ Simulation ------------------

# simulation time
T = 100
# T_gap = 1
# T_rls = int(T / T_gap)
# time_rls = np.arange(0, T_rls + 1)
# time_rls_s = np.arange(0, T_rls)
time = dt * np.arange(0, T + 1)
time_s = dt * np.arange(0, T)
# lr = 0.1

# # number of disturbance realizations for simulation
# N_sim = 100

# # Initialize the output trajectory for adaptive safe control
# xa_traj = np.zeros((N_sim, T + 1, x_dim)) # state
# u_asf_traj = np.zeros((N_sim, T, u_dim)) # filtered safe input
# para_asf_traj = np.zeros((N_sim, T + 1, num_para)) # parameter
# bound_asf_traj = np.zeros((N_sim, T + 1)) # parameter error bound

# # Initialize the output trajectory for nominal safe control
# xn_traj = np.zeros((N_sim, T + 1, x_dim)) # state
# u_nsf_traj = np.zeros((N_sim, T, u_dim)) # nominal input
# para_nsf_traj = np.zeros((N_sim, T + 1, num_para)) # parameter
# bound_nsf_traj = np.zeros((N_sim, T + 1)) # parameter error bound

# # Initialize the output trajectory for unsafe control
# x_traj = np.zeros((N_sim, T + 1, x_dim)) # state
# u_traj = np.zeros((N_sim, T, u_dim)) # unsafe input
# para_traj = np.zeros((N_sim, T + 1, num_para)) # parameter
# bound_traj = np.zeros((N_sim, T + 1)) # parameter error bound

# # Initialize the output trajectory for robust adaptive MPC
# xs_traj = np.zeros((N_sim, T + 1, x_dim)) # state
# u_ssf_traj = np.zeros((N_sim, T, u_dim)) # filtered safe input
# para_ssf_traj = np.zeros((N_sim, T + 1, num_para)) # parameter
# bound_ssf_traj = np.zeros((N_sim, T + 1)) # parameter error bound

# # Extra term for review response
# E_theta_traj_sf = np.zeros((N_sim, T))
# E_theta_traj = np.zeros((N_sim, T))

# # Simulation main loops
# for i in range(N_sim):
# # outer loop for different realizations
#     # Initialize the state
#     xa_traj[i, 0, :] = x_0
#     xn_traj[i, 0, :] = x_0
#     x_traj[i, 0, :] = x_0
#     xs_traj[i, 0, :] = x_0
#     # initialize estimation for adaptive safe control
#     para_asf_traj[i, 0, :] = para_0
#     bound_asf_traj[i, 0] = bound_0
#     # initialize estimation for nominal safe control
#     para_asf_traj[i, 0, :] = para_0
#     bound_asf_traj[i, 0] = bound_0
#     # initialize estimation for unsafe control
#     para_traj[i, 0, :] = para_0
#     # initialize estimation for robust adaptive MPC
#     para_ssf_traj[i, 0, :] = para_0
#     bound_ssf_traj[i, 0] = bound_0

#     # Initialize the first parameter difference for adaptive safe control (no update)
#     diff_para = 0

#     # reset the initial covaraince matrix
#     var_para = np.array([500, 1000])
#     cov_sf = np.diag(var_para) # for adaptive safe control
#     cov = np.diag(var_para) # for unsafe adaptive MPC
#     cov_ss = np.diag(var_para) # for robust adaptive MPC

#     # sample a disturbance realization
#     w = dt * np.random.uniform(-w_lim, w_lim, size=(T, x_dim))

#     # reset the parameter matrix
#     H_para_sf = interleave_diag(-np.ones(num_para), np.ones(num_para)) # for adaptive safe control
#     h_para_sf = interleave_vec(LB_para, UB_para) # for adaptive safe control

#     H_para = interleave_diag(-np.ones(num_para), np.ones(num_para)) # for unsafe adaptive MPC
#     h_para = interleave_vec(LB_para, UB_para) # for unsafe adaptive MPC

#     H_para_ss = interleave_diag(-np.ones(num_para), np.ones(num_para)) # for adaptive safe control
#     h_para_ss = interleave_vec(LB_para, UB_para) # for adaptive safe control

#     for t in range(T):
#     # inner loop for time simulation
#         # ----------------------- Nominal Input Computation --------------------------
#         # nominal input of adaptive safe control
#         u_asf_prior = my_mpc.solve_closed(xa_traj[i, t, :], para_asf_traj[i, t, :])
#         # nominal input of nominal safe control
#         u_nsf_prior = my_mpc.solve_closed(xn_traj[i, t, :], para_0)
#         # nominal MPC input
#         u_traj[i, t, :] = my_mpc.solve_closed(x_traj[i, t, :], para_traj[i, t, :])
#         # robust adaptive MPC
#         u_ssf_traj[i, t, :] = my_rampc.solve_closed(xs_traj[i, t, :], para_ssf_traj[i, t, :], bound_ssf_traj[i, t])

#         # ----------------------- Safety Filter Implementation --------------------------
#         # safe input of adaptive safe control
#         u_asf_traj[i, t, :], E_theta_traj_sf[i, t] = my_sf.filter(xa_traj[i, t, :], para_asf_traj[i, t, :], u_asf_prior,
#                                            diff_para, bound_asf_traj[i, t])
#         # safe input of nominal safe control
#         u_nsf_traj[i, t, :], E_theta_traj[i, t] = my_sf.filter(xn_traj[i, t, :], para_0, u_nsf_prior, 0, bound_0)

#         # ----------------------- State Propagation --------------------------
#         # adaptive safe control
#         xa_traj[i, t+1, :] = sacc_dynamics(xa_traj[i, t, :], u_asf_traj[i, t, :], para_star) + w[t, :]
#         # nominal safe control
#         xn_traj[i, t+1, :] = sacc_dynamics(xn_traj[i, t, :], u_nsf_traj[i, t, :], para_star) + w[t, :]
#         # unsafe control
#         x_traj[i, t+1, :] = sacc_dynamics(x_traj[i, t, :], u_traj[i, t, :], para_star) + w[t, :]
#         # robust adaptive MPC
#         xs_traj[i, t+1, :] = sacc_dynamics(xs_traj[i, t, :], u_ssf_traj[i, t, :], para_star) + w[t, :]

#         # ----------------------- Parameter Update ------------------------
#         para_info_sf = my_rls.update_para(xa_traj[i, t+1, :], xa_traj[i, t, :], 
#                                        u_asf_traj[i, t, :], para_asf_traj[i, t, :], cov_sf, t+1)
#         para_info = my_rls.update_para(x_traj[i, t+1, :], x_traj[i, t, :], 
#                                        u_traj[i, t, :], para_traj[i, t, :], cov, t+1)
#         para_info_ss = my_rls.update_para(xs_traj[i, t+1, :], xs_traj[i, t, :], 
#                                        u_ssf_traj[i, t, :], para_ssf_traj[i, t, :], cov_ss, t+1)

#         # para_traj[t_rls + 1, :] = para_info["para"]
#         H_para_next_sf, h_para_next_sf = my_rls.update_paraset(xa_traj[i, t+1, :], xa_traj[i, t, :], u_asf_traj[i, t, :],
#                                                             H_para_sf, h_para_sf, t+1)
#         H_para_next, h_para_next = my_rls.update_paraset(x_traj[i, t+1, :], x_traj[i, t, :], u_traj[i, t, :],
#                                                             H_para, h_para, t+1)
#         H_para_next_ss, h_para_next_ss = my_rls.update_paraset(xs_traj[i, t+1, :], xs_traj[i, t, :], u_ssf_traj[i, t, :],
#                                                             H_para_ss, h_para_ss, t+1)
#         para_post_sf = my_rls.posterior(H_para_next_sf, h_para_next_sf, 
#                                         para_info_sf["para"], para_asf_traj[i, t, :])
#         para_post = my_rls.posterior(H_para_next, h_para_next, 
#                                         para_info["para"], para_traj[i, t, :])
#         para_post_ss = my_rls.posterior(H_para_next_ss, h_para_next_ss, 
#                                         para_info_ss["para"], para_ssf_traj[i, t, :])
#         # -------- Parameter update ----------
#         para_asf_traj[i, t+1, :] = para_post_sf["para"]
#         para_traj[i, t+1, :] = para_post["para"]
#         para_ssf_traj[i, t+1, :] = para_post_ss["para"]
#         diff_para = para_post_sf["delta"] # for raCBF-based safe control
#         # bound_asf_traj[i, t+1] = np.random.uniform(1, 3)*np.linalg.norm(para_asf_traj[i, t+1, :] - para_star, ord=2)
#         # bound_traj[i, t+1] = np.random.uniform(1, 3)*np.linalg.norm(para_traj[i, t+1, :] - para_star, ord=2)
#         # bound_ssf_traj[i, t+1] = np.random.uniform(1, 3)*np.linalg.norm(para_ssf_traj[i, t+1, :] - para_star, ord=2)
#         bound_asf_traj[i, t+1] = para_post_sf["diff"]
#         bound_traj[i, t+1] = para_post["diff"]
#         bound_ssf_traj[i, t+1] = para_post_ss["diff"]
#         cov_sf = para_info_sf["cov"]
#         cov = para_info["cov"]
#         cov_ss = para_info_ss["cov"]
#         # innovation[t_rls] = para_info["inc_state"]
#         print("time completed:", t+1)
    
#     # progress check
#     print("The round completed:", i+1)

# # save data for reuse
# np.savez('data_asf.npz', state=xa_traj, input=u_asf_traj, 
#          para=para_asf_traj, bound=bound_asf_traj, E=E_theta_traj_sf)
# np.savez('data_nsf.npz', state=xn_traj, input=u_nsf_traj)
# np.savez('data_usf.npz', state=x_traj, input=u_traj)
# np.savez('data_ssf.npz', state=xs_traj, input=u_ssf_traj)

data_asf = np.load("data_asf.npz")
data_nsf = np.load("data_nsf.npz")
data_usf = np.load("data_usf.npz")
data_ssf = np.load("data_ssf.npz")

xa_traj = data_asf["state"]
xn_traj = data_nsf["state"]
x_traj = data_usf["state"]
xs_traj = data_ssf["state"]

u_asf_traj = data_asf["input"]
u_nsf_traj = data_nsf["input"]
u_traj = data_usf["input"]
u_ssf_traj = data_ssf["input"]

para_asf_traj = data_asf["para"]
bound_asf_traj = data_asf["bound"]
E_theta_traj_sf = data_asf["E"]

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
legend_1 = "aMPC-raCBF"
legend_2 = "MPC-rCBF"
legend_3 = "aMPC"
legend_4 = "raMPC"
linestyle_1 = '-'
linestyle_2 = '-'
linestyle_3 = '-'
linestyle_4 = '-'

# --------- State and input subplots (3 rows, 1 column)
fig_sys, axes_sys = plt.subplots(3, 1, figsize=(4.5, 4))

# Plot 1
# axes[0].plot(time_state, v_traj_opt, label='velocity (OPT)')
plotter_kernel(axes_sys[0], time, xa_traj[:, :, 0],
               legend_1, my_linewidth, mygreen, linestyle_1)
plotter_kernel(axes_sys[0], time, xn_traj[:, :, 0],
               legend_2, my_linewidth, myblue, linestyle_2)
plotter_kernel(axes_sys[0], time, x_traj[:, :, 0],
               legend_3, my_linewidth, myred, linestyle_3)
plotter_kernel(axes_sys[0], time, xs_traj[:, :, 0],
               legend_4, my_linewidth, mypurple, linestyle_4)
# axes_sys[0].axhline(y=para_star[0], label = 'true drag coefficient', color='r', linestyle='--', linewidth=2)
# axes_sys[0].set_title("Adaptive MPC with CBF-based safety filter")
axes_sys[0].legend(loc='upper center',
    bbox_to_anchor=(0.5, 1.75),  # position relative to the whole figure
    ncol=2,                        # all items in one row
    frameon=True)
axes_sys[0].set_facecolor((0.95, 0.95, 0.95))
axes_sys[0].set_ylabel(r'$v$ [m/s]')
axes_sys[0].grid(True, linestyle='--', color='white', linewidth=1)

axins_0 = zoomed_inset_axes(axes_sys[0], zoom = 6, loc='lower center')
plotter_kernel(axins_0, time, xa_traj[:, :, 0], 
               legend_1, my_linewidth, mygreen, linestyle_1)
plotter_kernel(axins_0, time, xn_traj[:, :, 0], 
               legend_2, my_linewidth, myblue, linestyle_2)
plotter_kernel(axins_0, time, x_traj[:, :, 0],
               legend_3, my_linewidth, myred, linestyle_3)
plotter_kernel(axins_0, time, xs_traj[:, :, 0],
               legend_4, my_linewidth, mypurple, linestyle_4)
axins_0.set_xticks([])
axins_0.yaxis.tick_right()
axins_0.set_xlim(6.3, 6.9)
axins_0.set_ylim(29.2, 29.8)
mark_inset(axes_sys[0], axins_0, loc1=2, loc2=4, fc="none", ec="gray")

axes_sys[0].set_xticklabels([])

# Plot 2
# axes[1].plot(time_state, D_traj_opt, label='distance (OPT)')
plotter_kernel(axes_sys[1], time, xa_traj[:, :, 1] - 1.8*xa_traj[:, :, 0], 
               legend_1, my_linewidth, mygreen, linestyle_1)
plotter_kernel(axes_sys[1], time, xn_traj[:, :, 1] - 1.8*xn_traj[:, :, 0], 
               legend_2, my_linewidth, myblue, linestyle_2)
plotter_kernel(axes_sys[1], time, x_traj[:, :, 1] - 1.8*x_traj[:, :, 0],
               legend_3, my_linewidth, myred, linestyle_3)
plotter_kernel(axes_sys[1], time, xs_traj[:, :, 1] - 1.8*xs_traj[:, :, 0],
               legend_4, my_linewidth, mypurple, linestyle_4)
# axes_sys[1].axhline(y=para_star[1], label = 'true velocity', color='r', linestyle='--', linewidth=2)
# axes_sys[1].legend()
axes_sys[1].set_facecolor((0.95, 0.95, 0.95))
axes_sys[1].set_ylabel(r'$d - 1.8v$ [m]')
axes_sys[1].grid(True, linestyle='--', color='white', linewidth=1)

axins_1 = zoomed_inset_axes(axes_sys[1], zoom = 8, loc='upper right')
plotter_kernel(axins_1, time, xa_traj[:, :, 1] - 1.8*xa_traj[:, :, 0], 
               legend_1, my_linewidth, mygreen, linestyle_1)
plotter_kernel(axins_1, time, xn_traj[:, :, 1] - 1.8*xn_traj[:, :, 0], 
               legend_2, my_linewidth, myblue, linestyle_2)
plotter_kernel(axins_1, time, x_traj[:, :, 1] - 1.8*x_traj[:, :, 0],
               legend_3, my_linewidth, myred, linestyle_3)
plotter_kernel(axins_1, time, xs_traj[:, :, 1] - 1.8*xs_traj[:, :, 0],
               legend_4, my_linewidth, mypurple, linestyle_4)
axins_1.set_xticks([])
axins_1.set_xlim(6.3, 6.9)
axins_1.set_ylim(3, 6.5)
mark_inset(axes_sys[1], axins_1, loc1=2, loc2=4, fc="none", ec="gray")

axes_sys[1].set_xticklabels([])

# Plot 3
plotter_kernel(axes_sys[2], time_s, u_asf_traj[:, :, 0],
               legend_1, my_linewidth, mygreen, linestyle_1)
plotter_kernel(axes_sys[2], time_s, u_nsf_traj[:, :, 0],
               legend_2, my_linewidth, myblue, linestyle_2)
plotter_kernel(axes_sys[2], time_s, u_traj[:, :, 0],
               legend_3, my_linewidth, myred, linestyle_3)
plotter_kernel(axes_sys[2], time_s, u_ssf_traj[:, :, 0],
               legend_4, my_linewidth, mypurple, linestyle_4)
# axes[1].axhline(y=para_star[1], label = 'true velocity', color='r', linestyle='--', linewidth=2)
# axes_sys[2].legend()
axes_sys[2].set_facecolor((0.95, 0.95, 0.95))
axes_sys[2].grid(True, linestyle='--', color='white', linewidth=1)
axes_sys[2].set_ylabel(r'$u$ [N]')

axins_2 = zoomed_inset_axes(axes_sys[2], zoom = 2.5, loc='lower left')
plotter_kernel(axins_2, time_s, u_asf_traj[:, :, 0], 
               legend_1, my_linewidth, mygreen, linestyle_1)
plotter_kernel(axins_2, time_s, u_nsf_traj[:, :, 0], 
               legend_2, my_linewidth, myblue, linestyle_2)
plotter_kernel(axins_2, time_s, u_traj[:, :, 0],
               legend_3, my_linewidth, myred, linestyle_3)
plotter_kernel(axes_sys[2], time_s, u_ssf_traj[:, :, 0],
               legend_4, my_linewidth, mypurple, linestyle_4)
axins_2.set_xticks([])
axins_2.yaxis.tick_right()
axins_2.set_xlim(7, 8)
axins_2.set_ylim(-5800, -2100)
mark_inset(axes_sys[2], axins_2, loc1=2, loc2=4, fc="none", ec="gray")

formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_powerlimits((-3, 3))  # Force multiplier display if within 10^3 range
axes_sys[2].yaxis.set_major_formatter(formatter)

# Set x-axis label only on the last plot
axes_sys[2].set_xlabel('Time [s]')

fig_sys.tight_layout()
plt.subplots_adjust(hspace=0.3)
fig_sys.savefig('acc_control.pdf', format='pdf', bbox_inches='tight', dpi=300)

# --------- RLS subplots (1 row, 2 columns) -----------
fig_rls, axes_rls = plt.subplots(2, 1, figsize=(4.5, 2))

# Plot 1
# axes[0].plot(time_state, v_traj_opt, label='velocity (OPT)')
plotter_kernel(axes_rls[0], time, para_asf_traj[:, :, 0],
               r'$\hat{\mu}_{\mathrm{aero}}$', my_linewidth, mydarkblue, linestyle_1)
axes_rls[0].axhline(y=para_star[0], label = r'$\mu^\ast_{\mathrm{aero}}$',
                    color=mydarkblue_cop, linestyle=':', linewidth=my_linewidth)
# axes_rls[0].set_title("RLS estimation with SMID")
axes_rls[0].legend()
axes_rls[0].set_facecolor((0.95, 0.95, 0.95))
axes_rls[0].grid(True, linestyle='--', color='white', linewidth=1)
axes_rls[0].set_xticklabels([])

# Plot 2
# axes[1].plot(time_state, D_traj_opt, label='distance (OPT)')
plotter_kernel(axes_rls[1], time, para_asf_traj[:, :, 1],
               r'$\hat{v}_{\mathrm{f}}$', my_linewidth, myleaveyellow, linestyle_1)
axes_rls[1].axhline(y=para_star[1], label = r'$v^\ast_{\mathrm{f}}$',
                    color=myleaveyellow_cop, linestyle=':', linewidth=my_linewidth)
axes_rls[1].legend()
axes_rls[1].set_facecolor((0.95, 0.95, 0.95))
axes_rls[1].grid(True, linestyle='--', color='white', linewidth=1)

axins_1 = zoomed_inset_axes(axes_rls[1], zoom = 2.5, loc='upper center')
plotter_kernel(axins_1, time, para_asf_traj[:, :, 1],
               r'$\hat{v}_{\mathrm{f}}$', my_linewidth, myleaveyellow, linestyle_1)
axins_1.axhline(y=para_star[1], label = r'$v^\ast_{\mathrm{f}}$',
                    color=myleaveyellow_cop, linestyle=':', linewidth=my_linewidth)
axins_1.set_xticks([])
axins_1.yaxis.tick_right()
axins_1.set_xlim(-0.1, 0.5)
axins_1.set_ylim(21.8, 24.8)
mark_inset(axes_rls[1], axins_1, loc1=2, loc2=4, fc="none", ec="gray")

axes_rls[1].set_xlabel('Time [s]')

fig_rls.tight_layout()

# # Plot 3
# axes_rls[2].plot(dt * time, bound_traj, label=r'$\varepsilon_{\theta,t}(1)$')
# # axes[1].axhline(y=para_star[1], label = 'true velocity', color='r', linestyle='--', linewidth=2)
# axes_rls[2].legend()
# axes_rls[2].set_facecolor((0.95, 0.95, 0.95))
# axes_rls[2].grid(True, linestyle='--', color='white', linewidth=1)

# # Set x-axis label only on the last plot
# axes_rls[2].set_xlabel('Time[s]')


# # --------- for reviewer additional plot (1 row, 2 columns) -----------
# fig_rw, axes_rw = plt.subplots(1, 2, figsize=(4.5, 1.5))

# # Plot 1
# # axes[0].plot(time_state, v_traj_opt, label='velocity (OPT)')
# plotter_kernel(axes_rw[0], time_s, E_theta_traj_sf,
#                r'$E_{\theta,t}(x_t)$', my_linewidth, mydarkblue, linestyle_1)
# # axes_rls[0].set_title("RLS estimation with SMID")
# axes_rw[0].legend()
# axes_rw[0].set_facecolor((0.95, 0.95, 0.95))
# axes_rw[0].grid(True, linestyle='--', color='white', linewidth=1)

# axes_rw[0].set_xlabel('Time[s]')

# # Plot 2
# # axes[1].plot(time_state, D_traj_opt, label='distance (OPT)')
# plotter_kernel(axes_rw[1], time, bound_asf_traj,
#                r'$\varepsilon_{\theta,t}(2)$', my_linewidth, myleaveyellow, linestyle_1)
# axes_rw[1].legend()
# axes_rw[1].set_facecolor((0.95, 0.95, 0.95))
# axes_rw[1].grid(True, linestyle='--', color='white', linewidth=1)

# axes_rw[1].set_xlabel('Time[s]')

# fig_rw.tight_layout()

# # # Plot 3
# # axes_rls[2].plot(dt * time, bound_traj, label=r'$\varepsilon_{\theta,t}(1)$')
# # # axes[1].axhline(y=para_star[1], label = 'true velocity', color='r', linestyle='--', linewidth=2)
# # axes_rls[2].legend()
# # axes_rls[2].set_facecolor((0.95, 0.95, 0.95))
# # axes_rls[2].grid(True, linestyle='--', color='white', linewidth=1)

# # # Set x-axis label only on the last plot
# # axes_rls[2].set_xlabel('Time[s]')
# fig_rw.savefig('reviewer.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.subplots_adjust(hspace=0.3)

fig_rls.savefig('acc_estimation.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.show()

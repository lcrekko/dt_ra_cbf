import numpy as np
import matplotlib.pyplot as plt
from simulation_utils.generators import generate_step_reference, generate_noise_reference, rpm_to_rads

# Time parameters
dt = 1e-4 # sampling interval [s]
T  = 10 # total time horizon [s]
N_t = int(T / dt) # total time steps
T_initial = 0.5

# disturbance parameters
load_max = 0.225 # maximum load disturbance (torque) [N*m]
id_max = 0.5 # maximum id disturbance (current) [A]

# Create a Generator with a specific seed (e.g., 42)
rng_load = np.random.default_rng(seed=42)
rng_id = np.random.default_rng(seed=24)

# generate disturbance trajectories
load_disturbance = 2 * generate_noise_reference(N_t, rng_load, t_initial=int((T_initial)/dt), noise_scale=load_max) - load_max # scale to [-load_max, load_max]
id_disturbance = 2 * generate_noise_reference(N_t, rng_id, t_initial=int((T_initial)/dt), noise_scale=id_max) - id_max # scale to [-id_max, id_max]

fig_sys, axes_sys = plt.subplots(2, 1, figsize=(7, 7*5/9))

# Plot 1
axes_sys[0].plot(np.arange(N_t) * dt, load_disturbance, label=r'$w_{\omega}$')
#plt.xlabel('Time (s)')
axes_sys[0].axhline(-load_max, color='gray', linestyle='--', linewidth=0.5)
axes_sys[0].axhline(load_max, color='gray', linestyle='--', linewidth=0.5)
axes_sys[0].set_ylabel('Load Disturbance (N*m)')
axes_sys[0].legend()
axes_sys[0].set_title('Disturbance Visualization')
axes_sys[0].set_facecolor((0.95, 0.95, 0.95))
axes_sys[0].grid(True, linestyle='--', color='white', linewidth=1)

# Plot 2
# axes[1].plot(time_state, D_traj_opt, label='distance (OPT)')
axes_sys[1].plot(np.arange(N_t) * dt, id_disturbance, label=r'$w_{i_d}$')
axes_sys[1].axhline(-id_max, color='gray', linestyle='--', linewidth=0.5)
axes_sys[1].axhline(id_max, color='gray', linestyle='--', linewidth=0.5)
axes_sys[1].set_ylabel('Id Disturbance (A)')
axes_sys[1].legend()
axes_sys[1].set_facecolor((0.95, 0.95, 0.95))
axes_sys[1].grid(True, linestyle='--', color='white', linewidth=1)

plt.tight_layout()
plt.show()
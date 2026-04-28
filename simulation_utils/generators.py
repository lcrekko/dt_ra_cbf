"""
This provides many trajectory generators

(1) reference generators, etc.

"""

import numpy as np

def generate_step_reference(T, t_vec, a_vec):
    """
    Generate a step reference signal based on the provided time stamps and values.
        :param T: int, total time horizon
        :param t_vec: list of time stamps where the reference changes (e.g., [0, 10, 20])
        :param a_vec: list of reference values corresponding to each time stamp (e.g., [1, 0.5, 0])
        :return: np.ndarray, the generated reference signal of length T
    """
    # --- Inner Checks ---
    # 1. Check if the last time stamp exceeds the horizon
    if t_vec and t_vec[-1] > T - 1:
        raise ValueError(f"Time stamp {t_vec[-1]} exceeds horizon T-1 ({T-1}).")
    
    # 2. Check if the inputs have matching lengths
    if len(t_vec) != len(a_vec):
        raise ValueError(f"Mismatch: t_vec has {len(t_vec)} elements, but a_vec has {len(a_vec)}.")
        
    # 3. Optional: Check if time stamps are strictly increasing
    if not all(x < y for x, y in zip(t_vec, t_vec[1:])):
        raise ValueError("Time stamps in t_vec must be strictly increasing.")

    # --- Generation Logic ---
    ref = np.zeros(T)
    boundaries = t_vec + [T] 
    
    for i, val in enumerate(a_vec):
        start_idx = boundaries[i]
        end_idx = boundaries[i+1]
        ref[start_idx:end_idx] = val
        
    return ref

import numpy as np

def generate_noise_simulations(T, N_sim, rng, t_initial=0, t_final=None, noise_scale=1.0):
    """
    Generate N_sim noise reference signals of length T.
    
    :param T: int, total time horizon
    :param N_sim: int, number of noise realizations (simulations)
    :param rng: np.random.Generator, a random number generator instance
    :param t_initial: int, starting time index (default: 0)
    :param t_final: int, ending time index (default: T)
    :param noise_scale: float, scaling factor (default: 1.0)
    :return: np.ndarray, 2-D array of shape (N_sim, T)
    """
    if t_final is None:
        t_final = T
        
    if t_initial < 0 or t_final > T:
        raise ValueError(f"Indices out of bounds: t_initial={t_initial}, t_final={t_final}, T={T}.")
    
    # Initialize a 2-D array of zeros: (Rows = Simulations, Cols = Time)
    noise_ref = np.zeros((N_sim, T))
    
    # Calculate the duration of the noise
    duration = t_final - t_initial
    
    # Fill the slice for all simulations at once
    # rng.random can take a tuple for size: (N_sim, duration)
    noise_ref[:, t_initial:t_final] = noise_scale * rng.random((N_sim, duration))
    
    return noise_ref

def rpm_to_rads(rpm):
    """
    Convert revolutions per minute (RPM) to radians per second (rad/s).
        :param rpm: float or np.ndarray, the input RPM value(s)
        :return: float or np.ndarray, the corresponding rad/s value(s)
        Note: 1 RPM = 2 * pi / 60 rad/s, so the
    """
    # Multiplication factor is approx 0.1047
    return rpm * (2 * np.pi / 60)
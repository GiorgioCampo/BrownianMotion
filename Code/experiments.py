import numpy as np
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from plotting import *
from utilities import *

def simulate_particles(N, x_0, dt, tau, k_b, m, T, p = 1000, f = lambda x: 0, 
                       barrier = 0, cap = 1e6, return_times = False):
    x = np.zeros((p, N))
    x[:, 0] = x_0
    times = np.ones(p) * N

    for k in tqdm(range(p)):
        first_time = False
        for i in range(1, N):
            x_new = (x[k, i - 1] + dt * tau / m * f(x[k, i - 1]) +
                np.sqrt(2 * k_b * T * tau / m) * np.sqrt(dt) * np.random.randn())
            if np.abs(x_new) > barrier and not first_time:
                # print(f"Particle {k} escaped at time {i}")
                # print(f"{x_new}, {f(x[k, i - 1])}")
                times[k] = i
                first_time = True
            if np.abs(x_new) < cap:
                x[k, i] = x_new
            else:
                x[k, i] = cap if x_new > 0 else -cap
            
    if return_times:
        return x, times
    return x

def msd_fit(x, N, t_0, dt):
    p = x.shape[0]
    x_2 = np.array([sum(x[:, i] ** 2) / p for i in range(N)])

    # Line fit
    b, a = np.polyfit(t_0 + dt * np.arange(N), x_2, 1)
    print(f"Estimated line: {a:.2f} + {b:.2f} * t")

    return x_2, a, b

def T_D_relation(N, x_0, t_0, dt, tau, k_b, m, T):
    T_s = np.logspace(1, 4, 5)

    sim_Ds = np.zeros(len(T_s))
    calc_Ds = np.zeros(len(T_s))

    for k, T in enumerate(T_s):
        x = simulate_particles(N, x_0, dt, tau, k_b, m, T)
        # Mean square displacement
        x_2, a, b = msd_fit(x, N, t_0, dt)
        # Diffusion constant D
        theorical_D = 2 * tau * k_b * T / m

        sim_Ds[k] = b
        calc_Ds[k] = theorical_D

    plot_lines([T_s, T_s], [sim_Ds, calc_Ds], 'D_vs_T', ['k', 'r'], ['-', '--'], 
               ['Simulation D', 'Theorical D'], ['o', None], 'T', 'D', loglog=True)
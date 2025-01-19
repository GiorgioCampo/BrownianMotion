from experiments import *


def first_experiment():
    # First experiment: no force
    T = 1
    tau = 1
    dt = 0.01
    N = 1000

    x = simulate_particles(N, x_0, dt, tau, k_b, m, T)
    time = np.arange(N) * dt
    plot_simulation(time, x[:100, :], 'no_force')

    # Mean square displacement
    x_2, a, b = msd_fit(x, N, t_0, dt)
    # Diffusion constant D
    theorical_D = tau * k_b * T / m
    measured_D = b / 2
    print(f"Measured D: {measured_D:.6f}\nTheorical D: {theorical_D:.6f}")

    plot_msd(x_2, (a, b), N, t_0, dt, 'no_force_D')

    # Relation between D and T
    T_D_relation(N, x_0, t_0, dt, tau, k_b, m, T)
    
def second_experiment():
    # Second experiment: potential force, A = 0
    p = 1000
    N = 1000
    dt = 0.5
    tau = 1
    A = 0
    k = 1
    dV_x = lambda x: dV(k, x, A)

    # Temperatures
    T_s = [10, 50, 100]
    limit_distributions = np.zeros((p, len(T_s)))

    for j, T in enumerate(T_s):
        x = simulate_particles(N, x_0, dt, tau, k_b, m, T, f=dV_x)
        # plot_simulation(x, f"A_{A}_k_{j}_T_{T}")
        limit_distributions[:, j] = x[:, -1]

    plot_distributions(limit_distributions, "limit_distributions", xlabel="x(t)", 
                       ylabel="p(x; T)", hist=False, labels=[f"T = {t:.2f}" for t in T_s])

def third_experiment():
    # Third experiment: potential force, A = 1
    p = 1000
    N = 1000
    dt = 0.5
    tau = 1
    A = 1

    k_s = np.linspace(1, 4, 4)
    max_potential_0 = V(k_s[0], np.sqrt(k_s[0] / (A * 4)), A)
    # print(max_potential_0 * 1e-1)
    T_s = np.linspace(max_potential_0, max_potential_0 * 5, 4)
    
    time_means = list()
    barriers = list()

    for k in k_s:
        dV_x = lambda x: dV(k, x, A)
        barrier = np.sqrt(k / (A * 4))
        barrier_height = V(k, barrier, A)
        barriers.append(barrier_height)
        # x_space = np.linspace(-(barrier * 1.01), (barrier * 1.01), 1000)
        # delta = max(dV_x(x_space))

        times_stats = np.zeros((p, len(T_s))) # * N

        for j, T in enumerate(T_s):
            x, times = simulate_particles(N, x_0, dt, tau, k_b, m, T, p, f=dV_x,
                                           barrier=barrier, return_times=True)
            # plot_simulation(x, "test")
            times_stats[:, j] = sorted(times * dt)
        # trunc = int((N-200))
        plot_distributions(times_stats, f"times_k_{k:.2f}", xlabel=r'$\tau_1$', 
                           min_val=0, max_val=N * dt / 10, ylabel=r"p($\tau_1$; T)", 
                           hist=False, labels=[f"T = {t:.2f}" for t in T_s])
        
        time_means.append(np.mean(times_stats, axis=0))

    pow = 1
    # improve readability with **kwargs
    plot_lines([T_s]*(len(k_s) + 1), time_means + [1 / (T_s ** pow)], "time_means", ['k']*len(k_s) + ['r'], 
               ['-', '--', ':', '-.', '-'], [f"$\Delta = {b:.2f}$" for b in barriers] + [f'x$^-{pow}$'], 
               ['o']*(len(k_s) + 1), 'T', r'$\langle \tau_1 \rangle$', loglog=True)
    
    time_means_delta = list(np.array(time_means).T)

    plot_lines([barriers]*(len(T_s) + 1), time_means_delta + [1 / (np.array(barriers) ** -pow)], 
               "time_means_delta", ['k']*len(T_s) + ['r'], ['-', '--', ':', '-.', '-'], 
               [f"T = {t:.2f}" for t in T_s] + [f'x$^{pow}$'], 
               ['o']*(len(T_s) + 1), '$\Delta$', r'$\langle \tau_1 \rangle$', loglog=True)
        

def main():
    np.random.seed(6184)
    print("Running first experiment\n")
    first_experiment()
    print("\nRunning second experiment\n")
    second_experiment()
    print("\nRunning third experiment\n")
    third_experiment()

    return 0


if __name__ == "__main__":
    # Initial conditions
    t_0 = 0
    x_0 = 0
    k_b = 1
    m = 1

    # Potential
    V = lambda k, x, A: k * x**2 / 2 - A * x**4
    dV = lambda k, x, A: - (k * x - 4 * A * x**3)
    
    main()
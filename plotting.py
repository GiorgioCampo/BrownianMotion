import numpy as np
import matplotlib.pyplot as plt

from scipy import stats

# Modify font size
from matplotlib import rc
rc('font', size=20, family='serif')

show_plots = False

def plot_simulation(time, x, namefig, ylim = None ):
    p = x.shape[0]

    plt.figure(figsize = (12, 8))
    for k in range(p):
        plt.plot(time, x[k, :], color='k', alpha=1 / p * k)
    
    plt.plot(time, np.mean(x, axis=0), color='r', linewidth=2, 
             label=r'$\langle$x(t)$\rangle$')
    if ylim:
        plt.ylim(ylim)
    plt.xlabel('t')
    plt.ylabel('x(t)')
    plt.legend()
    plt.savefig(f"./Images/{namefig}.pdf")
    plt.show() if show_plots else None

def plot_msd(msd, fitted_msd, N, t_0, dt, namefig):
    a, b = fitted_msd
    plt.figure(figsize = (12, 8))
    plt.plot(t_0 + dt * np.arange(N), msd, label=r'$\langle$x$^2$(t)$\rangle$', 
             c='k')
    plt.plot(t_0 + dt * np.arange(N), b * (t_0 + dt * np.arange(N)) + a, 
             label=f'{a:.2f} + {b:.2f}t', c='r', linestyle='--')
    plt.xlabel('t')
    plt.ylabel(r'$\langle$x$^2$(t)$\rangle$')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"./Images/{namefig}.pdf")
    plt.show() if show_plots else None

def plot_lines(xs, lines, namefig, colors, styles, labels, markers, 
               x_label = None, y_label = None, loglog = False):
    plt.figure(figsize = (12, 8))
    for k in range(len(lines)):
        if loglog:
            plt.loglog(xs[k],lines[k], c=colors[k], ls=styles[k], label=labels[k],
                       marker=markers[k])
        else:
            plt.plot(xs[k], lines[k], c=colors[k], ls=styles[k], label=labels[k],
                     marker=markers[k])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"./Images/{namefig}.pdf")
    plt.show() if show_plots else None

def plot_distributions(data, namefig, xlabel, ylabel, min_val=None, max_val=None, 
                       labels=None, kde=True, hist=True, bins=100, figsize=(12, 6)):
    """
    Plot multiple distributions on the same axes, with both histogram and KDE options.
    
    Parameters:
    -----------
    data : numpy.ndarray
        2D array where each column is a distribution to plot
    labels : list, optional
        Labels for each distribution (default: Distribution 1, 2, etc.)
    kde : bool, optional
        Whether to plot kernel density estimation (default: True)
    hist : bool, optional
        Whether to plot histogram (default: True)
    bins : int, optional
        Number of bins for histogram (default: 50)
    figsize : tuple, optional
        Figure size (width, height) in inches (default: (12, 6))
        
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    N = data.shape[1]
    
    # Generate labels if not provided
    if labels is None:
        labels = [f'Distribution {i+1}' for i in range(data.shape[1])]
    
    # Calculate common range for all distributions
    min_val = min_val or np.min(data)
    max_val = max_val or np.max(data)
    x_range = np.linspace(min_val, max_val, 200)
    
    # Plot each distribution
    for i in range(N):
        column_data = data[:, i]
        
        if hist:
            # Plot histogram with transparency
            ax.hist(column_data, bins=bins, density=True, color='k', 
                    alpha=1 / N * i + 0.1)
        
        if kde:
            # Calculate and plot KDE
            kde = stats.gaussian_kde(column_data)
            ax.plot(x_range, kde(x_range), 
                   label=f'{labels[i]}',
                   color='k',
                   alpha=1 / N * i + 0.1,
                   linewidth=2)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"./Images/{namefig}.pdf")
    plt.show() if show_plots else None
import matplotlib.pyplot as plt
import numpy as np

def show_statistical_plots(x_vals, n_vals, w_vals):
    """
    Будує три графіки в одному вікні:
    1. Полігон частот
    2. Полігон відносних частот
    3. Графік емпіричної функції розподілу F*(x)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # 1. Полігон частот
    axes[0].plot(x_vals, n_vals, marker='o', color='royalblue', linewidth=1.5, markersize=6)
    axes[0].set_title('Полігон частот', fontsize=11)
    axes[0].set_xlabel('$x_i$')
    axes[0].set_ylabel('$n_i$')
    axes[0].set_xticks(x_vals)
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # 2. Полігон відносних частот
    axes[1].plot(x_vals, w_vals, marker='s', color='forestgreen', linewidth=1.5, markersize=6)
    axes[1].set_title('Полігон відносних частот', fontsize=11)
    axes[1].set_xlabel('$x_i$')
    axes[1].set_ylabel(r'$\omega_i$')
    axes[1].set_xticks(x_vals)
    axes[1].grid(True, linestyle='--', alpha=0.6)

    # 3. Емпірична функція розподілу F*(x)
    cum_w = np.cumsum(w_vals)
    
    # Побудова східчастого графіка (горизонтальні відрізки)
    for i in range(len(x_vals)):
        x_start = x_vals[i - 1] if i > 0 else x_vals[0] - 1
        x_end = x_vals[i]
        val = cum_w[i - 1] if i > 0 else 0
        axes[2].hlines(val, x_start, x_end, colors='crimson', linewidth=2)
        
    # Останній відрізок F*(x) = 1 для x > x_max
    axes[2].hlines(1.0, x_vals[-1], x_vals[-1] + 1, colors='crimson', linewidth=2)

    axes[2].set_title('Емпірична функція розподілу $F^*(x)$', fontsize=11)
    axes[2].set_xlabel('$x$')
    axes[2].set_ylabel('$F^*(x)$')
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()
import numpy as np
import matplotlib.pyplot as plt
from problimite import solve_boundary_problem


def y_exact(x):
    c = 0.4
    d = 0.81
    return (c - 0.4 / x**2) - (c - 0.4 / d) * np.log(x) / np.log(0.9)


def run_simulation(h):
    a, b = 0.9, 1.0

    N = int(round((b - a) / h))
    x = np.linspace(a, b, N + 1)
    P = -1.0 / x
    Q = np.zeros_like(x)
    R = -1.6 / x**4
    y = solve_boundary_problem((b - a) / N, P, Q, R, a, b, alpha=0.0, beta=0.0)
    return x, y

#  --- Figure 1
if __name__ == '__main__':

    h1, h2 = 1/30, 1/100

    x1, y1 = run_simulation(h1)

    x2, y2 = run_simulation(h2)

    hd = np.linspace(0.9, 1.0, 500)
    yd = y_exact(hd)

    plt.figure(figsize=(8, 6))
    plt.plot(hd, yd, color='gray', linewidth=2, label='y exact')
  
    plt.plot(x1, y1, color='tab:blue', marker='o', markersize=6,
             linewidth=2, label='approximation avec h/30')
 
    plt.plot(x2, y2, color='tab:red', marker='o', linestyle='--',
             markersize=6, linewidth=2, label='approximation avec h/100')
    plt.title('Figure 1 - y en fonction de h', fontsize=14, fontweight='bold')
    plt.xlabel('h')
    plt.ylabel('y')
    plt.xlim(0.9, 1.0)
    plt.ylim(0, 0.0025)
    plt.xticks(np.linspace(0.9, 1.0, 6))
    plt.yticks(np.linspace(0, 0.0025, 6))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(loc='lower center', fontsize=10)
    plt.tight_layout()
    plt.show()

    # --- Figure 2 
    hs = [1e-5, 1e-4, 1e-3, 1e-2]
    errors = []
    for h in hs:
        x, y_num = run_simulation(h)
        xi = x[1:-1]
        errors.append(np.max(np.abs(y_num[1:-1] - y_exact(xi))))

    plt.figure(figsize=(8, 6))
    plt.loglog(hs, errors, color='tab:green', marker='o', markersize=8,
               linewidth=2)
    plt.title('Figure 2 - Erreur en fonction de h', fontsize=14, fontweight='bold')
    plt.xlabel('h')
    plt.ylabel('E(h)')
    plt.xlim(1e-5, 1e-2)
    plt.ylim(1e-13, 1e-7)
    plt.xticks([1e-5, 1e-4, 1e-3, 1e-2],
               ['10$^{-5}$', '10$^{-4}$', '10$^{-3}$', '10$^{-2}$'])
    plt.yticks([1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7],
               ['10$^{-12}$', '10$^{-11}$', '10$^{-10}$', '10$^{-9}$', '10$^{-8}$', '10$^{-7}$'])
    plt.minorticks_on()
    plt.grid(which='major', linestyle='--', linewidth=0.7, alpha=0.7)
    plt.grid(which='minor', linestyle=':', linewidth=0.3, alpha=0.5)
    plt.tight_layout()
    plt.show()
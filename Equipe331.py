# equipeX.py
import numpy as np
import matplotlib.pyplot as plt
from problimite import solve_boundary_problem

def p(x):
    return -1 / x

def q(x):
    return 0

def r(x):
    return -1.6 / x**4

def y_exact(x):
    c = 1
    d = 1
    return (c - 0.4 / x**2) - (c - 0.4 / d) * np.log(x) / np.log(0.9)

def run_simulation(h):
    a, b = 0.9, 1.0
    alpha, beta = 0, 0
    N = int((b - a) / h) - 1
    x_nodes = np.linspace(a + h, b - h, N)
    P = p(x_nodes)
    Q = q(x_nodes)
    R = r(x_nodes)
    y_approx = solve_boundary_problem(h, P, Q, R, a, b, alpha, beta)
    return x_nodes, y_approx

def plot_solutions():
    for h in [1/30, 1/100]:
        x_nodes, y_approx = run_simulation(h)
        y_true = y_exact(x_nodes)
        plt.plot(x_nodes, y_approx, label=f"Approx h={h:.5f}")
    x_full = np.linspace(0.9, 1.0, 1000)
    plt.plot(x_full, y_exact(x_full), label="Exact", linestyle="--")
    plt.xlabel("x")
    plt.ylabel("y(x)")
    plt.title("Comparaison des solutions")
    plt.legend()
    plt.grid()
    plt.show()

def plot_error():
    hs = [10**(-i) for i in range(2, 6)]
    errors = []
    for h in hs:
        x_nodes, y_approx = run_simulation(h)
        y_true = y_exact(x_nodes)
        error = np.max(np.abs(y_approx - y_true))
        errors.append(error)
    plt.loglog(hs, errors, marker='o')
    plt.xlabel("h")
    plt.ylabel("E(h)")
    plt.title("Erreur maximale vs pas h")
    plt.grid(True, which="both", ls="--")
    plt.show()

if __name__ == "__main__":
    plot_solutions()
    plot_error()
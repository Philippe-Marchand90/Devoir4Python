import numpy as np
import matplotlib.pyplot as plt
from problimite import problimite

def p(x):
    return 1 / x

def q(x):
    return 0 * x  # q(x) = 0

def r(x):
    return -1.6 / x**4

def solution_exacte(x):
    d = 0.9**2
    c = 0.4 / d
    return (c - 0.4 / x**2) - (c - 0.4 / d) * (np.log(x) / np.log(0.9))

def resoudre(h):
    a, b = 0.9, 1.0
    N = int((b - a) / h)
    x = np.linspace(a, b, N + 2)

    P = p(x)
    Q = q(x)
    R = r(x)

    y_num = problimite(h, P, Q, R, a, b, 0, 0)
    y_exact = solution_exacte(x)

    erreur_max = np.max(np.abs(y_num - y_exact))
    return x, y_num, y_exact, erreur_max

# --- Partie a : Comparaison solutions ---
x1, y1, y_ex1, _ = resoudre(1/30)
x2, y2, y_ex2, _ = resoudre(1/100)

plt.figure(figsize=(8, 5))
plt.plot(x1, y1, label='h = 1/30')
plt.plot(x2, y2, label='h = 1/100')
plt.plot(x1, y_ex1, '--', label='Solution exacte')
plt.xlabel('x')
plt.ylabel('y(x)')
plt.title("Comparaison des solutions numériques et exactes")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("fig_comparaison.png")
plt.show()

# --- Partie b : Erreur max E(h) en log-log ---
hs = [1e-2, 1e-3, 1e-4, 1e-5]
errors = []

for h in hs:
    _, _, _, err = resoudre(h)
    errors.append(err)

plt.figure(figsize=(8, 5))
plt.loglog(hs, errors, 'o-', label='Erreur max E(h)')
plt.xlabel("h (pas)")
plt.ylabel("Erreur maximale")
plt.title("Erreur en fonction du pas h (échelle log-log)")
plt.grid(True, which="both", linestyle="--")
plt.legend()
plt.tight_layout()
plt.savefig("fig_erreur_loglog.png")
plt.show()
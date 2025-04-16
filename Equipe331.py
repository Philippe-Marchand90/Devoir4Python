import numpy as np
import matplotlib.pyplot as plt
from problimite import problimite

# Définir les fonctions p(x), q(x), r(x)
def p(x):
    return 1 / x

def q(x):
    return 0 * x

def r(x):
    return -1.6 / x**4

# Solution exacte de l'équation
def solution_exacte(x):
    d = 0.9**2
    c = 0.4 / d  # impose y(0.9) = 0
    return (c - 0.4 / x**2) - (c - 0.4 / d) * (np.log(x) / np.log(0.9))

# Fonction pour calculer la solution numérique et exacte pour un pas donné
def resoudre(h):
    a, b = 0.9, 1
    N = int((b - a) / h)
    x = np.linspace(a, b, N + 2)

    P = p(x)
    Q = q(x)
    R = r(x)

    y_num = problimite(h, P, Q, R, a, b, 0, 0)
    y_exact = solution_exacte(x)

    erreur_max = np.max(np.abs(y_num - y_exact))

    return x, y_num, y_exact, erreur_max

# --- Partie a : Comparaison des solutions pour deux valeurs de h ---
x1, y1, y_ex1, _ = resoudre(1 / 30)
x2, y2, y_ex2, _ = resoudre(1 / 100)

plt.figure(figsize=(8, 5))
plt.plot(x1, y1, label='h = 1/30')
plt.plot(x2, y2, label='h = 1/100')
plt.plot(x1, y_ex1, '--', label='Solution exacte')
plt.xlabel("x")
plt.ylabel("y(x)")
plt.title("Comparaison solutions numérique / exacte")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("fig_comparaison.png")
plt.show()

# --- Partie b : Graphe de l’erreur max E(h) ---
hs = [1e-2, 1e-3, 1e-4, 1e-5]
erreurs = []

for h in hs:
    _, _, _, erreur = resoudre(h)
    erreurs.append(erreur)

plt.figure(figsize=(8, 5))
plt.loglog(hs, erreurs, 'o-', label='Erreur E(h)')
plt.xlabel("h")
plt.ylabel("Erreur maximale")
plt.title("Erreur en fonction du pas h (log-log)")
plt.grid(True, which="both", linestyle="--")
plt.legend()
plt.tight_layout()
plt.savefig("fig_erreur_loglog.png")
plt.show()
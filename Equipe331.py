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
    d = 0.9**2  # d = 0.81
    c = 0.4 / d  # impose y(0.9) = 0
    return (c - 0.4 / x**2) - (c - 0.4 / d) * (np.log(x) / np.log(0.9))

def resoudre_et_tracer(h, tracer=True):
    a = 0.9
    b = 1.0
    N = int((b - a) / h)
    x = np.linspace(a, b, N + 2)

    P = p(x)
    Q = q(x)
    R = r(x)

    y_num = problimite(h, P, Q, R, a, b, 0, 0)
    y_exact = solution_exacte(x)

    if tracer:
        plt.plot(x, y_num, label=f'Solution numérique h={h:.5f}')
        plt.plot(x, y_exact, '--', label='Solution exacte')

    erreur_max = np.max(np.abs(y_num - y_exact))
    return h, erreur_max

# --- Partie a : Tracé pour h = 1/30 et h = 1/100 ---
plt.figure()
resoudre_et_tracer(1/30)
resoudre_et_tracer(1/100)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Comparaison solution numérique / exacte")
plt.legend()
plt.grid()
plt.savefig("fig_comparaison.png")
plt.show()

# --- Partie b : Tracé de l'erreur E(h) en log-log ---
hs = [1e-2, 1e-3, 1e-4, 1e-5]
erreurs = []

for h in hs:
    _, err = resoudre_et_tracer(h, tracer=False)
    erreurs.append(err)

plt.figure()
plt.loglog(hs, erreurs, 'o-', label='Erreur max E(h)')
plt.xlabel("h")
plt.ylabel("Erreur max")
plt.title("Erreur en fonction de h (échelle log-log)")
plt.grid(True, which="both", ls="--")
plt.legend()
plt.savefig("fig_erreur_loglog.png")
plt.show()
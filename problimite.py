import numpy as np
from tridiagonal import tridiagonal

def problimite(h, P, Q, R, a, b, alpha, beta):
    N = len(P) - 2  # nombre de points intérieurs

    D = np.zeros(N)         # diagonale principale
    I = np.zeros(N - 1)     # diagonale inférieure
    S = np.zeros(N - 1)     # diagonale supérieure
    b_vec = np.zeros(N)     # vecteur de droite

    for i in range(N):
        pi = P[i + 1]
        qi = Q[i + 1]
        ri = R[i + 1]

        D[i] = 2 + qi * h**2
        b_vec[i] = -ri * h**2

        if i > 0:
            I[i - 1] = -1 - pi * h / 2
        if i < N - 1:
            S[i] = -1 + pi * h / 2

    # Ajout des conditions de bord
    b_vec[0] += (1 + P[1] * h / 2) * alpha
    b_vec[-1] += (1 - P[N] * h / 2) * beta

    # Résolution du système
    y_inner = tridiagonal(D, I, S, b_vec)

    # Assemblage du vecteur y complet (avec bornes)
    y = np.concatenate(([alpha], y_inner, [beta]))
    return y
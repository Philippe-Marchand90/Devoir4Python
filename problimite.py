# problimite.py
import numpy as np
from tridiagonal import tridiagonal

def solve_boundary_problem(h, P, Q, R, a, b, alpha, beta):
    """
    Résout y''(x) - p(x) y'(x) - q(x) y(x) = r(x) sur [a,b]
    par différences finies centrées avec pas h, conditions y(a)=alpha, y(b)=beta.

    Arguments:
      h     : pas de maillage
      P, Q, R : tableaux de longueur N+2 des valeurs p(x_i), q(x_i), r(x_i) aux noeuds
      a, b  : bornes de l'intervalle
      alpha, beta : conditions aux limites

    Retourne:
      y : tableau de longueur N+2 contenant la solution approchée aux noeuds.
    """
    # Nombre de mailles intérieures
    N = len(P) - 2

    # initialise diagonales et second membre
    D = np.zeros(N)
    I = np.zeros(N-1)
    S = np.zeros(N-1)
    b_vec = np.zeros(N)

    # remplir les coefficients de la matrice tridiagonale et le vecteur RHS
    for i in range(N):
        pi = P[i+1]
        qi = Q[i+1]
        ri = R[i+1]
        D[i] = 2 + qi * h**2
        b_vec[i] = - ri * h**2
        if i > 0:
            I[i-1] = -1 - pi * h / 2
        if i < N-1:
            S[i] = -1 + pi * h / 2

    # appliquer les conditions aux limites dans b_vec
    b_vec[0]     += (1 + P[1] * h / 2) * alpha
    b_vec[-1]    += (1 - P[N] * h / 2) * beta

    # résolution du système tridiagonal
    y_inner = tridiagonal(D, I, S, b_vec)

    # reconstitution du vecteur complet avec conditions aux limites
    y = np.empty(N+2)
    y[0]       = alpha
    y[1:-1]    = y_inner
    y[-1]      = beta
    return y

if __name__ == '__main__':
    # simple test: résout y'' = -1 sur [0,1] avec y(0)=y(1)=0 => y(x)=x(1-x)/2
    a, b = 0.0, 1.0
    alpha, beta = 0.0, 0.0
    h = 0.1
    x = np.arange(a, b+h, h)
    P = np.zeros_like(x)
    Q = np.zeros_like(x)
    R = -np.ones_like(x)
    y_approx = solve_boundary_problem(h, P, Q, R, a, b, alpha, beta)
    y_exact  = x*(1-x)/2
    print("Max error test:", np.max(np.abs(y_exact - y_approx)))
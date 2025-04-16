import numpy as np
from tridiagonal import solve_tridiagonal

def solve_boundary_problem(h, P, Q, R, a, b, alpha, beta):
    N = len(P)
    
    D = np.zeros(N)
    I = np.zeros(N-1)
    S = np.zeros(N-1)
    rhs = np.zeros(N)

    for i in range(N):
        pi, qi, ri = P[i], Q[i], R[i]
        D[i] = 2 + h**2 * qi
        if i != 0:
            I[i-1] = -1 - h * pi / 2
        if i != N - 1:
            S[i] = -1 + h * pi / 2
        rhs[i] = -h**2 * ri

    rhs[0] += (1 + h * P[0] / 2) * alpha
    rhs[-1] += (1 - h * P[-1] / 2) * beta

    return solve_tridiagonal(D, I, S, rhs)
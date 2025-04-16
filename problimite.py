import numpy as np
from tridiagonal import tridiagonal

def problimite(h, P, Q, R, a, b, alpha, beta):
    N = len(P) - 2
    D = np.zeros(N)
    I = np.zeros(N - 1)
    S = np.zeros(N - 1)
    b_vec = np.zeros(N)

    for i in range(1, N + 1):  # i de 1 à N (inclus)
        pi = P[i]
        qi = Q[i]
        ri = R[i]
        
        D[i-1] = 2 + qi * h**2
        if i != 1:
            I[i-2] = -1 - pi * h / 2
        if i != N:
            S[i-1] = -1 + pi * h / 2
        
        b_vec[i-1] = -ri * h**2
    
    b_vec[0] += (1 + P[1] * h / 2) * alpha
    b_vec[-1] += (1 - P[N] * h / 2) * beta
    y_inner = tridiagonal(D, I, S, b_vec)
    y = np.concatenate(([alpha], y_inner, [beta]))
    return y
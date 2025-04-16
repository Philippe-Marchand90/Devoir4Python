import numpy as np
from tridiagonal import tridiagonal

def problimite(h, P, Q, R, a, b, alpha, beta):
    N = len(P) - 2
    D = np.zeros(N)
    I = np.zeros(N - 1)
    S = np.zeros(N - 1)
    b_vec = np.zeros(N)

    

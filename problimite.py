import numpy as np
from tridiagonal import tridiagonal

def problimite(h, P, Q, R, a, b, alpha, beta):
  
    N = len(P)
    h_sq = h**2
    
    D = 2 + Q * h_sq
    I = -1 - P[1:] * h / 2
    S = -1 + P[:-1] * h / 2
    
    b_vec = -R * h_sq
    b_vec[0] += (1 + P[0] * h / 2) * alpha
    b_vec[-1] += (1 - P[-1] * h / 2) * beta
    
    y_interior = tridiagonal(D, I, S, b_vec)
    
    y_hat = np.zeros(N + 2)
    y_hat[0] = alpha
    y_hat[-1] = beta
    y_hat[1:-1] = y_interior
    
    return y_hat
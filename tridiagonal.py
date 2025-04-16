import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

def solve_tridiagonal(D, I, S, b):
    """
    Résout le système tridiagonal Ax = b
    A est une matrice avec:
      - D sur la diagonale principale
      - I sur la diagonale inférieure
      - S sur la diagonale supérieure
    """
    N = len(D)
    diagonals = [I, D, S]
    A = diags(diagonals, offsets=[-1, 0, 1], format='csr')
    return spsolve(A, b)
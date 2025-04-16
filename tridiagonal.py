import numpy as np
import scipy as sp
import time


def tridiagonal(D, I, S, b):
    diagonals = [I, D, S]
    matrice = sp.sparse.diags_array(diagonals, offsets=[-1, 0, 1]).toearray()
    A = sp.sparse.csr_matrix(matrice)
    return sp.sparse.linalg.spsolve(A, b)
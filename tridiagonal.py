import numpy as np
import time
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve


def solve_tridiagonal(I, D, S, b):
    A = diags([I, D, S], offsets=[-1, 0, 1])
    A_csr = csr_matrix(A)
    x = spsolve(A_csr, b)
    return x

import numpy as np
import time
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve


def solve_tridiagonal(I, D, S, b):

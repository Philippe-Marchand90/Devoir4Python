import numpy as np
import time
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve


def tridiagonal(I, D, S, b):
    A = diags([I, D, S], offsets=[-1, 0, 1])
    A_csr = csr_matrix(A)
    x = spsolve(A_csr, b)
    return x

#test
if __name__ == "__main__":
    N = 15000
    D = 4 * np.ones(N)
    I = 1 * np.ones(N - 1)
    S = 1 * np.ones(N - 1)
    b = np.random.rand(N)
    
    #Resolution avec dense
    print("Résolution avec matrice dense...")
    A_dense = np.diag(D) + np.diag(I, -1) + np.diag(S, 1)
    t1 = time.time()
    x_dense = np.linalg.solve(A_dense, b)
    t2 = time.time()
    print(f"Temps (dense) : {t2 - t1:.4f} secondes")

    #Resolution avec sparse
    print("Résolution avec matrice sparse (csr_matrix)...")
    A_sparse = diags([I, D, S], offsets=[-1, 0, 1])
    A_csr = csr_matrix(A_sparse)
    t3 = time.time()
    x_sparse = spsolve(A_csr, b)
    t4 = time.time()
    print(f"Temps (sparse) : {t4 - t3:.4f} secondes")
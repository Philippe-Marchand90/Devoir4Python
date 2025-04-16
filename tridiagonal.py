import numpy as np
import scipy as sp
import time


def tridiagonal(D, I, S, b):
    diagonals = [I, D, S]
    matrice = sp.sparse.diags_array(diagonals, offsets=[-1, 0, 1]).toearray()
    A = sp.sparse.csr_matrix(matrice)
    return sp.sparse.linalg.spsolve(A, b)


if __name__ == "__main__":
    N = 15000
    D = 4 * np.ones(N)
    I = 1 * np.ones(N - 1)
    S = 1 * np.ones(N - 1)
    b = np.random.rand(N)

    # --- Résolution avec matrice pleine (dense) ---
    print("Test avec matrice dense")
    A_dense = np.diag(D) + np.diag(I, -1) + np.diag(S, 1)
    t1 = time.time()
    x_dense = np.linalg.solve(A_dense, b)
    t2 = time.time()
    print(f"Temps (dense) : {t2 - t1:.4f} secondes")

    # --- Résolution avec matrice creuse (sparse) ---
    print("\nTest avec matrice sparse")
    diagonals = [I, D, S]
    A_sparse = sp.sparse.diags(diagonals, offsets=[-1, 0, 1])
    A_csr = sp.sparse.csr_matrix(A_sparse)
    t3 = time.time()
    x_sparse = sp.sparse.linalg.spsolve(A_csr, b)
    t4 = time.time()
    print(f"Temps (sparse) : {t4 - t3:.4f} secondes")

    # Vérification de la précision
    erreur = np.linalg.norm(x_dense - x_sparse)
    print(f"Erreur entre les deux solutions : {erreur:.2e}")
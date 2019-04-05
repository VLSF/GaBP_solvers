import numpy as np

def BP_solver(A, b, tol, verbose=False, write=False):
    ind = np.where(A != 0)
    L, L_new, mu, mu_new = np.zeros_like(A), np.zeros_like(A), np.zeros_like(A), np.zeros_like(A)
    x = np.zeros_like(b)
    L[ind] += 0.1
    mu[ind] += 0.1
    error = 100
    it_number = 0
    E = []
    while error > tol:
        it_number += 1
        for i, j in zip(ind[0], ind[1]):
            if i != j:
                diagonal = np.ones_like(A[j, :])
                neighbours = np.where(A[j, :] != 0)
                diagonal[neighbours] = A[neighbours, neighbours]
                down = A[j, j] - np.sum(A[j, :]*L[:,j]/diagonal[:]) + L[i,j]*A[j, i]/A[i,i]
                up = b[j] - np.sum(A[j,:]*mu[:, j]/diagonal[:]) + A[j,i]*mu[i, j]/A[i,i]
                L_new[j, i] = A[j,j]*A[j, i]/down
                mu_new[j, i] = A[j, j]*up/down

        L = np.copy(L_new)
        mu = np.copy(mu_new)
        for i in range(len(x)):
            diagonal = np.ones_like(A[i, :])
            neighbours = np.where(A[i, :] != 0)
            diagonal[neighbours] = A[neighbours, neighbours]
            x[i] = (b[i] - np.sum(A[i, :]*mu[:, i]/diagonal))/(A[i, i] - np.sum(A[i, :]*L[:, i]/diagonal))

        error = np.linalg.norm(A @ x - b, ord = np.inf)
        if write:
            E.append(error)
        if verbose:
            print(f'Iteration #{it_number}\n      error = {error:1.2}')

    if write:
        return x, E
    else:
        return x

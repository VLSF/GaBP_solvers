import numpy as np
from scipy.linalg import solve_triangular

def exact_marginalization(A, b):
    inverse = np.linalg.inv(A)
    variances = np.diag(inverse)
    means = (inverse @ b)
    return variances, means

def stripes_BP_solver(A, b, tol=1e-10, verbose=False, write=False):
    N = len(b)
    me, var = np.zeros_like(b), np.zeros_like(b)
    M = int(np.sqrt(N))
    horizontal_regions = np.arange(N).reshape((M,-1))
    m_h = np.zeros((M, M))# [from which horizontal, to which vertical]
    m_h_old = np.zeros((M, M))
    vertical_regions = np.arange(N).reshape((M,-1)).T
    m_v = np.zeros((M, M))# [from which vertical, to which horizontal]
    m_v_old = np.zeros((M, M))

    A_horizontal = np.zeros((M, M, M))
    b_horizontal = np.zeros((M, M))
    A_vertical = np.zeros((M, M, M))
    b_vertical = np.zeros((M, M))

    for i in range(M):
        x, y = np.meshgrid(horizontal_regions[i], horizontal_regions[i], indexing='ij')
        w, z = np.meshgrid(vertical_regions[i], vertical_regions[i], indexing='ij')
        A_horizontal[i] = A[x, y]
        A_vertical[i] = A[w, z]
        b_horizontal[i] = b[horizontal_regions[i]]
        b_vertical[i] = b[vertical_regions[i]]

    k=0
    me = np.zeros_like(b)
    E = []
    E.append(np.linalg.norm(A @ me - b, ord=np.inf))
    error = 1
    while error>tol:
        k+=1

        # horizontal update
        for i in range(M):
            delta_A = np.diag(m_v[:, i])
            variances, _ = exact_marginalization(A_horizontal[i]+delta_A, b_horizontal[i])
            m_h[i, :] = 1/variances - (np.diag(A_horizontal[i]) + m_v[:, i])

        # vertical update
        for i in range(M):
            delta_A = np.diag(m_h[:, i])
            variances, _ = exact_marginalization(A_vertical[i]+delta_A, b_vertical[i])
            m_v[i, :] = 1/variances - (np.diag(A_vertical[i]) + m_h[:, i])

        var = 1/(np.diag(A) + (m_h[:, :]).reshape((-1,)) + (m_v[:, :].T).reshape((-1,)))

        error = np.linalg.norm(m_h - m_h_old, ord=np.inf) + np.linalg.norm(m_v - m_v_old, ord=np.inf)
        m_h_old = m_h; m_v_old = m_v
        if write:
            E.append(error)
        if verbose:
            print(f'Iteration #{k}\n      error = {error:1.2}')
    if write:
        return var, E
    else:
        return var

def exact_diagonal(A, b):
    inv_A = np.linalg.inv(A)
    return np.diag(inv_A), inv_A @ b

def split_BP_solver(A, b, solve=exact_diagonal, tol=1e-10, verbose=False, write=False):
    N = int(np.sqrt(len(b)))
    M = int(N/2) + 1
    grid = np.arange(N**2).reshape((N, N))
    left_grid = grid[:, :M]
    right_grid = grid[:, M-1:].reshape((-1,))
    intersection = left_grid[:, -1]
    left_grid = left_grid.reshape((-1,))
    local_left_intersection = np.arange(M-1, N*M, M)
    local_right_intersection = np.arange(0, N*(N-M+1), N-M+1)
    left_coords = np.meshgrid(left_grid, left_grid, indexing='ij')
    right_coords = np.meshgrid(right_grid, right_grid, indexing='ij')
    local_right_coords = np.meshgrid(local_right_intersection, local_right_intersection, indexing='ij')
    left_A = A[left_coords[0], left_coords[1]]
    left_b = b[left_coords[1][0]]

    right_A = A[right_coords[0], right_coords[1]]
    right_A[local_right_coords[0], local_right_coords[1]] = 0
    right_A[local_right_coords[1][0],local_right_coords[1][0]] = A[intersection, intersection]
    right_b = b[right_coords[1][0]]

    left_to_right_precision = np.zeros_like(intersection) + 0.1
    left_to_right_mean = np.zeros_like(intersection) + 0.1
    right_to_left_precision = np.zeros_like(intersection) + 0.1
    right_to_left_mean = np.zeros_like(intersection) + 0.1

    x = np.zeros_like(b)
    E = []
    E.append(np.linalg.norm(A @ x - b, ord=np.inf))
    sigma = np.zeros_like(b)
    old_sigma = np.zeros_like(b)

    error = 1
    k=0
    while error>tol:
        k+=1
        left_A[local_left_intersection, local_left_intersection] += right_to_left_precision
        L_0, mu_0 = solve(left_A, left_b)
        left_A[local_left_intersection, local_left_intersection] -= right_to_left_precision
        left_to_right_precision = 1/L_0[local_left_intersection] - (right_to_left_precision + left_A[local_left_intersection, local_left_intersection])

        sigma[left_grid] = 1/L_0

        right_A[local_right_intersection, local_right_intersection] += left_to_right_precision
        L_0, mu_0 = solve(right_A, right_b)
        right_A[local_right_intersection, local_right_intersection] -= left_to_right_precision
        right_to_left_precision = 1/L_0[local_right_intersection] - (left_to_right_precision + right_A[local_right_intersection, local_right_intersection])

        sigma[right_grid] = 1/L_0

        error = np.linalg.norm(sigma - old_sigma, ord=np.inf)
        old_sigma = sigma
        if write:
            E.append(error)
        if verbose:
            print(f'Iteration #{k}\n      error = {error:1.2}')
    if write:
        return sigma, E
    else:
        return sigma

def BP_solver(A, b, tol, verbose=False, write=False):
    ind = np.where(A != 0)
    L, L_new, mu, mu_new = np.zeros_like(A), np.zeros_like(A), np.zeros_like(A), np.zeros_like(A)
    var = np.zeros_like(b)
    var_old = np.zeros_like(b)
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
        for i in range(len(b)):
            diagonal = np.ones_like(A[i, :])
            neighbours = np.where(A[i, :] != 0)
            diagonal[neighbours] = A[neighbours, neighbours]
            var[i] = 1/(A[i, i] - np.sum(A[i, :]*L[:, i]/diagonal))

        error = np.linalg.norm(var - var_old, ord = np.inf)
        var_old = var
        if write:
            E.append(error)
        if verbose:
            print(f'Iteration #{it_number}\n      error = {error:1.2}')

    if write:
        return var, E
    else:
        return var

def thick_stripes_BP_solver(A, b, tol=1e-10, verbose=False, write=False):
    mean = np.zeros_like(b)
    N = int(np.sqrt(len(b)))
    vertical_coords = np.arange(2*N).reshape((-1, 4))
    vertical_local_x = []
    vertical_local_y = []
    for c in vertical_coords:
        x, y = np.meshgrid(c, c, indexing='ij')
        vertical_local_x.append(x)
        vertical_local_y.append(y)
    horizontal_coords = np.sort(np.arange(2*N).reshape((2, -1)).T.reshape((-1, 4)), axis=1)
    horizontal_local_x = []
    horizontal_local_y = []
    for c in horizontal_coords:
        x, y = np.meshgrid(c, c, indexing='ij')
        horizontal_local_x.append(x)
        horizontal_local_y.append(y)
    global_horizontal = np.arange(N**2).reshape((-1, 2*N))
    N_h, D_h = global_horizontal.shape
    A_h = np.zeros((N_h, D_h, D_h))
    b_h = np.zeros((N_h, D_h))

    for i in range(N_h):
        X, Y = np.meshgrid(global_horizontal[i], global_horizontal[i], indexing='ij')
        A_h[i] = A[X, Y]
        b_h[i] = b[global_horizontal[i]]

    global_vertical = np.sort(np.arange(N**2).reshape((N, N)).T.reshape((-1, 2*N)), axis=1)
    N_v, D_v = global_vertical.shape
    A_v = np.zeros((N_v, D_v, D_v))
    b_v = np.zeros((N_v, D_v))

    for i in range(N_h):
        X, Y = np.meshgrid(global_vertical[i], global_vertical[i], indexing='ij')
        A_v[i] = A[X, Y]
        b_v[i] = b[global_vertical[i]]

    V_to_H_mean = np.zeros((N_v, N_h, 4)) # from which vertical, to which horizontal
    V_to_H_precision = np.zeros((N_v, N_h, 4, 4))
    H_to_V_mean = np.zeros((N_h, N_v, 4))
    H_to_V_precision = np.zeros((N_h, N_v, 4, 4))

    error = 1
    sigma = np.zeros_like(b)
    old_sigma = np.zeros_like(b)
    E = []
    E.append(np.linalg.norm(A @ mean - b, ord=np.inf))
    k = 0
    while error>tol:
        for i in range(N_v):
            A_v[i][vertical_local_x, vertical_local_y] += H_to_V_precision[:, i]
            ###
            L_0 = np.linalg.inv(A_v[i])
            ###
            V_to_H_precision[i, :] = np.linalg.inv(L_0[vertical_local_x, vertical_local_y]) - A_v[i][vertical_local_x, vertical_local_y]

            A_v[i][vertical_local_x, vertical_local_y] -= H_to_V_precision[:, i]


        for i in range(N_h):
            A_h[i][horizontal_local_x, horizontal_local_y] += V_to_H_precision[:, i]
            ###
            L_0 = np.linalg.inv(A_h[i])
            ###
            H_to_V_precision[i, :] = np.linalg.inv(L_0[horizontal_local_x, horizontal_local_y]) - A_h[i][horizontal_local_x, horizontal_local_y]

            A_h[i][horizontal_local_x, horizontal_local_y] -= V_to_H_precision[:, i]

            sigma[global_horizontal[i]] = np.diag(L_0)
        k+=1
        error = np.linalg.norm(sigma - old_sigma, ord=np.inf)
        old_sigma = sigma
        if write:
            E.append(error)
        if verbose:
            print(f'Iteration #{k}, error = {error:1.2}')
    if write:
        return sigma, E
    else:
        return sigma

def GaBP(A, b, tol=1e-10, type='Sequential', verbose=False, write=False):
    v_out = []
    v_in = []
    for i in range(len(A[0, :])):
        v_out.append(np.where(A[i] != 0)[0])
        v_in.append(np.where(A[:, i] != 0)[0])

        v_out[i] = v_out[i][v_out[i] != i]
        v_in[i] = v_in[i][v_in[i] != i]

    L, L_new = np.zeros_like(A), np.zeros_like(A)
    var, var_old = np.zeros_like(b), np.zeros_like(b)
    x = np.zeros_like(b)
    it_number = 0
    defect = 1
    E = []

    while defect > tol:
        it_number += 1
        for j in range(len(A[0,:])):
            Sigma = A[j, j] + np.sum(A[v_out[j], j]*L[v_out[j], j])
            var[j] = 1/Sigma
            if type == 'Sequential':
                L[j, v_in[j]] = -A[v_in[j], j]/(Sigma - L[v_in[j], j]*A[v_in[j], j])
            if type == 'Parallel':
                L_new[j, v_in[j]] = -A[v_in[j], j]/(Sigma - L[v_in[j], j]*A[v_in[j], j])
        if type == 'Parallel':
            L = np.copy(L_new)
        defect = np.linalg.norm(var - var_old, ord=np.inf)
        var_old = np.copy(var)
        if write:
            E.append(defect)
        if verbose:
            print(f'Iteration #{it_number}, error = {defect:1.2}')
    if write:
        return var, E
    else:
        return var

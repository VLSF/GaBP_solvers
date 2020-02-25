from scipy.linalg import solve_triangular
from scipy.sparse.linalg import cg, bicgstab, gmres, spilu
from scipy.sparse import csc_matrix
import numpy as np

JJJ = 0
E = []
fill_factor = 10

def write_error(A, b, x):
    global E
    error = np.linalg.norm(A @ x - b, ord=np.inf)
    E.append(error)

def print_error(A, b, x):
    global JJJ
    global E
    JJJ += 1
    error = np.linalg.norm(A @ x - b, ord=np.inf)
    print(f'Iteration #{JJJ}\n      error = {error:1.2}')

def print_write_error(A, b, x):
    global JJJ
    global E
    JJJ += 1
    error = np.linalg.norm(A @ x - b, ord=np.inf)
    E.append(error)
    print(f'Iteration #{JJJ}\n      error = {error:1.2}')

def CG(A, b, tol=1e-10, verbose=False, write=False):
    global JJJ
    global E
    if write:
        error = lambda x: write_error(A, b, x)
    if verbose:
        error = lambda x: print_error(A, b, x)
    if  not write and  not verbose:
        error = lambda x: True
    if write and verbose:
        error = lambda x: print_write_error(A, b, x)
    x = cg(A, b, tol=tol, callback=error)
    JJJ = 0
    E1 = E
    E = []
    if write or (write and verbose):
        return x, E1
    else:
        return x

def BICGSTAB(A, b, tol=1e-10, verbose=False, write=False):
    global JJJ
    global E
    if write:
        error = lambda x: write_error(A, b, x)
    if verbose:
        error = lambda x: print_error(A, b, x)
    if not write and  not verbose:
        error = lambda x: True
    if write and verbose:
        error = lambda x: print_write_error(A, b, x)
    x = bicgstab(A, b, tol=tol, callback=error)
    E1 = E
    E = []
    JJJ = 0
    if write or (write and verbose):
        return x[0], E1
    else:
        return x[0]

def GS(A, b, tol=1e-10, verbose=False, write=False):
    L_A = np.tril(A)
    r_A = A - L_A
    error = 1
    i = 0
    x = np.zeros_like(b)
    E = []
    E.append(np.linalg.norm(A @ x - b, ord=np.inf))
    while error>tol:
        x = solve_triangular(L_A, b - r_A @ x, lower=True)
        error = np.linalg.norm(A @ x - b, ord=np.inf)
        i+=1
        if write:
            E.append(error)
        if verbose:
            print(f'Iteration #{i}\n      error = {error:1.2}')
    if write:
        return x, E
    else:
        return x

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
    m_h = np.zeros((M, M, 2))# [from which horizontal, to which vertical, mean/variance]
    vertical_regions = np.arange(N).reshape((M,-1)).T
    m_v = np.zeros((M, M, 2))# [from which vertical, to which horizontal, mean/variance]

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
            delta_A = np.diag(m_v[:, i, 1])
            delta_b = m_v[:, i, 0]*m_v[:, i, 1]
            variances, means = exact_marginalization(A_horizontal[i]+delta_A, b_horizontal[i]+delta_b)
            m_h[i, :, 1] = 1/variances - (np.diag(A_horizontal[i]) + m_v[:, i, 1])
            ############
            #m_h[i, :, 1][np.where(abs(m_h[i, :, 1]) < 1e-10)] = 1e-5 # regularization
            ############
            m_h[i, :, 0] = (means/variances - (b_horizontal[i] + delta_b))/m_h[i, :, 1]

        # vertical update
        for i in range(M):
            delta_A = np.diag(m_h[:, i, 1])
            delta_b = m_h[:, i, 0]*m_h[:, i, 1]
            variances, means = exact_marginalization(A_vertical[i]+delta_A, b_vertical[i]+delta_b)
            m_v[i, :, 1] = 1/variances - (np.diag(A_vertical[i]) + m_h[:, i, 1])
            ############
            #m_v[i, :, 1][np.where(abs(m_v[i, :, 1]) < 1e-10)] = 1e-5 # regularization
            ############
            m_v[i, :, 0] = (means/variances - (b_vertical[i] + delta_b))/m_v[i, :, 1]

            ##
            me[vertical_regions[i]] = means
            var[vertical_regions[i]] = variances
            ##

        # Two lines below give solution via ALL message accumulation. This is equivalent to the two lines above.
        #var = 1/(np.diag(A) + (m_h[:, :, 1]).reshape((-1,)) + (m_v[:, :, 1].T).reshape((-1,)))
        #me = var*(b + (np.prod(m_h[:, :, :], axis=2)).reshape((-1,)) + (np.prod(m_v[:, :, :], axis=2).T).reshape((-1,)))
        error = np.linalg.norm(A @ me - b, ord=np.inf)
        if write:
            E.append(error)
        if verbose:
            print(f'Iteration #{k}\n      error = {error:1.2}')
    if write:
        return me, E
    else:
        return me

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

    error = 1
    k=0
    while error>tol:
        k+=1
        left_A[local_left_intersection, local_left_intersection] += right_to_left_precision
        left_b[local_left_intersection] += right_to_left_precision*right_to_left_mean
        L_0, mu_0 = solve(left_A, left_b)
        left_A[local_left_intersection, local_left_intersection] -= right_to_left_precision
        left_b[local_left_intersection] -= right_to_left_precision*right_to_left_mean
        left_to_right_precision = 1/L_0[local_left_intersection] - (right_to_left_precision + left_A[local_left_intersection, local_left_intersection])
        left_to_right_mean = (mu_0[local_left_intersection]/L_0[local_left_intersection] - (right_to_left_mean*right_to_left_precision + left_b[local_left_intersection]))/left_to_right_precision

        x[left_grid] = mu_0
        sigma[left_grid] = 1/L_0

        right_A[local_right_intersection, local_right_intersection] += left_to_right_precision
        right_b[local_right_intersection] += left_to_right_precision*left_to_right_mean
        L_0, mu_0 = solve(right_A, right_b)
        right_A[local_right_intersection, local_right_intersection] -= left_to_right_precision
        right_b[local_right_intersection] -= left_to_right_precision*left_to_right_mean
        right_to_left_precision = 1/L_0[local_right_intersection] - (left_to_right_precision + right_A[local_right_intersection, local_right_intersection])
        right_to_left_mean = (mu_0[local_right_intersection]/L_0[local_right_intersection] - (left_to_right_mean*left_to_right_precision + right_b[local_right_intersection]))/right_to_left_precision

        x[right_grid] = mu_0
        sigma[right_grid] = 1/L_0

        error = np.linalg.norm(A @ x - b, ord=np.inf)
        if write:
            E.append(error)
        if verbose:
            print(f'Iteration #{k}\n      error = {error:1.2}')
    if write:
        return x, E
    else:
        return x

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
    mean = np.zeros_like(b)
    E = []
    E.append(np.linalg.norm(A @ mean - b, ord=np.inf))
    k = 0
    while error>tol:
        for i in range(N_v):
            A_v[i][vertical_local_x, vertical_local_y] += H_to_V_precision[:, i]
            b_v[i][vertical_coords] += np.einsum('ijk, ik -> ij', H_to_V_precision[:, i], H_to_V_mean[:, i])
            ###
            L_0 = np.linalg.inv(A_v[i])
            m_0 = L_0 @ b_v[i]
            ###
            V_to_H_precision[i, :] = np.linalg.inv(L_0[vertical_local_x, vertical_local_y]) - A_v[i][vertical_local_x, vertical_local_y]
            V_to_H_mean[i, :] = np.einsum('ijk, ik -> ij', np.linalg.inv(V_to_H_precision[i, :] + np.eye(4)*1e-10), (np.einsum('ijk, ik -> ij', np.linalg.inv(L_0[vertical_local_x, vertical_local_y]), m_0[vertical_coords]) - b_v[i][vertical_coords]))

            A_v[i][vertical_local_x, vertical_local_y] -= H_to_V_precision[:, i]
            b_v[i][vertical_coords] -= np.einsum('ijk, ik -> ij', H_to_V_precision[:, i], H_to_V_mean[:, i])


        for i in range(N_h):
            A_h[i][horizontal_local_x, horizontal_local_y] += V_to_H_precision[:, i]
            b_h[i][horizontal_coords] += np.einsum('ijk, ik -> ij', V_to_H_precision[:, i], V_to_H_mean[:, i])
            ###
            L_0 = np.linalg.inv(A_h[i])
            m_0 = L_0 @ b_h[i]
            ###
            H_to_V_precision[i, :] = np.linalg.inv(L_0[horizontal_local_x, horizontal_local_y]) - A_h[i][horizontal_local_x, horizontal_local_y]
            H_to_V_mean[i, :] = np.einsum('ijk, ik -> ij', np.linalg.inv(H_to_V_precision[i, :] + np.eye(4)*1e-10), (np.einsum('ijk, ik -> ij', np.linalg.inv(L_0[horizontal_local_x, horizontal_local_y]), m_0[horizontal_coords]) - b_h[i][horizontal_coords]))

            A_h[i][horizontal_local_x, horizontal_local_y] -= V_to_H_precision[:, i]
            b_h[i][horizontal_coords] -= np.einsum('ijk, ik -> ij', V_to_H_precision[:, i], V_to_H_mean[:, i])
            mean[global_horizontal[i]] = m_0
        k+=1
        error = np.linalg.norm(A @ mean - b, ord=np.inf)
        if write:
            E.append(error)
        if verbose:
            print(f'Iteration #{k}, error = {error:1.2}')
    if write:
        return mean, E
    else:
        return mean


def GaBP(A, b, tol=1e-10, type='Sequential', verbose=False, write=False):
    v_out = []
    v_in = []
    for i in range(len(A[0, :])):
        v_out.append(np.where(A[i] != 0)[0])
        v_in.append(np.where(A[:, i] != 0)[0])

        v_out[i] = v_out[i][v_out[i] != i]
        v_in[i] = v_in[i][v_in[i] != i]

    L, L_new, mu, mu_new = np.zeros_like(A), np.zeros_like(A), np.zeros_like(A), np.zeros_like(A)
    x = np.zeros_like(b)
    it_number = 0
    defect = 1
    E = []

    while defect > tol:
        it_number += 1
        for j in range(len(A[0,:])):
            m = b[j] + np.sum(L[v_out[j], j]*mu[v_out[j], j])
            Sigma = A[j, j] + np.sum(A[v_out[j], j]*L[v_out[j], j])
            x[j] = m/Sigma
            if type == 'Sequential':
                mu[j, v_in[j]] = m - L[v_in[j], j]*mu[v_in[j], j]
                L[j, v_in[j]] = -A[v_in[j], j]/(Sigma - L[v_in[j], j]*A[v_in[j], j])
            if type == 'Parallel':
                mu_new[j, v_in[j]] = m - L[v_in[j], j]*mu[v_in[j], j]
                L_new[j, v_in[j]] = -A[v_in[j], j]/(Sigma - L[v_in[j], j]*A[v_in[j], j])
        if type == 'Parallel':
            mu, L = np.copy(mu_new), np.copy(L_new)
        defect = np.linalg.norm(A@x - b, ord=np.inf)
        if write:
            E.append(defect)
        if verbose:
            print(f'Iteration #{it_number}, error = {defect:1.2}')
    if write:
        return x, E
    else:
        return x

def full_weight_restriction(u):
    K = int(np.sqrt(len(u)))
    M = int((K-1)/2)
    v = np.zeros((M, M))
    w = u.reshape((K, K))
    v += w[1::2, 1::2]/4 + (w[::2, 1::2][:-1, :] + w[2::2, 1::2])/8
    v += (w[1::2, ::2][:, :-1] + w[1::2, 2::2])/8
    v += (w[::2, ::2][:-1, :-1] + w[2::2, 2::2] + w[::2, 2::2][:-1, :] + w[2::2, ::2][:, :-1])/16
    v = v.reshape(-1,)
    return v

def full_weight_prolongation(u):
    K = int(np.sqrt(len(u)))
    v = np.zeros((2*K+1, 2*K+1))
    v[1::2, 1::2] = u.reshape((K, K))
    v[1::2, ::2][:, :-1] += v[1::2, 1::2]/2
    v[1::2, 2::2] += v[1::2, 1::2]/2
    v[::2, :][:-1, :] += v[1::2, :]/2
    v[2::2, :] += v[1::2, :]/2
    v = v.reshape(-1,)
    return v

def GS_smoother(A, b, x_0, N):
    L_A = np.tril(A)
    r_A = A - L_A
    x = np.copy(x_0)
    for _ in range(N):
        x = solve_triangular(L_A, b - r_A @ x, lower=True)
    return x

def red_black_GS_smoother(A, b, x_0, N):
    n = len(b)
    coords = np.arange(n).reshape(int(np.sqrt(n)), -1)
    black = np.sort(np.hstack([coords[::2, ::2].reshape(-1, ), coords[1::2, 1::2].reshape(-1, )]))
    red = np.sort(np.hstack([coords[1::2, ::2].reshape(-1, ), coords[::2, 1::2].reshape(-1, )]))
    x = np.copy(x_0)
    for _ in range(N):
        x[red] = 0
        x[red] = (b[red] - (A @ x)[red])/A[red, red]
        x[black] = 0
        x[black] = (b[black] - (A @ x)[black])/A[black, black]
    return x

def four_colours_GS_smoother(A, b, x_0, N):
    n = len(b)
    coords = np.arange(n).reshape(int(np.sqrt(n)), -1)
    black_1 = np.sort(coords[::2, ::2].reshape(-1, ))
    black_2 = np.sort(coords[1::2, 1::2].reshape(-1, ))
    red_1 = np.sort(coords[1::2, ::2].reshape(-1, ))
    red_2 = np.sort(coords[::2, 1::2].reshape(-1, ))
    x = np.copy(x_0)
    for _ in range(N):
        for colour in [red_1, red_2, black_1, black_2]:
            x[colour] = 0
            x[colour] = (b[colour] - (A @ x)[colour])/A[colour, colour]
    return x

def Jacobi_smoother(A, b, x_0, N, w=4/5):
    D_A = np.diag(A)
    r_A = A - np.diag(D_A)
    x = np.copy(x_0)
    for _ in range(N):
        x = (1-w)*x + w*(b - r_A @ x)/D_A
    return x

def GaBP_smoother(A, d, x_0, N, type='Sequential'):
    v_out = []
    v_in = []
    for i in range(len(A[0, :])):
        v_out.append(np.where(A[i] != 0)[0])
        v_in.append(np.where(A[:, i] != 0)[0])

        v_out[i] = v_out[i][v_out[i] != i]
        v_in[i] = v_in[i][v_in[i] != i]

    L, L_new, mu, mu_new = np.zeros_like(A), np.zeros_like(A), np.zeros_like(A), np.zeros_like(A)
    x = np.zeros_like(d)
    b = d - A@x_0

    for _ in range(N):
        for j in range(len(A[0,:])):
            m = b[j] + np.sum(L[v_out[j], j]*mu[v_out[j], j])
            Sigma = A[j, j] + np.sum(A[v_out[j], j]*L[v_out[j], j])
            x[j] = m/Sigma
            if type == 'Sequential':
                mu[j, v_in[j]] = m - L[v_in[j], j]*mu[v_in[j], j]
                L[j, v_in[j]] = -A[v_in[j], j]/(Sigma - L[v_in[j], j]*A[v_in[j], j])
            if type == 'Parallel':
                mu_new[j, v_in[j]] = m - L[v_in[j], j]*mu[v_in[j], j]
                L_new[j, v_in[j]] = -A[v_in[j], j]/(Sigma - L[v_in[j], j]*A[v_in[j], j])
        if type == 'Parallel':
            mu, L = np.copy(mu_new), np.copy(L_new)
    return x_0 + x

def red_black_GaBP_smoother(A, d, x_0, N):
    v_out = []
    v_in = []
    for i in range(len(A[0, :])):
        v_out.append(np.where(A[i] != 0)[0])
        v_in.append(np.where(A[:, i] != 0)[0])

        v_out[i] = v_out[i][v_out[i] != i]
        v_in[i] = v_in[i][v_in[i] != i]

    L, L_new, mu, mu_new = np.zeros_like(A), np.zeros_like(A), np.zeros_like(A), np.zeros_like(A)
    x = np.zeros_like(d)
    b = d - A@x_0
    n = len(b)
    coords = np.arange(n).reshape(int(np.sqrt(n)), -1)
    black = np.sort(np.hstack([coords[::2, ::2].reshape(-1, ), coords[1::2, 1::2].reshape(-1, )]))
    red = np.sort(np.hstack([coords[1::2, ::2].reshape(-1, ), coords[::2, 1::2].reshape(-1, )]))
    order = np.hstack([red, black])

    for _ in range(N):
        for j in order:
            m = b[j] + np.sum(L[v_out[j], j]*mu[v_out[j], j])
            Sigma = A[j, j] + np.sum(A[v_out[j], j]*L[v_out[j], j])
            x[j] = m/Sigma
            mu[j, v_in[j]] = m - L[v_in[j], j]*mu[v_in[j], j]
            L[j, v_in[j]] = -A[v_in[j], j]/(Sigma - L[v_in[j], j]*A[v_in[j], j])
    return x_0 + x

def four_colours_GaBP_smoother(A, d, x_0, N):
    v_out = []
    v_in = []
    for i in range(len(A[0, :])):
        v_out.append(np.where(A[i] != 0)[0])
        v_in.append(np.where(A[:, i] != 0)[0])

        v_out[i] = v_out[i][v_out[i] != i]
        v_in[i] = v_in[i][v_in[i] != i]

    L, L_new, mu, mu_new = np.zeros_like(A), np.zeros_like(A), np.zeros_like(A), np.zeros_like(A)
    x = np.zeros_like(d)
    b = d - A@x_0
    n = len(b)
    coords = np.arange(n).reshape(int(np.sqrt(n)), -1)
    black_1 = np.sort(coords[::2, ::2].reshape(-1, ))
    black_2 = np.sort(coords[1::2, 1::2].reshape(-1, ))
    red_1 = np.sort(coords[1::2, ::2].reshape(-1, ))
    red_2 = np.sort(coords[::2, 1::2].reshape(-1, ))
    order = np.hstack([red_1, red_2, black_1, black_2])

    for _ in range(N):
        for j in order:
            m = b[j] + np.sum(L[v_out[j], j]*mu[v_out[j], j])
            Sigma = A[j, j] + np.sum(A[v_out[j], j]*L[v_out[j], j])
            x[j] = m/Sigma
            mu[j, v_in[j]] = m - L[v_in[j], j]*mu[v_in[j], j]
            L[j, v_in[j]] = -A[v_in[j], j]/(Sigma - L[v_in[j], j]*A[v_in[j], j])
    return x_0 + x

def GaBP_parallel_smoother(A, d, x_0, N, w=4/5):
    v_out = []
    v_in = []
    for i in range(len(A[0, :])):
        v_out.append(np.where(A[i] != 0)[0])
        v_in.append(np.where(A[:, i] != 0)[0])

        v_out[i] = v_out[i][v_out[i] != i]
        v_in[i] = v_in[i][v_in[i] != i]

    L, L_new, mu, mu_new = np.zeros_like(A), np.zeros_like(A), np.zeros_like(A), np.zeros_like(A)
    x = np.zeros_like(d)
    b = d - A@x_0

    for _ in range(N):
        for j in range(len(A[0,:])):
            m = b[j] + np.sum(L[v_out[j], j]*mu[v_out[j], j])
            Sigma = A[j, j] + np.sum(A[v_out[j], j]*L[v_out[j], j])
            x[j] = m/Sigma
            mu_new[j, v_in[j]] = m - L[v_in[j], j]*mu[v_in[j], j]
        L_new[j, v_in[j]] = -A[v_in[j], j]/(Sigma - L[v_in[j], j]*A[v_in[j], j])
        mu, L = np.copy(mu_new), np.copy(L_new)
    return x_0 + w*x

def stripes_GaBP_smoother(A, d, x_0, N_it):
    b = d - A@x_0
    N = len(b)
    me, var = np.zeros_like(b), np.zeros_like(b)
    M = int(np.sqrt(N))
    horizontal_regions = np.arange(N).reshape((M,-1))
    m_h = np.zeros((M, M, 2))# [from which horizontal, to which vertical, mean/variance]
    vertical_regions = np.arange(N).reshape((M,-1)).T
    m_v = np.zeros((M, M, 2))# [from which vertical, to which horizontal, mean/variance]

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

    me = np.zeros_like(b)
    for _ in range(N_it):

        # horizontal update
        for i in range(M):
            delta_A = np.diag(m_v[:, i, 1])
            delta_b = m_v[:, i, 0]*m_v[:, i, 1]
            variances, means = exact_marginalization(A_horizontal[i]+delta_A, b_horizontal[i]+delta_b)
            m_h[i, :, 1] = 1/variances - (np.diag(A_horizontal[i]) + m_v[:, i, 1])
            m_h[i, :, 0] = (means/variances - (b_horizontal[i] + delta_b))/m_h[i, :, 1]

        # vertical update
        for i in range(M):
            delta_A = np.diag(m_h[:, i, 1])
            delta_b = m_h[:, i, 0]*m_h[:, i, 1]
            variances, means = exact_marginalization(A_vertical[i]+delta_A, b_vertical[i]+delta_b)
            m_v[i, :, 1] = 1/variances - (np.diag(A_vertical[i]) + m_h[:, i, 1])
            m_v[i, :, 0] = (means/variances - (b_vertical[i] + delta_b))/m_v[i, :, 1]

            ##
            me[vertical_regions[i]] = means
            var[vertical_regions[i]] = variances
            ##

    return x_0 + me

def GS_line_x_smoother(A, b, x_0, N):
    M = len(x_0)
    K = int(np.sqrt(M))
    coords = np.arange(M).reshape(K, -1)
    x = np.copy(x_0)
    for _ in range(N):
        for i in range(K)[::2]:
            X, Y = np.meshgrid(coords[i], coords[i], indexing='ij')
            b_ = b[coords[i]]
            if 0 <= i-1 <= K-1:
                X_, Y_ = np.meshgrid(coords[i], coords[i-1], indexing='ij')
                b_ -= A[X_, Y_] @ x[coords[i-1]]
            if 0 <= i+1 <= K-1:
                X_, Y_ = np.meshgrid(coords[i], coords[i+1], indexing='ij')
                b_ -= A[X_, Y_] @ x[coords[i+1]]
            x[coords[i]] = np.linalg.inv(A[X, Y]) @ b_
        for i in range(K)[1::2]:
            X, Y = np.meshgrid(coords[i], coords[i], indexing='ij')
            b_ = b[coords[i]]
            if 0 <= i-1 <= K-1:
                X_, Y_ = np.meshgrid(coords[i], coords[i-1], indexing='ij')
                b_ -= A[X_, Y_] @ x[coords[i-1]]
            if 0 <= i+1 <= K-1:
                X_, Y_ = np.meshgrid(coords[i], coords[i+1], indexing='ij')
                b_ -= A[X_, Y_] @ x[coords[i+1]]
            x[coords[i]] = np.linalg.inv(A[X, Y]) @ b_
    return x

def GS_line_y_smoother(A, b, x_0, N):
    M = len(x_0)
    K = int(np.sqrt(M))
    coords = np.arange(M).reshape(K, -1).T
    x = np.copy(x_0)
    for _ in range(N):
        for i in range(K)[::2]:
            X, Y = np.meshgrid(coords[i], coords[i], indexing='ij')
            b_ = b[coords[i]]
            if 0 <= i-1 <= K-1:
                X_, Y_ = np.meshgrid(coords[i], coords[i-1], indexing='ij')
                b_ -= A[X_, Y_] @ x[coords[i-1]]
            if 0 <= i+1 <= K-1:
                X_, Y_ = np.meshgrid(coords[i], coords[i+1], indexing='ij')
                b_ -= A[X_, Y_] @ x[coords[i+1]]
            x[coords[i]] = np.linalg.inv(A[X, Y]) @ b_
        for i in range(K)[1::2]:
            X, Y = np.meshgrid(coords[i], coords[i], indexing='ij')
            b_ = b[coords[i]]
            if 0 <= i-1 <= K-1:
                X_, Y_ = np.meshgrid(coords[i], coords[i-1], indexing='ij')
                b_ -= A[X_, Y_] @ x[coords[i-1]]
            if 0 <= i+1 <= K-1:
                X_, Y_ = np.meshgrid(coords[i], coords[i+1], indexing='ij')
                b_ -= A[X_, Y_] @ x[coords[i+1]]
            x[coords[i]] = np.linalg.inv(A[X, Y]) @ b_
    return x

def GS_line_xy_smoother(A, b, x_0, N):
    x = np.copy(x_0)
    for _ in range(N):
        x = GS_line_y_smoother(A, b, x, 1)
        x = GS_line_x_smoother(A, b, x, 1)
    return x

def ilu_smoother(A, b, x_0, N):
    global fill_factor
    B = csc_matrix(A)
    inv_B_approx = spilu(B, fill_factor=fill_factor)
    x = np.copy(x_0)
    for _ in range(N):
        r = b - A@x
        x = x + inv_B_approx.solve(r)
    return x

def multigrid_V_sweep(A, b, x_0, smoother, N_pre, N_post):
    if len(A) == 1:
        return x_0 + np.linalg.inv(A[0]) @ b
    # pre smoothing
    x_0 = smoother(A[0], b, x_0, N_pre)
    # restriction
    y = full_weight_restriction(b - A[0]@x_0)
    # coarse solution
    e = multigrid_V_sweep(A[1:], y, 0*y, smoother, N_pre, N_post)
    # prolongation
    x_0 += full_weight_prolongation(e)
    # post smoothing
    x_0 = smoother(A[0], b, x_0, N_post)
    return x_0

def multigrid_solver(A, b, smoother, N_pre, N_post, tol=1e-10, verbose=False, write=False):
    error = 1
    i = 0
    x = np.zeros_like(b)
    E = []
    E.append(np.linalg.norm(A[0] @ x - b, ord=np.inf))
    while error>tol:
        x = multigrid_V_sweep(A, b, x, smoother, N_pre, N_post)
        error = np.linalg.norm(A[0] @ x - b, ord=np.inf)
        i+=1
        if write:
            E.append(error)
        if verbose:
            print(f'Iteration #{i}\n      error = {error:1.2}')
    if write:
        return x, E
    else:
        return x

def generic_solver(A, b, smoother, tol=1e-10, verbose=False, write=False):
    error = 1
    i = 0
    x = np.zeros_like(b)
    E = []
    E.append(np.linalg.norm(A @ x - b, ord=np.inf))
    while error>tol:
        x = smoother(A, b, x, 1)
        error = np.linalg.norm(A @ x - b, ord=np.inf)
        i+=1
        if write:
            E.append(error)
        if verbose:
            print(f'Iteration #{i}\n      error = {error:1.2}')
    if write:
        return x, E
    else:
        return x

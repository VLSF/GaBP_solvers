from graph_tool.all import *
import numpy as np

def asymmetric_BP_solver(A, b, tol=1e-10, verbose=False, write=False):
    g = Graph()
    g.add_vertex(len(b))
    diag_A = np.diag(np.diag(A))
    g.add_edge_list(np.array(np.where(A - diag_A + (A - diag_A).T != 0)).T)
    edges = g.get_edges()
    reverse_edges = edges[np.lexsort(edges[:, 0:2].T)]
    vertices = g.get_vertices()

    mu = g.new_edge_property('double', 0.1)
    L = g.new_edge_property('double', 0.1)
    A_g = g.new_edge_property('double', A[edges[:, 0], edges[:, 1]])
    diag_A_g = g.new_vertex_property('double', A[vertices, vertices])
    b_g = g.new_vertex_property('double', b[vertices])
    x = np.zeros_like(b)
    E = []
    E.append(np.linalg.norm(A @ x - b, ord=np.inf))

    error = 1
    i = 0
    while error>tol:
        ### single BP sweep
        mu_old = np.copy(mu.a)
        L_old = np.copy(L.a)
        mu.a = mu.a*L.a
        mu_L_in = g.get_in_degrees(vertices, eweight=mu)
        L.a = L.a*A_g.a
        L_in = g.get_in_degrees(vertices, eweight=L)
        mu.a = mu_L_in[edges[:, 0]] + b_g.a[edges[:, 0]] - mu.a[reverse_edges[:, 2]]
        L.a = -A_g.a[reverse_edges[:, 2]]/(diag_A_g.a[edges[:, 0]] + L_in[edges[:, 0]] - L.a[reverse_edges[:, 2]])
        ###
        sigma = 1/(diag_A_g.a + L_in)
        x = (mu_L_in + b_g.a)*sigma
        error = np.linalg.norm(A @ x - b, ord = np.inf)
        i+=1
        if write:
            E.append(error)
        if verbose:
            print(f'Iteration #{i}\n      error = {error:1.2}')
    if write:
        return x, E
    else:
        return x

import numpy as np

def d2A(a, alpha, b, beta, q, L_x, L_y, n_x, n_y):
    '''
    The function construct second order differential operator of general form
    inside the square [0, L_x]*[0, L_y]. Finite differences, central stencils,
    second order. Mapping between coefficients and input parameters is

    a*d2x + alpha*dx + b*d2y + beta*dy + q*d2xy

    The order is lexicographical, first direction is Y. Order corresponds to the
    indexing='ij' for the numpy.meshgrid. Function can be used to construct sparse
    matrix in the coordinate format.

    Parameters
    ----------
    a: callable
        Coefficient before d^2/dx^2. Supposed to take two arguments a(x, y). Function
        is called for scalar values so math module is more preferable than numpy.
    alpha: callable
        Coefficient before d/dx. Supposed to take two arguments alpha(x, y). Function
        is called for scalar values so math module is more preferable than numpy.
    b: callable
        Coefficient before d^2/dy^2. Supposed to take two arguments b(x, y). Function
        is called for scalar values so math module is more preferable than numpy.
    beta: callable
        Coefficient before d/dy. Supposed to take two arguments beta(x, y). Function
        is called for scalar values so math module is more preferable than numpy.
    q: callable
        Coefficient before d^2/dxdy. Supposed to take two arguments q(x, y). Function
        is called for scalar values so math module is more preferable than numpy.
    L_x: float
        Length of the domain along x.
    L_y: float
        Length of the domain along y.
    n_x: int
        Number of inner points along x. Overall number of points along x is n_x+2.
    n_y: int
        Number of inner points along y. Overall number of points along y is n_y+2.

    Returns
    -------
    row: np.array
        Rows of the matrix corresponding to the operator and discretization.
    col: np.array
        Columns of the matrix corresponding to the operator and discretization.
    values: np.array
        Values of the matrix corresponding to the operator and discretization.
        Matrix can be constructed with A[row, col] = values or in the coordinate format.
    '''
    h_x = L_x/(n_x+1)
    h_y = L_y/(n_y+1)

    row = []
    col = []
    values = []
    for i in range(n_x):
        for j in range(n_y):
            x = (i+1)*h_x
            y = (j+1)*h_y
            row.append(j + i*n_y)
            col.append(j + i*n_y)
            values.append(-2*(a(x, y)/h_x**2 + b(x, y)/h_y**2)) # d2x, d2y
            if 0<=i+1<n_x:
                row.append(j + i*n_y)
                col.append(j + (i+1)*n_y)
                values.append(a(x, y)/h_x**2 + alpha(x, y)/(2*h_x)) # d2x, dx
                if 0<=j+1<n_y:
                    row.append(j + i*n_y)
                    col.append(j+1 + (i+1)*n_y)
                    values.append(q(x, y)/(4*h_x*h_y)) # dxdy
                if 0<=j-1<n_y:
                    row.append(j + i*n_y)
                    col.append(j-1 + (i+1)*n_y)
                    values.append(-q(x, y)/(4*h_x*h_y)) # dxdy
            if 0<=i-1<n_x:
                row.append(j + i*n_y)
                col.append(j + (i-1)*n_y)
                values.append(a(x, y)/h_x**2 - alpha(x, y)/(2*h_x)) # d2x, dx
                if 0<=j+1<n_y:
                    row.append(j + i*n_y)
                    col.append(j+1 + (i-1)*n_y)
                    values.append(-q(x, y)/(4*h_x*h_y)) # dxdy
                if 0<=j-1<n_y:
                    row.append(j + i*n_y)
                    col.append(j-1 + (i-1)*n_y)
                    values.append(q(x, y)/(4*h_x*h_y)) # dxdy
            if 0<=j+1<n_y:
                row.append(j + i*n_y)
                col.append(j + 1 + i*n_y)
                values.append(b(x, y)/h_y**2 + beta(x, y)/(2*h_y)) # d2y, dy
            if 0<=j-1<n_y:
                row.append(j + i*n_y)
                col.append(j - 1 + i*n_y)
                values.append(b(x, y)/h_y**2 - beta(x, y)/(2*h_y)) # d2y, dy
    return np.array(row), np.array(col), np.array(values)

def mod_rhs(a, alpha, b, beta, q, r, up, left, down, right, L_x, L_y, n_x, n_y):
    '''
    The function correct right hand side for the second order differential operator
    of general form inside the square [0, L_x]*[0, L_y]. Finite differences,
    central stencils, second order. The equation under consideration is

    (a*d2x + alpha*dx + b*d2y + beta*dy + q*d2xy)(function) = r

    The order is lexicographical, first direction is Y. Order corresponds to the
    indexing='ij' for the numpy.meshgrid. Functions up, down, left, right
    represent boundary conditions.

    Parameters
    ----------
    a: callable
        Coefficient before d^2/dx^2. Supposed to take two arguments a(x, y). Function
        is called for scalar values so math module is more preferable than numpy.
    alpha: callable
        Coefficient before d/dx. Supposed to take two arguments alpha(x, y). Function
        is called for scalar values so math module is more preferable than numpy.
    b: callable
        Coefficient before d^2/dy^2. Supposed to take two arguments b(x, y). Function
        is called for scalar values so math module is more preferable than numpy.
    beta: callable
        Coefficient before d/dy. Supposed to take two arguments beta(x, y). Function
        is called for scalar values so math module is more preferable than numpy.
    q: callable
        Coefficient before d^2/dxdy. Supposed to take two arguments q(x, y). Function
        is called for scalar values so math module is more preferable than numpy.
    r: callable
        Right hand side. Supposed to take two arguments r(x, y). Function is called
        for scalar values so math module is more preferable than numpy.
    up: callable
        Boundary condition on y = L_y. Supposed to take one argument up(x). Function
        is called for scalar values so math module is more preferable than numpy.
    left: callable
        Boundary condition on x = 0. Supposed to take one argument left(y). Function
        is called for scalar values so math module is more preferable than numpy.
    down: callable
        Boundary conditions on y = 0. Supposed to take one argument down(x). Function
        is called for scalar values so math module is more preferable than numpy.
    right: callable
        Boundary conditions on x = L_x. Supposed to take one argument right(y). Function
        is called for scalar values so math module is more preferable than numpy.
    L_x: float
        Length of the domain along x.
    L_y: float
        Length of the domain along y.
    n_x: int
        Number of inner points along x. Overall number of points along x is n_x+2.
    n_y: int
        Number of inner points along y. Overall number of points along y is n_y+2.

    Returns
    -------
    rhs: np.array
        The shape is (n_x*ny,). Contains rhs with corrections from boundary condition.
        In case matrix A corresponding to the operator is available for all inner
        points, rhs can be used as a right hand side in the linear system Ax = b.
    '''
    h_x = L_x/(n_x+1)
    h_y = L_y/(n_y+1)

    rhs = np.zeros(n_x*n_y)
    for i in range(n_x):
        for j in range(n_y):
            x = (i+1)*h_x
            y = (j+1)*h_y
            rhs[j + i*n_y] = r(x, y)
            if i == 0:
                rhs[j + i*n_y] -= left(y)*(a(x, y)/h_x - alpha(x, y)/2)/h_x + q(x,y)*(left(y-h_y) - left(y+h_y))/(4*h_x*h_y)
            if i == n_x-1:
                rhs[j + i*n_y] -= right(y)*(a(x, y)/h_x + alpha(x, y)/2)/h_x + q(x,y)*(right(y+h_y) - right(y-h_y))/(4*h_x*h_y)
            if j == 0:
                if i == 0:
                    rhs[j + i*n_y] -= down(x)*(b(x, y)/h_y - beta(x, y)/2)/h_y + q(x, y)*( - down(x+h_x))/(4*h_x*h_y)
                elif i == n_x-1:
                    rhs[j + i*n_y] -= down(x)*(b(x, y)/h_y - beta(x, y)/2)/h_y + q(x, y)*(down(x-h_x) )/(4*h_x*h_y)
                else:
                    rhs[j + i*n_y] -= down(x)*(b(x, y)/h_y - beta(x, y)/2)/h_y + q(x, y)*(down(x-h_x) - down(x+h_x))/(4*h_x*h_y)
            if j == n_y-1:
                if i == 0:
                    rhs[j + i*n_y] -= up(x)*(b(x, y)/h_y + beta(x, y)/2)/h_y + q(x, y)*(up(x+h_x))/(4*h_x*h_y)
                elif i == n_x-1:
                    rhs[j + i*n_y] -= up(x)*(b(x, y)/h_y + beta(x, y)/2)/h_y + q(x, y)*( - up(x-h_x))/(4*h_x*h_y)
                else:
                    rhs[j + i*n_y] -= up(x)*(b(x, y)/h_y + beta(x, y)/2)/h_y + q(x, y)*(up(x+h_x) - up(x-h_x))/(4*h_x*h_y)
    return rhs

def d2A_Neumann(a, alpha, b, beta, L_x, L_y, n_x, n_y):
    '''
    The function construct a second order differential operator inside the square
    [0, L_x]*[0, L_y] with the Neumann boundary conditions. Finite differences,
    central stencils, second order. Mapping between coefficients and input parameters is

    a*d2x + alpha*dx + b*d2y + beta*dy

    The order is lexicographical, first direction is Y. Order corresponds to the
    indexing='ij' for the numpy.meshgrid. Function can be used to construct sparse
    matrix in the coordinate format.

    Parameters
    ----------
    a: callable
        Coefficient before d^2/dx^2. Supposed to take two arguments a(x, y). Function
        is called for scalar values so math module is more preferable than numpy.
    alpha: callable
        Coefficient before d/dx. Supposed to take two arguments alpha(x, y). Function
        is called for scalar values so math module is more preferable than numpy.
    b: callable
        Coefficient before d^2/dy^2. Supposed to take two arguments b(x, y). Function
        is called for scalar values so math module is more preferable than numpy.
    beta: callable
        Coefficient before d/dy. Supposed to take two arguments beta(x, y). Function
        is called for scalar values so math module is more preferable than numpy.
    L_x: float
        Length of the domain along x.
    L_y: float
        Length of the domain along y.
    n_x: int
        Number of inner points along x. Overall number of points along x is n_x+2.
    n_y: int
        Number of inner points along y. Overall number of points along y is n_y+2.

    Returns
    -------
    row: np.array
        Rows of the matrix corresponding to the operator and discretization.
    col: np.array
        Columns of the matrix corresponding to the operator and discretization.
    values: np.array
        Values of the matrix corresponding to the operator and discretization.
        Matrix can be constructed with A[row, col] = values or in the coordinate format.
    '''
    h_x = L_x/(n_x+1)
    h_y = L_y/(n_y+1)

    row = []
    col = []
    values = []
    # inside
    for i in range(1, n_x+1):
        for j in range(1, n_y+1):
            x = i*h_x
            y = j*h_y
            # d2x, d2y (diagonal)
            row.append(j + i*(n_y+2))
            col.append(j + i*(n_y+2))
            values.append(-2*(a(x, y)/h_x**2 + b(x, y)/h_y**2))
            # d2x, dx (east)
            row.append(j + i*(n_y+2))
            if i == n_x:
                col.append(j + (i+1)*(n_y+2))
            else:
                col.append(j + (i+1)*(n_y+2))
            values.append(a(x, y)/h_x**2 + alpha(x, y)/(2*h_x))
            # d2x, dx (west)
            row.append(j + i*(n_y+2))
            if i == 1:
                col.append(j + (i-1)*(n_y+2))
            else:
                col.append(j + (i-1)*(n_y+2))
            values.append(a(x, y)/h_x**2 - alpha(x, y)/(2*h_x))
            # d2y, dy (north)
            row.append(j + i*(n_y+2))
            col.append((j+1) + i*(n_y+2))
            values.append(b(x, y)/h_y**2 + beta(x, y)/(2*h_y))
            # d2y, dy (south)
            row.append(j + i*(n_y+2))
            col.append((j-1) + i*(n_y+2))
            values.append(b(x, y)/h_y**2 - beta(x, y)/(2*h_y))

    # west boundary
    for j in range(1, n_y+1):
        row.append(j)
        col.append(j)
        values.append(-3/(2*h_x))

        row.append(j)
        col.append(j + (n_y+2))
        values.append(2/h_x)

        row.append(j)
        col.append(j + 2*(n_y+2))
        values.append(-1/(2*h_x))

    # east boundary
    for j in range(1, n_y+1):
        row.append(j + (n_x+1)*(n_y+2))
        col.append(j + (n_x+1)*(n_y+2))
        values.append(3/(2*h_x))

        row.append(j + (n_x+1)*(n_y+2))
        col.append(j + n_x*(n_y+2))
        values.append(-2/h_x)

        row.append(j + (n_x+1)*(n_y+2))
        col.append(j + (n_x-1)*(n_y+2))
        values.append(1/(2*h_x))

    # north boundary
    for i in range(1, n_x+1):
        row.append((n_y+1) + i*(n_y+2))
        col.append((n_y+1) + i*(n_y+2))
        values.append(3/(2*h_y))

        row.append((n_y+1) + i*(n_y+2))
        col.append(n_y + i*(n_y+2))
        values.append(-2/h_y)

        row.append((n_y+1) + i*(n_y+2))
        col.append((n_y-1) + i*(n_y+2))
        values.append(1/(2*h_y))

    # south boundary
    for i in range(1, n_x+1):
        row.append(i*(n_y+2))
        col.append(i*(n_y+2))
        values.append(-3/(2*h_y))

        row.append(i*(n_y+2))
        col.append(1 + i*(n_y+2))
        values.append(2/h_y)

        row.append(i*(n_y+2))
        col.append(2 + i*(n_y+2))
        values.append(-1/(2*h_y))

    # corners
    row.append(0)
    col.append(0)
    values.append(1)

    row.append(n_y+1)
    col.append(n_y+1)
    values.append(1)

    row.append((n_y+2)*(n_x+2)-1)
    col.append((n_y+2)*(n_x+2)-1)
    values.append(1)

    row.append((n_y+2)*(n_x+2)-2-n_y)
    col.append((n_y+2)*(n_x+2)-2-n_y)
    values.append(1)

    return np.array(row), np.array(col), np.array(values)

def mod_rhs_Neumann(r, up, left, down, right, L_x, L_y, n_x, n_y):
    '''
    The function correct right hand side for the second order differential operator
    of general form inside the square [0, L_x]*[0, L_y].The order is lexicographical,
    irst direction is Y. Order corresponds to the indexing='ij' for the numpy.meshgrid.
    Functions up, down, left, right represent boundary conditions.

    Parameters
    ----------
    r: callable
        Right hand side. Supposed to take two arguments r(x, y). Function is called
        for scalar values so math module is more preferable than numpy.
    up: callable
        Boundary condition on y = L_y. Supposed to take one argument up(x). Function
        is called for scalar values so math module is more preferable than numpy.
    left: callable
        Boundary condition on x = 0. Supposed to take one argument left(y). Function
        is called for scalar values so math module is more preferable than numpy.
    down: callable
        Boundary conditions on y = 0. Supposed to take one argument down(x). Function
        is called for scalar values so math module is more preferable than numpy.
    right: callable
        Boundary conditions on x = L_x. Supposed to take one argument right(y). Function
        is called for scalar values so math module is more preferable than numpy.
    L_x: float
        Length of the domain along x.
    L_y: float
        Length of the domain along y.
    n_x: int
        Number of inner points along x. Overall number of points along x is n_x+2.
    n_y: int
        Number of inner points along y. Overall number of points along y is n_y+2.

    Returns
    -------
    rhs: np.array
        The shape is ((n_x+2)*(ny+2),). Contains rhs with corrections from boundary condition.
        In case matrix A corresponding to the operator is available for all points, rhs can be used
        as a right hand side in the linear system Ax = b.
    '''
    h_x = L_x/(n_x+1)
    h_y = L_y/(n_y+1)

    rhs = np.zeros((n_x+2, n_y+2))
    for i in range(n_x+2):
        for j in range(n_y+2):
            x = i*h_x
            y = j*h_y
            rhs[i, j] = r(x, y)
            if i == 0:
                rhs[i, j] = left(y)
            if i == n_x+1:
                rhs[i, j] = right(y)
            if j == 0:
                rhs[i, j] = down(x)
            if j == n_y+1:
                rhs[i, j] = up(x)
    rhs[0, 0] = rhs[-1, -1] = rhs[-1, 0] = rhs[0, -1] = 1
    rhs = rhs.reshape(-1,)
    return rhs

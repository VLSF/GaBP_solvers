import numpy as np

def dx(N, F):
    h = 1/(N+1)
    row_, col_, val_ = np.zeros(2*(N-1)), np.zeros(2*(N-1)), np.zeros(2*(N-1))
    row_[::2] = np.arange(N-1)
    row_[1::2] = np.arange(1, N)
    col_[::2] = row_[::2] + 1
    col_[1::2] = row_[1::2] - 1
    val_[::2] = 1/(2*h)
    val_[1::2] = -1/(2*h)

    a = np.ones(N)
    row = np.tensordot(a, row_, axes=0).reshape((-1,))
    col = np.tensordot(a, col_, axes=0).reshape((-1,))
    val = np.tensordot(a, val_, axes=0).reshape((-1,))

    b = np.ones(2*(N-1))
    c = np.arange(N)*N
    correct = np.tensordot(c, b, axes=0).reshape((-1,))
    row+=correct
    row = np.array(row, dtype=int)
    col+=correct
    col = np.array(col, dtype=int)

    dx = np.zeros((N**2, N**2))
    dx[row, col] = val

    X = np.linspace(h, 1-h, N)
    x, y = np.meshgrid(X, X)

    f = F(x, y).reshape((-1,)).reshape((N**2, -1))
    return dx*f

def d2x(N, F):
    h = 1/(N+1)
    row_, col_, val_ = np.zeros(3*N-2), np.zeros(3*N-2), np.zeros(3*N-2)

    val_[:-1] = np.tile([-2/h**2, 1/h**2, 1/h**2], N-1)
    val_[-1] = -2/h**2
    row_ = np.tensordot(np.arange(N), np.ones(3), axes=0).reshape((-1,))[1:-1]
    correct_ = np.hstack([np.tile([0, 1, -1], N-1), [0]])
    col_ = row_ + correct_

    a = np.ones(N)
    row = np.tensordot(a, row_, axes=0).reshape((-1,))
    col = np.tensordot(a, col_, axes=0).reshape((-1,))
    val = np.tensordot(a, val_, axes=0).reshape((-1,))

    b = np.ones(3*N-2)
    c = np.arange(N)*N
    correct = np.tensordot(c, b, axes=0).reshape((-1,))
    row+=correct
    row = np.array(row, dtype=int)
    col+=correct
    col = np.array(col, dtype=int)

    d2x = np.zeros((N**2, N**2))
    d2x[row, col] = val

    X = np.linspace(h, 1-h, N)
    x, y = np.meshgrid(X, X)

    f = F(x, y).reshape((-1,)).reshape((N**2, -1))
    return d2x*f

def dy(N, F):
    h = 1/(N+1)
    row_, col_, val_ = np.zeros(2*(N-1)), np.zeros(2*(N-1)), np.zeros(2*(N-1))
    row_[::2] = np.arange(N-1)*N
    row_[1::2] = np.arange(1, N)*N
    col_[::2] = row_[::2] + N
    col_[1::2] = row_[1::2] - N
    val_[::2] = 1/(2*h)
    val_[1::2] = -1/(2*h)

    a = np.ones(N)
    row = np.tensordot(a, row_, axes=0).reshape((-1,))
    col = np.tensordot(a, col_, axes=0).reshape((-1,))
    val = np.tensordot(a, val_, axes=0).reshape((-1,))

    b = np.ones(2*(N-1))
    c = np.arange(N)
    correct = np.tensordot(c, b, axes=0).reshape((-1,))
    row+=correct
    row = np.array(row, dtype=int)
    col+=correct
    col = np.array(col, dtype=int)

    dy = np.zeros((N**2, N**2))
    dy[row, col] = val

    X = np.linspace(h, 1-h, N)
    x, y = np.meshgrid(X, X)

    f = F(x, y).reshape((-1,)).reshape((N**2, -1))
    return dy*f

def d2y(N, F):
    h = 1/(N+1)
    row_, col_, val_ = np.zeros(3*N-2), np.zeros(3*N-2), np.zeros(3*N-2)

    val_[:-1] = np.tile([-2/h**2, 1/h**2, 1/h**2], N-1)
    val_[-1] = -2/h**2
    row_ = (np.tensordot(np.arange(N), np.ones(3), axes=0)*N).reshape((-1,))[1:-1]
    correct_ = np.hstack([np.tile([0, 1, -1], N-1), [0]])*N
    col_ = row_ + correct_

    a = np.ones(N)
    row = np.tensordot(a, row_, axes=0).reshape((-1,))
    col = np.tensordot(a, col_, axes=0).reshape((-1,))
    val = np.tensordot(a, val_, axes=0).reshape((-1,))

    b = np.ones(3*N-2)
    c = np.arange(N)
    correct = np.tensordot(c, b, axes=0).reshape((-1,))
    row+=correct
    row = np.array(row, dtype=int)
    col+=correct
    col = np.array(col, dtype=int)

    d2y = np.zeros((N**2, N**2))
    d2y[row, col] = val

    X = np.linspace(h, 1-h, N)
    x, y = np.meshgrid(X, X)

    f = F(x, y).reshape((-1,)).reshape((N**2, -1))
    return d2y*f

def delta(N, a, b, c, d):
    return dx(N, a) + d2x(N, b) + dy(N, c) + d2y(N, d)

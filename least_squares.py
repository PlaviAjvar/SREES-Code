import numpy as np

# rjesava bazni problem najmanjih kvadrata

def LS(A, b):
    Q = np.transpose(A).dot(A)
    g = np.transpose(A).dot(b)
    x = np.linalg.inv(Q).dot(g)
    return x

# rjesava weighed least squares

def WLS(A, b, W):
    Q = np.transpose(A).dot(W).dot(A)
    g = np.transpose(A).dot(W).dot(b)
    x = np.linalg.inv(Q).dot(g)
    return x

# dobivamo tezinsku matricu iz standardnih devijacija

def weight_matrix(std):
    n = len(std)
    W = np.array([[1 / (std[i]**2) if i == j else 0 for i in range(n)] for j in range(n)])
    return W

# WLS sa equality constraints

def EC_WLS(A, b, W, E, e):
    n = len(A)
    m = len(E)
    Q = np.transpose(A).dot(W).dot(A)
    g = np.transpose(A).dot(W).dot(b)

    # formiramo sistem
    M, y = sys_matrix(Q, E, g, e)
    x = np.linalg.solve(M, y)
    return x[:n]


# EC_WLS u matricnom obliku

def sys_matrix(Q, E, g, e):
    m = len(E)
    zero = np.zeros((m, m))
    M = np.hstack(np.vstack(Q, E), np.vstack(np.transpose(E), zero))
    y = np.vstack(g, e)
    return M, y


# nelinearni WLS Newtonova methoda

def newton_NLLS(W, r, grad, hess, x_init, eps = 1e-5, max_iter = 10000):
    x = x_init

    for iter in range(max_iter):
        g = grad(x)
        H = hess(x)

        # konvergencija
        if all(isclose(di, 0, eps) for di in g):
            return (x, np.transpose(r(x)).dot(W).dot(r(x)))

        dx = -np.linalg.inv(H).dot(g)
        x += dx

    raise Exception("Maximum iterations exceeded")


# pythons default implementation is rather unaplicable here
def isclose(a, b, eps = 1e-9):
    return abs(a - b) < eps


# nelinearni WLS Gauss Newton

def GN_NLLS(W, r, jac, x_init, eps = 1e-5, max_iter = 10000):
    x = x_init

    for iter in range(max_iter):
        r_k = r(x)
        H = jac(x)
        G = np.transpose(H).dot(W).dot(H)
        dx = np.linalg.solve(-G, np.transpose(H).dot(W).dot(r_k))

        if all(isclose(d_i, 0, eps) for d_i in dx):
            return (x, np.transpose(r_k).dot(W).dot(r_k))

        x += dx

    raise Exception("Maximum iterations exceeded")


# nelinearni WLS sa ogranicenjima jednakosti (EC)

def GN_EC_NLLS(W, r, jac, e, jac_con, x_init, eps = 1e-5, max_iter = 10000):
    x = x_init

    for iter in range(max_iter):
        r_k = r(x)
        H = jac(x)
        e_k = e(x)
        E = jac_con(x)

        G = np.transpose(H).dot(W).dot(H)
        g = np.transpose(H).dot(W).dot(r_k)

        M, y = sys_matrix(G, E, -g, -e_k)
        dx = np.linalg.solve(H, y)

        if all(isclose(d_i, 0, eps) for d_i in dx):
            return (x, np.transpose(r_k).dot(W).dot(r_k))

        x += dx

    raise Exception("Maximum iterations exceeded")


import node_types as nt
import numpy as np

# funkcije za doddavanje novih mjerenja

# dodavanje mjerenja injektovane snage u cvor

def add_inject_active(mu, jac, i, Y, p_i):
    i -= 1
    act, g_a = nt.injected_power_active(i, Y, p_i)
    mu.append(act)
    jac.append(g_a)

def add_inject_reactive(mu, jac, i, Y, q_i):
    i -= 1
    react, g_ra = nt.injected_power_reactive(i, Y, q_i)
    mu.append(react)
    jac.append(g_ra)


# dodavanje mjerenja transmisijske snage na grani

def add_transfer_active(mu, jac, i, j, Y_pi, p_i):
    i, j = i-1, j-1
    act, g_a = nt.transfer_power_active(i, j, Y_pi, p_i)
    mu.append(act)
    jac.append(g_a)

def add_transfer_reactive(mu, jac, i, j, Y_pi, q_i):
    i, j = i-1, j-1
    react, g_ra = nt.transfer_power_reactive(i, j, Y_pi, q_i)
    mu.append(react)
    jac.append(g_ra)


# dodavanje mjerenja za napon cvora

def add_voltage_module(mu, jac, i, v_i):
    i -= 1
    v, g_v = nt.voltage(i, v_i)
    mu.append(v)
    jac.append(g_v)


# dodavanje ZI mjerenja

def add_zero_real(eq, jac_con, i, Y):
    i -= 1
    re, g_re = nt.zero_input_real(i, Y)
    eq.append(re)
    jac_con.append(g_re)

def add_zero_image(eq, jac_con, i, Y):
    i -= 1
    im, g_im = nt.zero_input_imag(i, Y)
    eq.append(im)
    jac_con.append(g_im)


# helper functions for calculating matrices for algorithms

def eval_vec(vec_fun, x):
    n = len(vec_fun)
    y = np.array([[0] for _ in range(n)], dtype=float)

    for i in range(n):
        y[i,0] = (vec_fun[i])(np.ndarray.tolist(np.transpose(x))[0])

    return y

def eval_mat(mat_fun, x):
    n = len(mat_fun)
    m = len(x)
    Y = np.array([[0.0] * m for _ in range(n)], dtype=float)

    for i in range(n):
        row = (mat_fun[i])(np.ndarray.tolist(np.transpose(x))[0])
        for j in range(m):
            Y[i,j] = row[j]

    return Y
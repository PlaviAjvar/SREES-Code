from functools import partial
import math
import cmath


# daje vrijednosti r_ip i r_iq

def inj_power(i, Y, p_i, q_i, x):
    Y = [y[1:] for y in Y[1:]]
    n = len(x) // 2
    v, phi = x[:n], x[n:]
    y_ii, theta_ii = cmath.polar(Y[i][i])

    rip = v[i]**2 * y_ii * math.cos(theta_ii) - p_i
    riq = -v[i]**2 * y_ii * math.sin(theta_ii) - q_i

    for j in range(n):
        if i != j:
            y_ij, theta_ij = cmath.polar(Y[i][j])
            rip += v[i] * y_ij * v[j] * math.cos(phi[i] - theta_ij - phi[j])
            riq += v[i] * y_ij * v[j] * math.sin(phi[i] - theta_ij - phi[j])


    return rip, riq


# daje vrijednost gradijenta za mjerenje injektovanje snage

def inj_power_grad(i, Y, x):
    Y = [y[1:] for y in Y[1:]]
    n = len(x) // 2
    v, phi = x[:n], x[n:]
    y_ii, theta_ii = cmath.polar(Y[i][i])

    g_p = [0 for _ in range(2*n)]
    g_q = [0 for _ in range(2*n)]

    g_p[i] = 2 * v[i] * y_ii * math.cos(theta_ii)
    g_q[i] = -2 * v[i] * y_ii * math.sin(theta_ii)

    for j in range(n):
        if i != j:
            y_ij, theta_ij = cmath.polar(Y[i][j])

            g_p[i] += y_ij * v[j] * math.cos(phi[i] - theta_ij - phi[j])
            g_p[n + i] += (-v[i]) * y_ij * v[j] * math.sin(phi[i] - theta_ij - phi[j])

            g_p[j] = v[i] * y_ij * math.cos(phi[i] - theta_ij - phi[j])
            g_p[n + j] = v[i] * y_ij * v[j] * math.sin(phi[i] - theta_ij - phi[j])

            g_q[i] += y_ij * v[j] * math.sin(phi[i] - theta_ij - phi[j])
            g_q[n + i] += v[i] * y_ij * v[j] * math.cos(phi[i] - theta_ij - phi[j])

            g_q[j] = v[i] * y_ij * math.sin(phi[i] - theta_ij - phi[j])
            g_q[n + j] = -v[i] * y_ij * v[j] * math.cos(phi[i] - theta_ij - phi[j])

    return g_p, g_q


# vraca funkciju i gradijent za r_ip i r_iq

def injected_power(i, Y, p_i, q_i):
    inj = partial(inj_power, i, Y, p_i, q_i)
    g_inj = partial(inj_power_grad, i, Y)
    return inj, g_inj

# pomocne funkcije za razdjelu

def injected_power_active(i, Y, p_i):
    inj, g_inj = injected_power(i, Y, p_i, 0)
    act = lambda x : (inj)(x)[0]
    g_a = lambda x : (g_inj)(x)[0]
    return act, g_a

def injected_power_reactive(i, Y, q_i):
    inj, g_inj = injected_power(i, Y, 0, q_i)
    react = lambda x: (inj)(x)[1]
    g_ra = lambda x: (g_inj)(x)[1]
    return react, g_ra


# daje vrijednost rijp i qijp

def trans_power(i, j, Y_pi, p_ij, q_ij, x):
    n = len(x) // 2
    v, phi = x[:n], x[n:]
    y_ii, theta_ii = cmath.polar(Y_pi[0][0])
    y_ij, theta_ij = cmath.polar(Y_pi[0][1])

    rijp = v[i]**2 * y_ii * math.cos(theta_ii) + v[i] * y_ij * v[j] * math.cos(phi[i] - theta_ij - phi[j]) - p_ij
    qijp = -v[i]**2 * y_ii * math.sin(theta_ii) + v[i] * y_ij * v[j] * math.sin(phi[i] - theta_ij - phi[j]) - q_ij

    return rijp, qijp


# daje gradijente za rijp i qijp

def trans_power_grad(i, j, Y_pi, x):
    n = len(x) // 2
    v, phi = x[:n], x[n:]
    y_ii, theta_ii = cmath.polar(Y_pi[0][0])
    y_ij, theta_ij = cmath.polar(Y_pi[0][1])

    g_p = [0 for _ in range(2*n)]
    g_q = [0 for _ in range(2*n)]

    g_p[i] = 2 * v[i] * y_ii * math.cos(theta_ii) + y_ij * v[j] * math.cos(phi[i] - theta_ij - phi[j])
    g_p[n + i] = -v[i] * y_ij * v[j] * math.sin(phi[i] - theta_ij - phi[j])

    g_p[j] = v[i] * y_ij * math.cos(phi[i] - theta_ij - phi[j])
    g_p[n + j] = v[i] * y_ij * v[j] * math.sin(phi[i] - theta_ij - phi[j])

    g_q[i] = -2 * v[i] * y_ii * math.sin(theta_ii) + y_ij * v[j] * math.sin(phi[i] - theta_ij - phi[j])
    g_q[n + i] = v[i] * y_ij * v[j] * math.cos(phi[i] - theta_ij - phi[j])

    g_q[j] = v[i] * y_ij * math.sin(phi[i] - theta_ij - phi[j])
    g_q[n + j] = -v[i] * y_ij * v[j] * math.cos(phi[i] - theta_ij - phi[j])

    return g_p, g_q


# vraca funkciju i gradijent za r_ijp i r_ijq

def transfer_power(i, j, Y_pi, p_ij, q_ij):
    trans = partial(trans_power, i, j, Y_pi, p_ij, q_ij)
    g_trans = partial(trans_power_grad, i, j, Y_pi)
    return trans, g_trans

# pomocne fje za razdjelu

def transfer_power_active(i, j, Y_pi, p_ij):
    trans, g_trans = transfer_power(i, j, Y_pi, p_ij, 0)
    act = lambda x: (trans)(x)[0]
    g_a = lambda x: (g_trans)(x)[0]
    return act, g_a

def transfer_power_reactive(i, j, Y_pi, q_ij):
    trans, g_trans = transfer_power(i, j, Y_pi, 0, q_ij)
    react = lambda x: (trans)(x)[1]
    g_ra = lambda x: (g_trans)(x)[1]
    return react, g_ra


# mjerenje modula napona

def volt(i, v_i, x):
    n = len(x) // 2
    v, phi = x[:n], x[n:]
    return v[i] - v_i

# gradijent trivijalan

def volt_grad(i, x):
    n = len(x) // 2
    g_v = [0 for _ in range(2*n)]
    g_v[i] = 1
    return g_v

# vraca funkciju i gradijent za r_iv

def voltage(i, v_i):
    v = partial(volt, i, v_i)
    g_v = partial(volt_grad, i)
    return v, g_v



# vraca vrijednost funkcija za ZI mjerenja

def zero(i, Y, x):
    Y = [y[1:] for y in Y[1:]]
    n = len(Y)
    v, phi = x[:n], x[n:]
    re = 0
    im = 0

    for j in range(n):
        y_ij, theta_ij = cmath.polar(Y[i][j])
        re += y_ij * v[j] * math.cos(theta_ij + phi[j])
        im += y_ij * v[j] * math.sin(theta_ij + phi[j])

    return re, im


# vraca vrijednost gradijenta za ZI mjerenja

def grad_zero(i, Y, x):
    Y = [y[1:] for y in Y[1:]]
    n = len(Y)
    v, phi = x[:n], x[n:]
    g_re = [0 for _ in range(2*n)]
    g_im = [0 for _ in range(2*n)]

    for j in range(n):
        y_ij, theta_ij = cmath.polar(Y[i][j])

        g_re[j] = y_ij * math.cos(theta_ij + phi[j])
        g_re[n + j] = -y_ij * v[j] * math.sin(theta_ij + phi[j])

        g_im[j] = y_ij * math.sin(theta_ij + phi[j])
        g_im[n + j] = y_ij * v[j] * math.cos(theta_ij + phi[j])

    return g_re, g_im


# funkcija i gradijent za ZI mjerenja

def zero_input(i, Y):
    ZI = partial(zero, i, Y)
    g_zi = partial(grad_zero, i, Y)
    return ZI, g_zi

# pomocne fje za razdvajanje

def zero_input_real(i, Y):
    ZI, g_zi = zero_input(i, Y)
    re = lambda x : (ZI)(x)[0]
    g_re = lambda x : (g_zi)(x)[0]
    return re, g_re

def zero_input_imag(i, Y):
    ZI, g_zi = zero_input(i, Y)
    im = lambda x : (ZI)(x)[1]
    g_im = lambda x : (g_zi)(x)[1]
    return im, g_im



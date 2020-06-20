import numpy as np
import call_helper
import least_squares
import cmath
from functools import partial


# uz zanemarenje shantovskih admitansi

def line_eq(z_12):
    return [[1 / z_12, -1 / z_12], [-1 / z_12, 1 / z_12]]

def adm_matrix(z):
    n = len(z)
    Y = [[0] * 4 for _ in range(4)]

    for i in range(1, n):
        for j in range(1, n):
            if i == j:
                Y[i][i] = 0
                for k in range(1, n):
                    if k != i:
                        Y[i][i] += 1 / z[i][k]
            else:
                Y[i][j] = -1 / z[i][j]

    return Y


if __name__ == "__main__":
    # formirati inicijalnu tacku x_init
    x_init = np.array([[2], [2], [2], [1], [1], [1]], dtype=float)

    # formirati matricu tezina W
    std = [0.008, 0.008, 0.010, 0.008, 0.008, 0.010, 0.004, 0.004]
    W = least_squares.weight_matrix(std)

    # formirati pi ekvivalente linije i matricu admitansi sistema
    z = [[0] * 4 for _ in range(4)]
    z[1][2] = z[2][1] = 0.01 + 0.03*1j
    z[1][3] = z[3][1] = 0.02 + 0.05*1j
    z[2][3] = z[3][2] = 0.03 + 0.08*1j

    Y12 = line_eq(z[1][2])
    Y13 = line_eq(z[1][3])
    Y = adm_matrix(z)

    # formirati gresku r i jakobijan H
    r = []
    jac = []

    call_helper.add_transfer_active(r, jac, 1, 2, Y12, 0.88)
    call_helper.add_transfer_active(r, jac, 1, 3, Y13, 1.173)
    call_helper.add_inject_active(r, jac, 2, Y, -0.5)

    call_helper.add_transfer_reactive(r, jac, 1, 2, Y12, 0.568)
    call_helper.add_transfer_reactive(r, jac, 1, 3, Y13, 0.663)
    call_helper.add_inject_reactive(r, jac, 2, Y, -0.286)

    call_helper.add_voltage_module(r, jac, 1, 1.006)
    call_helper.add_voltage_module(r, jac, 2, 0.968)

    # pretvoriti r i jac u callable funkcije
    r_f = partial(call_helper.eval_vec, r)
    jac_f = partial(call_helper.eval_mat, jac)

    # pozvati GN funkciju
    (x, sq_err) = least_squares.GN_NLLS(W, r_f, jac_f, x_init, eps=1e-5)
    print("x =", x)
    print("Energy =", sq_err)




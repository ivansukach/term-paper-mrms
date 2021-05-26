import numpy as np
from scipy.optimize import leastsq, least_squares
import sys


def f(_y, _t):
    rhp = np.zeros_like(_y)
    rhp[0] = -1.71 * _y[0] + 0.43 * _y[1] + 8.32 * _y[2] + 0.0007
    rhp[1] = 1.71 * _y[0] - 8.75 * _y[1]
    rhp[2] = -10.03 * _y[2] + 0.43 * _y[3] + 0.035 * _y[4]
    rhp[3] = 8.32 * _y[1] + 1.71 * _y[2] - 1.12 * _y[3]
    rhp[4] = -1.745 * _y[4] + 0.43 * _y[5] + 0.43 * _y[6]
    rhp[5] = -280 * _y[5] * _y[7] + 0.69 * _y[3] + 1.71 * _y[4] - 0.43 * _y[5] + 0.69 * _y[6]
    rhp[6] = 280 * _y[5] * _y[7] - 1.81 * _y[6]
    rhp[7] = - rhp[6]
    return rhp


def g_const():
    s = np.zeros(size_of_y)
    for i in range(1, p + 1):
        s += c[p - i] * np.array(y[-i])
    return s


def residual(_x):
    return tau * f(_x, t[-1]+tau) - c[p] * _x - g


def v_transpose():
    _v = []
    for i in range(0, k):
        _v.append(-1 * np.asarray(y[-(k-i)]))
    for m in range(0, k):
        _v.append(tau * f(y[-(k-m)], t[-(k-m)]))
    return np.asarray(_v)


def dfi_dyj(i, j, _x):
    if i == 0:
        if j == 0:
            return -1.71
        if j == 1:
            return 0.43
        if j == 2:
            return 8.32
        else:
            return 0
    if i == 1:
        if j == 0:
            return 1.71
        if j == 1:
            return -8.75
        else:
            return 0
    if i == 2:
        if j == 2:
            return -10.03
        if j == 3:
            return 0.43
        if j == 4:
            return 0.035
        else:
            return 0
    if i == 3:
        if j == 1:
            return 8.32
        if j == 2:
            return 1.71
        if j == 3:
            return -1.12
        else:
            return 0
    if i == 4:
        if j == 4:
            return -1.745
        if j == 5:
            return 0.43
        if j == 6:
            return 0.43
        else:
            return 0
    if i == 5:
        if j == 3:
            return 0.69
        if j == 4:
            return 1.71
        if j == 5:
            return -0.43 - 280 * _x[7]
        if j == 6:
            return 0.69
        if j == 7:
            return -280 * _x[5]
        else:
            return 0
    if i == 6:
        if j == 5:
            return 280 * _x[7]
        if j == 6:
            return -1.81
        if j == 7:
            return 280 * _x[5]
        else:
            return 0
    if i == 7:
        if j == 5:
            return -280 * _x[7]
        if j == 6:
            return 1.81
        if j == 7:
            return -280 * _x[5]
        else:
            return 0


def transposed_f_gradient(_x):
    _gradient = np.zeros((size_of_y, size_of_y))
    for i in range(0, size_of_y):
        for j in range(0, size_of_y):
            _gradient[j][i] = dfi_dyj(i, j, _x)
    return _gradient


def f_gradient(_x):
    _gradient = np.zeros((size_of_y, size_of_y))
    for i in range(0, size_of_y):
        for j in range(0, size_of_y):
            _gradient[i][j] = dfi_dyj(i, j, _x)
    return _gradient


def transposed_jacobi_matrix(_x):
    print("transposed_f_gradient: ", transposed_f_gradient(_x))
    m_jacobi_t = v_transposed.dot(tau * transposed_f_gradient(_x) - I)
    print("TRANSPOSED JACOBI MATRIX: ", m_jacobi_t)
    return m_jacobi_t


def jacobi_matrix(_gamma):
    _x = v.dot(_gamma)
    # print("transposed_f_gradient: ", transposed_f_gradient(_x))
    m_jacobi_t = (tau * f_gradient(_x) - I).dot(v)
    # print("JACOBI MATRIX: ", m_jacobi_t)
    return m_jacobi_t


def residual_by_gamma(_gamma):
    _x = v.dot(_gamma)
    return residual(_x)


def norm_of_residual_by_gamma(_gamma):
    _residual = residual_by_gamma(_gamma)
    _sum = 0
    for i in range(0, size_of_y):
        _sum += _residual[i] ** 2
    return _sum


# amount_of_points = 8
# size_of_y = len(y[0])
eps = 0.001
c = [1/2, -2, 3/2]
k = 3
p = 2
# f = new_f(size_of_y)
y = [[1.0, 0, 0, 0, 0, 0, 0, 0.0057],
     [9.99999983e-01, 1.70999985e-08, 4.35663436e-24, 1.20060748e-15, 2.53677461e-32, 6.99087886e-24, 9.41556329e-32, 5.70000000e-03],
     [9.99999966e-01, 3.41999940e-08, 3.48530685e-23, 4.80242944e-15, 4.05883890e-31, 5.59270245e-23, 1.50648993e-30, 5.70000000e-03]]
size_of_y = len(y[0])
t_min = 0
t_max = 0.0001
t = [t_min, 0.00000001, 0.00000002]
# tau = (t_max - t_min) / (amount_of_points - 1)
tau = 0.00000001
I = np.zeros((size_of_y, size_of_y))
for _i in range(0, size_of_y):
    I[_i][_i] = 1
g = g_const()
v_transposed = v_transpose()
v = v_transposed.transpose()
gamma = np.array([1, 1, 1, 1, 1, 1])
print("V: ", v)
x = v.dot(gamma)
print("RESIDUAL: ", residual(x))
gradient = 2 * transposed_jacobi_matrix(x).dot(residual(x))
print("GRADIENT: ", gradient)
print("NORM OF RESIDUAL: ", norm_of_residual_by_gamma(gamma))


while t[-1] < t_max:
    gamma_tmp = least_squares(residual_by_gamma, gamma, jac=jacobi_matrix).x
    # bounds_min = gamma_tmp - 1e-3
    # bounds_max = gamma_tmp + 1e-3
    # gamma_tmp = least_squares(residual_by_gamma, gamma_tmp, jac=jacobi_matrix).x
    r_norm = norm_of_residual_by_gamma(gamma_tmp)

    # while r_norm > eps:
    #     # tau = tau / math.sqrt(r_norm / eps)
    #     # tau = tau * eps / r_norm
    #     # tau = tau / math.sqrt(r_norm / eps)
    #     gamma_tmp = least_squares(norm_of_residual_by_gamma, gamma).x
    #     r_norm = norm_of_residual_by_gamma(gamma_tmp)
    if r_norm > eps:
        print("BIG RESIDUAL")
        sys.exit()
    t.append(t[-1]+tau)
    y.append(v.dot(gamma_tmp))
    g = g_const()
    v_transposed = v_transpose()
    v = v_transposed.transpose()
    # tau = tau * eps / r_norm
    # tau = tau / math.sqrt(r_norm / eps)


print("Y:", y[-1])
_s = 0
for i in range(0, size_of_y):
    _s += y[-1][i]
print("Sum: ", _s)
print("T: ", t[-1])
import numpy as np
from scipy.optimize import leastsq, least_squares
import math


def jk(_y, _k):
    return _y[0] * _y[_k - 1] - math.exp(math.pow(_k, 2 / 3) - math.pow((_k - 1), 2 / 3)) * _y[_k]


def new_f(n):
    def _f(_y, _x):
        rhp = np.zeros_like(_y)
        for i in range(0, n):
            if i == 0:
                s = 0
                s -= 2 * jk(_y, 1)
                for j in range(1, n - 1):
                    s -= jk(_y, j + 1)
                rhp[0] = s
            elif i == n - 1:
                rhp[n - 1] = jk(_y, n - 1)
            else:
                rhp[i] = jk(_y, i) - jk(_y, i + 1)
        return rhp

    return _f


amount_of_points = 7
size_of_y = 3
eps = 0.001
c = [-1, 1]
k = 1
p = 1
f = new_f(size_of_y)
y = [[7.5, 0, 0]]
t_min = 0
t_max = 1
t = np.linspace(t_min, t_max, amount_of_points)
tau = (t_max - t_min) / (amount_of_points - 1)
I = np.zeros((size_of_y, size_of_y))
for _i in range(0, size_of_y):
    I[_i][_i] = 1


def g_const():
    s = np.zeros(size_of_y)
    for i in range(1, p + 1):
        s += c[k - i] * np.array(y[k - i])
    return s


g = g_const()


def residual(_x):
    return tau * f(_x, t[k]) - c[k] * _x - g


def v_transpose():
    _v = []
    for i in range(0, k):
        _v.append(-1 * np.asarray(y[i]))
    for m in range(0, k):
        _v.append(tau * f(y[m], t[m]))
    return np.asarray(_v)


v_transposed = v_transpose()
v = v_transposed.transpose()


def dfi_dyj(i, j, _x):
    if i == 0:
        if j == 0:
            s = 0
            for _k in range(1, size_of_y):
                s += _x[_k]
            return -4 * _x[0] - s
        if j == 1:
            return 2 * math.exp(1) - _x[0]
        if j == size_of_y - 1:
            return math.exp(math.pow(size_of_y - 1, 2 / 3) - math.pow(size_of_y - 2, 2 / 3))
        else:
            return -_x[0] + math.exp(math.pow(j, 2 / 3) - math.pow(j - 1, 2 / 3))
    if i == size_of_y - 1:
        if j == 0:
            return _x[i - 1]
        if j == size_of_y - 2:
            return _x[0]
        if j == size_of_y - 1:
            return -math.exp(math.pow(size_of_y - 1, 2 / 3) - math.pow(size_of_y - 2, 2 / 3))
        else:
            return 0
    else:
        if j == 0:
            if i == 1:
                return 2 * _x[0] - _x[1]
            else:
                return _x[i - 1] - _x[i]
        if j == i - 1:
            return _x[0]
        if j == i:
            return -math.exp(math.pow(j, 2 / 3) - math.pow(j - 1, 2 / 3)) - _x[0]
        if j == i + 1:
            return math.exp(math.pow(i + 1, 2 / 3) - math.pow(i, 2 / 3))
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


gamma = np.array([1, 1])
print("V: ", v)
x = v.dot(gamma)
print("RESIDUAL: ", residual(x))
gradient = 2 * transposed_jacobi_matrix(x).dot(residual(x))
print("GRADIENT: ", gradient)


# def func(_x):
#     return (_x[0] - 3) ** 2 + (_x[1] - 7) ** 2
#
#
# #print(least_squares(func, x0=np.array([0, 0])))
# print(leastsq(func, x0=np.array([0, 0])))
def residual_by_gamma(_gamma):
    _x = v.dot(_gamma)
    return residual(_x)


def norm_of_residual_by_gamma(_gamma):
    _residual = residual_by_gamma(_gamma)
    _sum = 0
    for i in range(0, size_of_y):
        _sum += _residual[i] ** 2
    return _sum


print("NORM OF RESIDUAL: ", norm_of_residual_by_gamma(gamma))
# print("DEFAULT?? scheme of Jacobi matrix approximation ")
# print("WHY RESIDUAL NORM != cost")
# gamma1 = least_squares(norm_of_residual_by_gamma, gamma).x
# print("GAMMA1:", gamma1)
# print("Residual by gamma:", residual_by_gamma(gamma1))
# print("NORM OF RESIDUAL1:", norm_of_residual_by_gamma(gamma1))
# gamma2 = least_squares(residual_by_gamma, gamma1).x
# print("GAMMA2:", gamma2)
# print("RESIDUAL2:", residual_by_gamma(gamma2))
# print("NORM OF RESIDUAL2:", norm_of_residual_by_gamma(gamma2))
# gamma3 = least_squares(residual_by_gamma, gamma2).x
# print("GAMMA3:", gamma3)
# print("RESIDUAL3:", residual_by_gamma(gamma3))
# print("NORM OF RESIDUAL3:", norm_of_residual_by_gamma(gamma3))
#
# print("3-point scheme of Jacobi matrix approximation ")
# gamma1_3point = least_squares(residual_by_gamma, gamma, jac="3-point").x
# print("GAMMA1 :", gamma1_3point)
# print("RESIDUAL1:", residual_by_gamma(gamma1_3point))
# print("NORM OF RESIDUAL1:", norm_of_residual_by_gamma(gamma1_3point))
# gamma2_3point = least_squares(residual_by_gamma, gamma1_3point, jac="3-point").x
# print("GAMMA2:", gamma2_3point)
# print("RESIDUAL2:", residual_by_gamma(gamma2_3point))
# print("NORM OF RESIDUAL2:", norm_of_residual_by_gamma(gamma2_3point))
# gamma3_3point = least_squares(residual_by_gamma, gamma2_3point, jac="3-point").x
# print("GAMMA3:", gamma3_3point)
# print("RESIDUAL3:", residual_by_gamma(gamma3_3point))
# print("NORM OF RESIDUAL3:", norm_of_residual_by_gamma(gamma3_3point))
#
# print("Exact Jacobi Matrix")
# gamma1_exact = least_squares(residual_by_gamma, gamma, jac=jacobi_matrix).x
# print("GAMMA1 :", gamma1_exact)
# print("RESIDUAL1:", residual_by_gamma(gamma1_exact))
# print("NORM OF RESIDUAL1:", norm_of_residual_by_gamma(gamma1_exact))
# gamma2_exact = least_squares(residual_by_gamma, gamma1_exact, jac=jacobi_matrix).x
# print("GAMMA2:", gamma2_exact)
# print("RESIDUAL2:", residual_by_gamma(gamma2_exact))
# print("NORM OF RESIDUAL2:", norm_of_residual_by_gamma(gamma2_exact))
# gamma3_exact = least_squares(residual_by_gamma, gamma2_exact, jac=jacobi_matrix).x
# print("GAMMA3:", gamma3_exact)
# print("RESIDUAL3:", residual_by_gamma(gamma3_exact))
# print("NORM OF RESIDUAL3:", norm_of_residual_by_gamma(gamma3_exact))

gamma_tmp = least_squares(norm_of_residual_by_gamma, gamma).x
r_norm = norm_of_residual_by_gamma(gamma_tmp)
while r_norm > eps:
    tau = tau / math.sqrt(r_norm / eps)
    gamma_tmp = least_squares(norm_of_residual_by_gamma, gamma).x
    r_norm = norm_of_residual_by_gamma(gamma_tmp)
tau = tau * math.sqrt(r_norm / eps)

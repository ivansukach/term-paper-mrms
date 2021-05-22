import numpy as np
from scipy.optimize import leastsq, least_squares
from scipy.integrate import solve_ivp
import math


def jk(_y, _k):
    return _y[0] * _y[_k - 1] - math.exp(math.pow(_k, 2 / 3) - math.pow((_k - 1), 2 / 3)) * _y[_k]


def f(_y, _t):
    n = len(_y)
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


def residual_by_gamma(_gamma):
    _x = v.dot(_gamma)
    return residual(_x)


def norm_of_residual_by_gamma(_gamma):
    _residual = residual_by_gamma(_gamma)
    _sum = 0
    for i in range(0, size_of_y):
        _sum += _residual[i] ** 2
    return _sum


amount_of_points = 7
size_of_y = 7
eps = 0.001
c = [1/2, -2, 3/2]
k = 3
p = 2
# f = new_f(size_of_y)
y = [[19.0, 0, 0, 0, 0, 0, 0],
     [1.89999928e+01, 3.60999702e-06, 5.78817767e-13, 9.28061727e-20, 1.48803064e-26, 2.38587059e-33, 3.82544501e-40],
     [1.89999856e+01, 7.21998809e-06, 2.31526787e-12, 7.42447946e-19, 2.38084310e-25, 7.63476267e-32, 2.44827640e-38]]
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


# def func(_x):
#     return (_x[0] - 3) ** 2 + (_x[1] - 7) ** 2
#
#
# #print(least_squares(func, x0=np.array([0, 0])))
# print(leastsq(func, x0=np.array([0, 0])))


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

# print(y[0])
# sol = solve_ivp(f, [t_min, t_max], np.array(y[0]))
# print(sol)

while t[-1] < t_max:
    gamma_tmp = least_squares(residual_by_gamma, gamma).x
    r_norm = norm_of_residual_by_gamma(gamma_tmp)
    while r_norm > eps:
        # tau = tau / math.sqrt(r_norm / eps)
        # tau = tau * eps / r_norm
        # tau = tau / math.sqrt(r_norm / eps)
        gamma_tmp = least_squares(norm_of_residual_by_gamma, gamma).x
        r_norm = norm_of_residual_by_gamma(gamma_tmp)
    # g = g_const()
    t.append(t[-1]+tau)
    y.append(v.dot(gamma_tmp))
    g = g_const()
    v_transposed = v_transpose()
    v = v_transposed.transpose()
    # tau = tau * eps / r_norm
    # tau = tau / math.sqrt(r_norm / eps)


print("Y:", y[-1])
print("T: ", t[-1])

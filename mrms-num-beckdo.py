import numpy as np
from scipy.optimize import leastsq, least_squares
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import math
import sys


def jk(_y, _k):
    return _y[0] * _y[_k - 1] - math.exp(math.pow(_k, 2 / 3) - math.pow((_k - 1), 2 / 3)) * _y[_k]


def f(_t, _y):
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
    return tau * f(t[-1]+tau, _x) - c[p] * _x - g


def v_transpose():
    _v = []
    for i in range(0, k):
        _v.append(-1 * np.asarray(y[-(k-i)]))
    for m in range(0, k):
        _v.append(tau * f(t[-(k-m)], y[-(k-m)]))
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


def norm(_y_diff):
    _sum = 0
    for i in range(0, size_of_y):
        _sum += _y_diff[i] ** 2
    return _sum


amount_of_points = 7
size_of_y = 7
eps = 0.001
c = [-1/3, 1.5, -3, 11/6]
k = 3
p = 3
y = [[19.0, 0, 0, 0, 0, 0, 0],
     [1.89992780e+01, 3.60973293e-04, 5.19256610e-09, 7.19493492e-14,
      9.90970457e-19, 1.36352986e-23, 1.87587333e-28],
     [1.89985561e+01, 7.21911493e-04, 1.72078949e-08, 3.41111307e-13,
      6.14807852e-18, 1.04688867e-22, 1.71727086e-27]]
t_min = 0
t_max = 0.01
t = [t_min, 1e-6, 2e-6]
# tau = (t_max - t_min) / (amount_of_points - 1)
tau = 1e-6
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
counter = 0

while t[-1] < t_max:
    gamma_tmp1 = least_squares(residual_by_gamma, gamma, jac=jacobi_matrix, xtol=3e-16).x
    r_norm = norm_of_residual_by_gamma(gamma_tmp1)
    bounds_min = gamma_tmp1 - 1e-3
    # bounds_max = gamma_tmp + 1e-3
    gamma_tmp = least_squares(residual_by_gamma, bounds_min, jac=jacobi_matrix).x
    r_norm2 = norm_of_residual_by_gamma(gamma_tmp)
    if r_norm2 > r_norm:
        gamma_tmp = gamma_tmp1
        counter += 1


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
print("Gamma recalculation turned out to be useless " + str(counter) + " times")
for i in range(0, len(t)):
    print("T: ", t[i], " Y: ", y[i])

print("")
print("")
print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
print("")
print("")

bdf_solution = solve_ivp(f, [t_min, t_max], np.array(y[0]), method='BDF', dense_output=True)
t_diff = []
y_diff_norms = []
for i in range(0, 100):
    t_diff.append(t[i*100])
    tmp_bdf_sol = bdf_solution.sol(t_diff[-1])
    print("T: ", t_diff[-1], " Y: ", tmp_bdf_sol)
    y_diff_norms.append(norm(tmp_bdf_sol-y[i*100]))

fig, ax = plt.subplots()
ax.plot(t_diff, y_diff_norms, color='brown')
plt.grid()
plt.legend(loc='best')
plt.show()

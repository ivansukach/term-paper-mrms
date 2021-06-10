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


y0 = [7.5, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
t_min = 0
t_max = 1
bdf_solution = solve_ivp(f, [t_min, t_max], np.array(y0), method='BDF', dense_output=True)
y_bdf_sol = bdf_solution.y.transpose()
size_of_y = 100
eps = 0.001
c = [-1/3, 1.5, -3, 11/6]
k = 3
p = 3
y = [y0, y_bdf_sol[1], y_bdf_sol[2]]
t = [t_min, bdf_solution.t[1], bdf_solution.t[2]]
# tau = (t_max - t_min) / (amount_of_points - 1)
tau = bdf_solution.t[2] - bdf_solution.t[1]
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
    gamma_tmp = least_squares(residual_by_gamma, gamma, jac=jacobi_matrix, xtol=3e-16).x
    r_norm = norm_of_residual_by_gamma(gamma_tmp)
    if r_norm > eps:
        print("BIG RESIDUAL")
        sys.exit()
    t.append(t[-1]+tau)
    y.append(v.dot(gamma_tmp))
    g = g_const()
    v_transposed = v_transpose()
    v = v_transposed.transpose()


print("Y:", y[-1])
_s = 0
for i in range(0, size_of_y):
    _s += y[-1][i]
print("Sum: ", _s)
print("T: ", t[-1])
print(str(len(t)) + " iterations")
y_diff_norms = []
tau_eps = (t[1] - t[0])/100
for i in range(0, len(bdf_solution.t)):
    for j in range(0, len(t)):
        if math.fabs(bdf_solution.t[i] - t[j]) < tau_eps:
            y_diff_norms.append(norm(y_bdf_sol[i]-y[j]))
            break

fig, ax = plt.subplots()
ax.plot(bdf_solution.t, y_diff_norms, color='brown')
plt.grid()
plt.legend(loc='best')
plt.show()

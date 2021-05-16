import numpy as np
import math


def jk(_y, _k):
    return _y[0] * _y[_k-1] - math.exp(math.pow(_k, 2 / 3) - math.pow((_k - 1), 2 / 3)) * _y[_k]


def new_f(n):
    def _f(_y, _x):
        rhp = np.zeros_like(_y)
        for i in range(0, n):
            if i == 0:
                s = 0
                s -= 2*jk(_y, 1)
                for j in range(1, n-1):
                    s -= jk(_y, j+1)
                rhp[0] = s
            elif i == n-1:
                rhp[n-1] = jk(_y, n - 1)
            else:
                rhp[i] = jk(_y, i) - jk(_y, i+1)
        return rhp
    return _f


amount_of_points = 7
size_of_y = 3
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
    for i in range(1, p+1):
        s += c[k-i]*np.array(y[k-i])
    return s


g = g_const()


def residual(_x):
    return tau * f(_x, t[k]) - c[k] * _x - g


def v_transpose():
    _v = []
    for i in range(0, k):
        _v.append(-1*np.asarray(y[i]))
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
            return math.exp(math.pow(size_of_y-1, 2/3)-math.pow(size_of_y-2, 2/3))
        else:
            return -_x[0] + math.exp(math.pow(j, 2 / 3) - math.pow(j - 1, 2 / 3))
    if i == size_of_y - 1:
        if j == 0:
            return _x[i-1]
        if j == size_of_y - 2:
            return _x[0]
        if j == size_of_y - 1:
            return -math.exp(math.pow(size_of_y-1, 2/3)-math.pow(size_of_y-2, 2/3))
        else:
            return 0
    else:
        if j == 0:
            if i == 1:
                return 2*_x[0] - _x[1]
            else:
                return _x[i - 1] - _x[i]
        if j == i-1:
            return _x[0]
        if j == i:
            return -math.exp(math.pow(j, 2/3)-math.pow(j-1, 2/3)) - _x[0]
        if j == i+1:
            return math.exp(math.pow(i+1, 2/3)-math.pow(i, 2/3))
        else:
            return 0


def transposed_f_gradient(_x):
    _gradient = np.zeros((size_of_y, size_of_y))
    for i in range(0, size_of_y):
        for j in range(0, size_of_y):
            _gradient[j][i] = dfi_dyj(i, j, _x)
    return _gradient


def transposed_jacobi_matrix(_x):
    print("transposed_f_gradient: ", transposed_f_gradient(_x))
    m_jacobi_t = v_transposed.dot(tau*transposed_f_gradient(_x) - I)
    print("JACOBI MATRIX: ", m_jacobi_t)
    return m_jacobi_t


gamma = np.array([1, 1])
print("V: ", v)
x = v.dot(gamma)
print("RESIDUAL: ", residual(x))
gradient = 2 * transposed_jacobi_matrix(x).dot(residual(x))
print("GRADIENT: ", gradient)

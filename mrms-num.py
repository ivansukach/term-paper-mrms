import numpy as np
import math


def jk(_y, _k):
    return _y[0] * _y[_k] - math.exp(math.pow((_k + 1), 2 / 3) - math.pow(_k, 2 / 3)) * _y[_k + 1]


def new_f(n):
    def _f(_y, _x):
        rhp = np.zeros_like(_y)
        for i in range(0, n):
            if i == 0:
                s = 0
                s -= 2*jk(_y, 0)
                for j in range(1, n-2):
                    s -= jk(_y, j)
                rhp[0] = s
            elif i == n-1:
                rhp[n-1] = jk(_y, n - 2)
            else:
                rhp[i] = jk(_y, i - 1) - jk(_y, i)
        return rhp
    return _f


amount_of_points = 7
size_of_y = 4
c = [-1 / 3, 3 / 2, -3, 11 / 6]
k = 3
p = 3
f = new_f(size_of_y)
y = [[7.5, 0, 0, 0],
 [1.52978522, 0.86092722, 0.73196761, 0.6841525],
 [1.52978522, 0.86092722, 0.73196761, 0.6841525],
 [1.52978522, 0.86092722, 0.73196761, 0.6841525]]
t_min = 0
t_max = 1
t = np.linspace(t_min, t_max, amount_of_points)
tau = (t_max - t_min) / (amount_of_points - 1)


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


v_transpose = v_transpose()
v = v_transpose.transpose()


def dfi_dyj(i, j, _x):
    if i == 0:
        if j == 0:
            s = 0
            for _k in range(1, size_of_y):
                s += _x[k]
            return -4 * _x[0] - s
        if j == 1:
            return 2 * math.exp(1) - _x[0]
        if j == size_of_y - 1:
            return math.exp(math.pow(size_of_y-1, 2/3)-math.pow(size_of_y-2, 2/3))
        else:
            return -_x[0] + math.exp(math.pow(j, 2 / 3) - math.pow(j - 1, 2 / 3))
    if i == size_of_y - 2:
        if j == 0:
            return _x[i]
        if j == i:
            return _x[0]
        if j == i+1:
            return -math.exp(math.pow(size_of_y-1, 2/3)-math.pow(size_of_y-2, 2/3))
        else:
            return 0
    else:
        if j == 0:
            return _x[i - 1] - _x[i]
        if j == i-1:
            return _x[0]
        if j == i:
            return -math.exp(math.pow(i, 2/3)-math.pow(i-1, 2/3)) - _x[0]
        if j == i+1:
            return math.exp(math.pow(i+1, 2/3)-math.pow(i, 2/3))
        else:
            return 0


def transposed_f_gradient(i, _x):
    _gradient = np.zeros(size_of_y)
    for j in range(0, size_of_y):
        _gradient[j] = dfi_dyj(i, j, _x)
    return _gradient


def transposed_jacobi_matrix(_x):
    m_jacobi_t = np.zeros((2*k, size_of_y))
    for j in range(0, 2*k):
        for i in range(0, size_of_y):
            f_i_gradient = transposed_f_gradient(i, _x)
            m_jacobi_t[j][i] = tau * f_i_gradient.dot(v_transpose[j]) - c[k] * _x[i]
    print("JACOBI MATRIX: ", m_jacobi_t)
    return m_jacobi_t


gamma = np.array([1, 1, 1, 1, 1, 1])
print("V: ", v)
x = v.dot(gamma)
print("RESIDUAL: ", residual(x))
gradient = 2 * transposed_jacobi_matrix(x).dot(residual(x))
print("GRADIENT: ", gradient)

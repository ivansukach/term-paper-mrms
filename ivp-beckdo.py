import numpy as np
from scipy.optimize import leastsq, least_squares
from scipy.integrate import solve_ivp
import math


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


y = [[8.0, 0, 0, 0, 0, 0, 0, 0]]
t_min = 0
t_max = 0.01

print(y[0])
solution = solve_ivp(f, [t_min, t_max], np.array(y[0]), method='BDF', dense_output=True)
print(solution)
# print(sol.y[0][5]+sol.y[1][5]+sol.y[2][5])
print(solution.sol(t_max))
y_last = solution.sol(t_max)
_s = 0
for i in range(0, len(y[0])):
    _s += y_last[i]
print("y0:", solution.y[0])
print("y1:", solution.y[1])
# print("Sum: ", _s)
# # print("T: ", solution.t)
# for i in range(0, len(solution.t)):
#     print("T: ", solution.t[i], " Y: ", (solution.y.transpose())[i])

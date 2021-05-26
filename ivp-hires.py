import numpy as np
from scipy.integrate import solve_ivp


def f(_t, _y):
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


y = [[1.0, 0, 0, 0, 0, 0, 0, 0.0057]]
t_min = 0
t_max = 0.0001

print(y[0])
solution = solve_ivp(f, [t_min, t_max], np.array(y[0]), method='BDF', dense_output=True)
print(solution)
print(solution.sol(t_max))

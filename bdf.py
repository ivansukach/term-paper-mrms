import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.integrate import *
from scipy.optimize import minimize_scalar


def jk(_y, k):
    res = _y[0]*_y[k]-math.exp(math.pow((k+1), 2/3)-math.pow(k, 2/3))*_y[k+1]
    return res


def new_f(n):
    def f(_y, _x):
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
    return f





def backward_differentiation_formula(y):
    k = 2
    while k < amount_of_points - 2:
        y[k+1] = (6/11)*(3*y[k]-(3/2)*y[k-1]+(1/3)*y[k-2]+h*f(y[k+1], x[k+1]))
        x_temp = x[k+1:k+3]
        # print("X_TEMP: ", x_temp)
        new_yk = np.array([odeint(f, y[k+1], x_temp)[-1]]).tolist()
        # print(new_yk)
        y = y.tolist() + new_yk
        y = np.asarray(y)
        # print("new list", y)
        k += 1
    print("latest k = ", k)
    return y


fig, ax = plt.subplots()
amount_of_points = 101
size_of_y = 300
f = new_f(size_of_y)
x_min = 0
# x_max = 10**15
x_max = 10
x = np.linspace(x_min, x_max, amount_of_points)
h = (x_max - x_min)/(amount_of_points-1)
y0 = 7.5
print("X[] = ", x)
y_initial_values = np.zeros([size_of_y])
y_initial_values[0] = 7.5
x_initial_values_for_bdf = x[:4]
y_initial_values_for_bdf = odeint(f, y_initial_values, x_initial_values_for_bdf)
x_initial_values_for_bdf = x[:4]
print("Y INITIAL VALUES:", y_initial_values_for_bdf)
print("X INITIAL VALUES:", x_initial_values_for_bdf)
y_solution = odeint(f, y_initial_values, x)
y_bdf_solution = backward_differentiation_formula(y_initial_values_for_bdf)
# y_solution = odeint(f, [7.5, 0, 0, 0, 0, 0, 0], x)
print("Exact solution by scipy.odeint(): ", y_solution)
print("BDF solution : ", y_bdf_solution)
y1 = np.zeros_like(x)
for i in range(0, amount_of_points):
    y1[i]=y_solution[i][0]
ax.plot(x, y_solution, color='brown', label='Exact solution')
ax.plot(x, y_bdf_solution, color='green', label='BDF solution')
plt.grid()
plt.legend(loc='best');
plt.show()
print("Right-hand-parts: ", f)


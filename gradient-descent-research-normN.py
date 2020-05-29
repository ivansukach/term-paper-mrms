import numpy as np
import math
import sympy as sp
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt


def f(_x1, _x2, _x3, _x4):
    rhp = np.zeros([4])
    rhp[0] = _x1*_x2-_x3*_x4
    rhp[1] = _x1 * _x3 - _x2 * _x4
    rhp[2] = _x1 * _x4 - _x2 * _x3
    rhp[3] = _x1 * _x2 * _x3 - _x4
    return rhp


fig, ax = plt.subplots()
size_of_y = 4
sub = [2, 3, 7]
k = 3


def x_template(_k):
    s_template = ""
    _x1 = 0.1
    _x2 = 0.2
    _x3 = 0.3
    _x4 = 0.4
    for j in range(0, _k):
        _x1 += 0.3
        _x2 += 0.3
        _x3 += 0.3
        _x4 += 0.3
        s_template += "(alpha"+str(j)+"-"+str(sub[j])+")**2*np.asarray("+str(f(_x1, _x2, _x3, _x4).tolist())+")"
        if j != _k-1:
            s_template += '+'
    return s_template


x_template_str = x_template(3)
print('x_template for k = ', 3, ' : ', x_template_str)
r_template_str = x_template_str
for j in range(0, k):
    r_template_str = r_template_str.replace("alpha"+str(j),
                                            "alpha["+str(j)+"]")
print('x_template after assignment of values: ', x_template_str)
print('r_template after assignment of values: ', r_template_str)
for j in range(0, k):
    exec("alpha" + str(j) + " = sp.Symbol('alpha" + str(j) + "')")
minimizedFunction = eval("lambda alpha: "+r_template_str)


def z_grad(a):
    _x = eval(x_template_str)
    alphas = []
    for i in range(0, k):
        # Partial derivative
        x_prime = eval('sp.diff(_x, alpha'+str(i)+')')
        dif_alpha = str(x_prime).replace('alpha0', str(a[0]))
        for j in range(1, k):
            dif_alpha = dif_alpha.replace('alpha'+str(j), str(a[j]))

        alphas.append(norm(eval(dif_alpha)))
    return np.asarray(alphas)


def mininize(_a):
    # .x returns the solution of optimization minimization problem
    print("a:", _a)
    gradient = z_grad(_a)
    print("z_grad(a):", z_grad(_a))
    print("На вход минимизации поступает", _a, gradient)
    return _a - 0.01*gradient


def norm(_a):
    _sum = 0
    for i in range(0, size_of_y):
        _sum += _a[i]**2
    return math.sqrt(_sum)


def normK(_a):
    _sum = 0
    for i in range(0, k):
        _sum += _a[i]**2
    return math.sqrt(_sum)


def grad_step(dot):
    return mininize(dot)


dot = [np.array([1, 1, 1])]
dot.append(grad_step(dot[0]))
eps = 0.0001
attempt = [i for i in range(0, 20)]
norms_dif = []
while normK(dot[-2] - dot[-1]) > eps:
    norms_dif.append(normK(dot[-2] - dot[-1]))
    print("norm :", norms_dif[-1])
    dot.append(grad_step(dot[-1]))
    if len(norms_dif) == 20:
        break

print("points: ", dot)
print("amount of iterations: ", len(dot)-1)
print("minimum of minimizedFunc: ", minimizedFunction(dot[-1]))

import numpy as np
import math
import sympy as sp
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt


def jk(_y, _k):
    res = _y[0] * _y[_k] - math.exp(math.pow((_k + 1), 2 / 3) - math.pow(_k, 2 / 3)) * _y[_k + 1]
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


step = 0

def residual(_x):
    s = tau * f(_x, t[k + step - 1]) - c[k] * _x
    for i in range(1, p+1):
        s -= c[k - i] * np.asarray(y[k + step - i])
    return s


def normOfResidual(_residual):
    _sum = ""
    for i in range(0, size_of_y):
        _sum += "(" + str(_residual[i])+")**2"
        if i != len(_residual)-1:
            _sum += "+"
    return eval(_sum)


def x_template(_k):
    s_template = ""
    for j in range(0, _k):
        s_template += "beta"+str(j)+"*tau*f(y"+str(j+step)+", t"+str(j+step)+")-alpha"+str(j)+"*y"+str(j+step)+" "
        # s_template += "tau*f(y" + str(j + step) + ", t" + str(j + step) + ")-y" + str(j + step)
        if j != _k-1:
            s_template += '+'
    return s_template


def r_template(_k):
    s_template = ""
    for j in range(0, _k):
        s_template += "ab["+str(j+k)+"]*tau*f(y"+str(j+step)+", t"+str(j+step)+")-ab["+str(j)+"]*y"+str(j+step)
        if j != _k-1:
            s_template += '+'
    return s_template


def z_grad(a):
    for j in range(0, k):
        exec("alpha" + str(j)+" = sp.Symbol('alpha" + str(j) + "')")
        exec("beta" + str(j) + " = sp.Symbol('beta" + str(j) + "')")
        if j == k-1:
            r_x = eval('normOfResidual(residual('+x_template_str+'))')
    alphas = []
    betas = []
    for i in range(0, k):
        # Partial derivative
        prime = eval('sp.diff(r_x, alpha'+str(i)+')')
        dif_alpha = str(prime).replace('alpha0', str(a[0]))
        dif_alpha = dif_alpha.replace('beta0', str(a[k]))
        for j in range(1, k):
            dif_alpha = dif_alpha.replace('alpha'+str(j), str(a[j]))
            dif_alpha = dif_alpha.replace('beta'+str(j), str(a[k+j]))

        # Partial derivative
        prime = eval('sp.diff(r_x, beta'+str(i)+')')
        dif_beta = str(prime).replace('alpha0', str(a[0]))
        dif_beta = dif_beta.replace('beta0', str(a[k]))
        for j in range(1, k):
            dif_beta = dif_beta.replace('alpha'+str(j), str(a[j]))
            dif_beta = dif_beta.replace('beta'+str(j), str(a[k+j]))

        alphas.append(eval(dif_alpha))
        betas.append(eval(dif_beta))
    return np.asarray(alphas + betas)


def mininize(_a):
    gradient = z_grad(_a)
    _a = np.asarray(_a[0].tolist()+_a[1].tolist())
    l_min = minimize_scalar(lambda l: normOfR(_a - l * gradient)).x
    return _a - l_min*gradient


def norm(_a):
    _sum = 0
    for i in range(0, size_of_y):
        _sum += _a[i]**2
    return math.sqrt(_sum)


def grad_step(point):
    return mininize(point)


def V():
    _v = []
    for i in range(step, k+step):
        _v.append(-1*np.asarray(y[i]))
    for m in range(step, k+step):
        _v.append(tau * f(y[m], t[m]))
    return np.asarray(_v)


points = [np.array([1, 1, 1, 1, 1, 1])]
eps = 0.0001
all_norms_dif = []
substeps = []
while step < amount_of_points-4:
    x_template_str = x_template(k)
    print(x_template_str)
    r_template_str = r_template(k)
    print(r_template_str)
    for j in range(step, k + step):
        x_template_str = x_template_str.replace('tau*f(y' + str(j) + ', t' + str(j) + ")",
                                                'np.array(' + str((f(y[j], t[j]) * tau).tolist()) + ')')
        x_template_str = x_template_str.replace('y' + str(j), 'np.array(' + str(y[j]) + ')')

        r_template_str = r_template_str.replace('tau*f(y' + str(j) + ', t' + str(j) + ")",
                                                'np.array(' + str((f(y[j], t[j]) * tau).tolist()) + ')')
        r_template_str = r_template_str.replace('y' + str(j), 'np.array(' + str(y[j]) + ')')
    for j in range(0, k):
        exec("alpha" + str(j) + " = sp.Symbol('alpha" + str(j) + "')")
        exec("beta" + str(j) + " = sp.Symbol('beta" + str(j) + "')")
    normOfR = eval("lambda ab: normOfResidual(residual(" + r_template_str + "))")
    alphaAndBeta = [np.array([1, 1, 1, 1, 1, 1]), grad_step(np.array([1, 1, 1, 1, 1, 1]))]
    norms_dif = []
    v = V()
    norm_dif = norm(alphaAndBeta[-2].dot(v) - alphaAndBeta[-1].dot(v))
    while norm_dif > eps:
        norms_dif.append(norm_dif)
        print("norm :", norms_dif[-1])
        alphaAndBeta.append(grad_step(alphaAndBeta[-1]))
        norm_dif = norm(alphaAndBeta[-2].dot(v) - alphaAndBeta[-1].dot(v))
        if len(norms_dif) == 20:
            break
    points.append(alphaAndBeta[-1])
    y.append(points[-1].dot(v).tolist())
    all_norms_dif.append(norms_dif)
    substeps.append([i for i in range(0, len(norms_dif))])
    step += 1

print("points: ", points)
print("Y: ", y)
fig, ax = plt.subplots()
plt.title("NORM OF THE DIFFERENCE")
colors = ["red", "green", "blue", "yellow"]
for i in range(0, len(all_norms_dif)):
    ax.plot(substeps[i], all_norms_dif[i], color=colors[i])
plt.grid()
plt.show()

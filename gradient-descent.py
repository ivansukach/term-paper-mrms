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


fig, ax = plt.subplots()
amount_of_points = 101
size_of_y = 4
a = [-1/3, 3/2, -3, 11/6]
k = 3
p = 3
f = new_f(size_of_y)
y = [[7.5, 0, 0, 0],
 [1.52978522, 0.86092722, 0.73196761, 0.6841525],
 [1.52978522, 0.86092722, 0.73196761, 0.6841525],
 [1.52978522, 0.86092722, 0.73196761, 0.6841525]]
y_transpose = np.array([[7.5, 0, 0, 0],
 [1.52978522, 0.86092722, 0.73196761, 0.6841525],
 [1.52978522, 0.86092722, 0.73196761, 0.6841525],
 [1.52978522, 0.86092722, 0.73196761, 0.6841525]]).transpose().tolist()
# print("Y INITIAL VALUES: ", np.asarray(y))
t_min = 0
t_max = 10
t = np.linspace(t_min, t_max, amount_of_points)
h = (t_max - t_min)/(amount_of_points-1)


def x(_k, _h, _beta, _f, _alpha, _y, _t):
    s = 0
    for j in range(0, _k):
        # print("j = ", j)
        s += _h * _beta[j] * _f(_y[j], _t[j]) - _alpha[j] * _y[j]
    return s


def residual(_x):
    s = h*f(_x, t[k-1])-a[k]*_x
    for i in range(1, p+1):
        s -= a[k-i] * np.asarray(y[k-i])
        # print("s=", s)
        # print("y=", y[k-i])
        # s -= 3 * np.asarray(y[k-i])
    return s


def x_template(_k):
    s_template = ""
    for j in range(0, _k):
        # print("j = ", j)
        s_template += "beta"+str(j)+"*h*f(y"+str(j)+", t"+str(j)+")-alpha"+str(j)+"*y"+str(j)+""
        if j != _k-1:
            s_template += '+'
    return s_template


def r_template(_k):
    s_template = ""
    for j in range(0, _k):
        # print("j = ", j)
        s_template += "ab["+str(j+k)+"]*h*f(y"+str(j)+", t"+str(j)+")-ab["+str(j)+"]*y"+str(j)+""
        if j != _k-1:
            s_template += '+'
    return s_template


x_template_str = x_template(3)
r_template_str = r_template(3)
# print('x_template for k = ', 3, ' : ', x_template_str)
# print('r_template for k = ', 3, ' : ', r_template_str)
for j in range(0, k):
    # print('y' + str(j) + ' =', y[j])
    # print('t' + str(j) + ' =', t[j])
    x_template_str = x_template_str.replace('h*f(y' + str(j) + ', t' + str(j)+")",
                                            'np.array('+str((f(y[j], t[j])*h).tolist())+')')
    x_template_str = x_template_str.replace('y'+str(j), 'np.array('+str(y[j])+')')

    r_template_str = r_template_str.replace('h*f(y' + str(j) + ', t' + str(j)+")",
                                            'np.array('+str((f(y[j], t[j])*h).tolist())+')')
    r_template_str = r_template_str.replace('y'+str(j), 'np.array('+str(y[j])+')')
# print('x_template after assignment of values: ', x_template_str)
# print('r_template after assignment of values: ', r_template_str)
for j in range(0, k):
    exec("alpha" + str(j) + " = sp.Symbol('alpha" + str(j) + "')")
    exec("beta" + str(j) + " = sp.Symbol('beta" + str(j) + "')")
r = eval("lambda ab: residual("+r_template_str+")")

# GRAD(z(x,y) in point (a[0], a[1]))
def z_grad(a):
    _x = eval('2+2')
    for j in range(0, k):
        exec("alpha" + str(j)+" = sp.Symbol('alpha" + str(j) + "')")
        exec("beta" + str(j) + " = sp.Symbol('beta" + str(j) + "')")
        if j == k-1:
            _x = eval('residual('+x_template_str+')')
    # Partial derivative
    # One type of notation for derivatives is sometimes called prime notation. The function f´(x),
    # which would be read ``f-prime of x'',
    # means the derivative of f(x) with respect to x. If we say y = f(x), then y´ (read ``y-prime'') = f´(x).
    alphas = []
    betas = []
    for i in range(0, k):
        # Partial derivative
        x_prime = eval('sp.diff(_x, alpha'+str(i)+')')
        dif_alpha = str(x_prime).replace('alpha0', str(a[0]))
        dif_alpha = dif_alpha.replace('beta0', str(a[k]))
        for j in range(1, k):
            dif_alpha = dif_alpha.replace('alpha'+str(j), str(a[j]))
            dif_alpha = dif_alpha.replace('beta'+str(j), str(a[k+j]))

        # Partial derivative
        x_prime = eval('sp.diff(_x, beta'+str(i)+')')
        dif_beta = str(x_prime).replace('alpha0', str(a[0]))
        dif_beta = dif_beta.replace('beta0', str(a[k]))
        for j in range(1, k):
            dif_beta = dif_beta.replace('alpha'+str(j), str(a[j]))
            dif_beta = dif_beta.replace('beta'+str(j), str(a[k+j]))

        alphas.append(eval(dif_alpha))
        betas.append(eval(dif_beta))
    return np.asarray(alphas + betas)


def mininize(_a):
    # .x returns the solution of optimization minimization problem
    # print("a:", _a)
    gradient = z_grad(_a)
    # print("z_grad(a):", z_grad(_a))
    _a = _a[0].tolist()+_a[1].tolist()
    _a = np.asarray(_a)
    # print("a after transformation:", _a)
    gradient = gradient.transpose()
    # print("gradient after transformation and transpose:", gradient)
    lambda_vector = []
    # print("На вход r поступают начальные приближения alpha и beta:", _a)
    for i in range(0, size_of_y):
        # print("На вход r поступает градиент №", i, " :", gradient[i])
        l_min = minimize_scalar(lambda l: norm(r(_a - l * gradient[i]))).x
        lambda_vector.append(l_min)
    # print("lambda_vector:", lambda_vector)
    # print("gradient:", gradient)
    # print("lambda_vector transposed:", np.asarray(lambda_vector))
    return _a - np.asarray(lambda_vector).dot(gradient)

def norm(_a):
    _sum = 0
    for i in range(0, size_of_y):
        _sum += _a[i]**2
    return math.sqrt(_sum)


def norm2k(_a):
    _sum = 0
    for i in range(0, 2*k):
        _sum += _a[i]**2
    return math.sqrt(_sum)


def grad_step(dot):
    return mininize(dot)

dot = [np.array([7, 0, 3, 7, 0, 9])]
dot.append(grad_step(dot[0]))
eps = 0.0001
attempt = [i for i in range(0, 20)]
norms_dif = []
while norm2k(dot[-2] - dot[-1]) > eps:
    norms_dif.append(norm2k(dot[-2] - dot[-1]))
    print("norm :", norms_dif[-1])
    dot.append(grad_step(dot[-1]))
    if len(norms_dif) == 20:
        break

print("points: ", dot)


def V():
    _v=[]
    for i in range(0, k):
        _v.append(-1*np.asarray(y[i]))
    for m in range(0, k):
        _v.append(h * f(y[m], t[m]))
    return np.asarray(_v)


v = V()
alpha_and_beta = dot[-1]
print("matrix V: ", v)
print("alpha and beta: ", alpha_and_beta)
print("yk: ", alpha_and_beta.dot(v))
ax.plot(attempt, norms_dif, color='red', label='Norm of the difference')
plt.grid()
plt.legend(loc='best');
plt.show()

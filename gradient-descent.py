import numpy
import math
from pylab import *
from sympy import *
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d, Axes3D


def jk(_y, _k):
    res = _y[0] * _y[_k] - math.exp(math.pow((_k + 1), 2 / 3) - math.pow(_k, 2 / 3)) * _y[_k + 1]
    return res


def new_f(n):
    def f(_y, _x):
        print('---------------')
        print('_y in rhp: ', _y)
        print('_x in rhp: ', _x)
        print('---------------')
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
size_of_y = 7
k = 3
f = new_f(size_of_y)
y = np.zeros([size_of_y])
y[0] = 7.5
t_min = 0
t_max = 10
t = np.linspace(t_min, t_max, amount_of_points)
h = (t_max - t_min)/(amount_of_points-1)


# Function to be minimized
# z_str = '3 * x ** 2 + y ** 2 - x * y - 4 * x'

# Formula from string
# exec('z = lambda a: ' + z_str)


def x(_k, _h, _beta, _f, _alpha, _y, _t):
    s = 0
    for j in range(0, _k):
        print("j = ", j)
        s += _h * _beta[j] * _f(_y[j], _t[j]) - _alpha[j] * _y[j]
    return s


def x_template(_k):
    s_template = ""
    for j in range(0, _k):
        print("j = ", j)
        s_template += "h*beta["+str(j)+"]*f(y["+str(j)+"], t["+str(j)+"])-alpha["+str(j)+"]*y["+str(j)+"]"
        if j != _k-1:
            s_template += '+'
    return s_template


x_template_str = x_template(3)
print('x_template for k = ', 3, ' : ', x_template_str)
# z_str = z_str.replace('a[0]', 'x')
# z_str = z_str.replace('a[1]', 'y')


def get_symbols(_k):
    _alpha = [Symbol('alpha[0]')]
    _beta = [Symbol('beta[0]')]
    for j in range(0, _k):
        _alpha.append(Symbol('alpha['+str(j)+']'))
        _beta.append(Symbol('beta[' + str(j) + ']'))
    return _alpha, _beta


# GRAD(z(x,y) in point (a[0], a[1]))
def z_grad(a):
    _alpha, _beta = get_symbols(k)

    _x = eval(x_template_str)
    # exec('_x =  ' + z_str)

    # Partial derivative
    # One type of notation for derivatives is sometimes called prime notation. The function f´(x),
    # which would be read ``f-prime of x'',
    # means the derivative of f(x) with respect to x. If we say y = f(x), then y´ (read ``y-prime'') = f´(x).
    alphas = []
    betas = []
    for i in range(0, k):
        # Partial derivative
        x_prime = _x.diff(_alpha[i])
        dif_alpha = str(x_prime).replace('alpha[0]', str(a[0][0]))
        dif_alpha = dif_alpha.replace('beta[0]', str(a[1][0]))
        for j in range(1, k):
            dif_alpha = dif_alpha.replace('alpha['+str(j)+']', str(a[0][j]))
            dif_alpha = dif_alpha.replace('beta['+str(j)+']', str(a[1][j]))

        # Partial derivative
        x_prime = _x.diff(_beta[i])
        dif_beta = str(x_prime).replace('alpha[0]', str(a[0][0]))
        dif_beta = dif_beta.replace('beta[0]', str(a[1][0]))
        for j in range(1, k):
            dif_beta = dif_beta.replace('alpha['+str(j)+']', str(a[0][j]))
            dif_beta = dif_beta.replace('beta['+str(j)+']', str(a[1][j]))
        alphas.append(eval(dif_alpha))
        betas.append(eval(dif_beta))
    return numpy.array([alphas, betas])


def mininize(a):
    # .x returns the solution of optimization minimization problem
    l_min = minimize_scalar(lambda l: x(a - l * z_grad(a))).x
    return a - l_min * z_grad(a)


def norm(a):
    return math.sqrt(a[0] ** 2 + a[1] ** 2)


def grad_step(dot):
    return mininize(dot)


dot = [numpy.array([[0, 0, 0], [0, 0, 0]])]
dot.append(grad_step(dot[0]))
eps = 0.0001

while norm(dot[-2] - dot[-1]) > eps:
    dot.append(grad_step(dot[-1]))


# def makeData ():
#     x = numpy.arange(-200, 200, 1.0)
#     y = numpy.arange(-200, 200, 1.0)
#     # GRID production from two arrays
#     xgrid, ygrid = numpy.meshgrid(x, y)
#     zgrid = z([xgrid, ygrid])
#     return xgrid, ygrid, zgrid


# xt, yt, zt = makeData()
#
# # Create a new figure.
# fig = plt.figure()
# # The Axes contains most of the figure elements: Axis, Tick, Line2D, Text, Polygon, etc., and sets the coordinate system.
# ax = plt.axes(projection='3d')
# # Create a surface plot.
# ax.plot_surface(xt, yt, zt, cmap=cm.hot)
# ax.plot([x[0] for x in dot], [x[1] for x in dot], [z(x) for x in dot], color='b')
#
# plt.show()
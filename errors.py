import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.integrate import *
from scipy.optimize import minimize_scalar

fig, ax = plt.subplots()
taus = [math.log10(0.1), math.log10(0.02), math.log10(0.01), math.log10(0.005)]
error33 = [math.log10(196.83087455351733), math.log10(2508.1796543523683), math.log10(0.00013002697405783652), math.log10(1.9102390022275413e-05)]
error23 = [math.log10(484.72591233160813), math.log10(5221.959498010563), math.log10(0.001027442022570851), math.log10(0.00016157311703248985)]
error22 = [math.log10(719.9455345150432), math.log10(5853.526264666207), math.log10(0.001968557294813569), math.log10(0.00019990911850046694)]
ax.plot(taus, error33, color='brown', label='3-step bdf, 3-step multistep')
ax.plot(taus, error23, color='green', label='2-step bdf, 3-step multistep')
ax.plot(taus, error22, color='red', label='2-step bdf, 2-step multistep')
plt.grid()
plt.legend(loc='best')
plt.show()
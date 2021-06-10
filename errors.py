import matplotlib.pyplot as plt

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()
fig4, ax4 = plt.subplots()
taus = [0.1, 0.02, 0.01, 0.005]
beckdo_max_error33 = [222.05463056556758, 2557.9311447972746, 0.00014054789044309335, 3.919083888146957e-06]
beckdo_max_error23 = [439.59101542589264, 4776.722545421239, 0.0008549418257235696, 0.00010499480026354137]
beckdo_max_error22 = [525.6183541604481, 6064.540205488747, 0.00184022190841921, 0.0001588197613855774]
beckdo_end_of_interval_error33 = [33.48424430939275, 10.841699156604479, 1.8349310299982405e-07, 4.478925153672108e-09]
beckdo_end_of_interval_error23 = [439.59101542589264, 19.797138600656044, 1.1157534925270847e-06, 1.1589952276151221e-07]
beckdo_end_of_interval_error22 = [171.20731775903752, 6.726047306500091, 2.947316130139182e-06, 2.3983962832341294e-07]
hires_end_of_interval_error33 = [9.691693855010962e-07, 6.400396381524192e-11, 9.20791167274657e-13, 1.0333697496932347e-12]
hires_end_of_interval_error23 = [2.7082540924248196e-05, 5.030149935543246e-08, 3.231650462924517e-09, 2.211306622247181e-10]
hires_end_of_interval_error22 = [0.0002018191939132248, 3.1304188557864776e-07, 2.2407670960116295e-08, 1.4061985384943705e-09]
hires_max_error33 = [3.939278967815713e-05, 1.9435782491528745e-08, 3.7203267926411356e-10, 7.0415017392284945e-12]
hires_max_error23 = [3.096648602526995e-05, 7.611825123039073e-07, 6.753824384069603e-08, 4.9975353349760685e-09]
hires_max_error22 = [0.0002018191939132248, 1.118622734407088e-06, 8.226136291846607e-08, 5.546935125982865e-09]
ax1.semilogy(taus, hires_max_error33, color='brown', label='3-step bdf, 3-step multistep')
ax1.semilogy(taus, hires_max_error23, color='green', label='2-step bdf, 3-step multistep')
ax1.semilogy(taus, hires_max_error22, color='red', label='2-step bdf, 2-step multistep')
ax1.set_xlabel('time step (s)')
ax1.set_ylabel('max error')
ax2.semilogy(taus, beckdo_max_error33, color='brown', label='3-step bdf, 3-step multistep')
ax2.semilogy(taus, beckdo_max_error23, color='green', label='2-step bdf, 3-step multistep')
ax2.semilogy(taus, beckdo_max_error22, color='red', label='2-step bdf, 2-step multistep')
ax2.set_xlabel('time step (s)')
ax2.set_ylabel('max error')
ax3.semilogy(taus, hires_end_of_interval_error33, color='brown', label='3-step bdf, 3-step multistep')
ax3.semilogy(taus, hires_end_of_interval_error23, color='green', label='2-step bdf, 3-step multistep')
ax3.semilogy(taus, hires_end_of_interval_error22, color='red', label='2-step bdf, 2-step multistep')
ax3.set_xlabel('time step (s)')
ax3.set_ylabel('error at the end of interval')
ax4.semilogy(taus, beckdo_end_of_interval_error33, color='brown', label='3-step bdf, 3-step multistep')
ax4.semilogy(taus, beckdo_end_of_interval_error23, color='green', label='2-step bdf, 3-step multistep')
ax4.semilogy(taus, beckdo_end_of_interval_error22, color='red', label='2-step bdf, 2-step multistep')
ax4.set_xlabel('time step (s)')
ax4.set_ylabel('error at the end of interval')
ax1.grid()
ax2.grid()
ax3.grid()
ax4.grid()
plt.legend(loc='best')
plt.show()

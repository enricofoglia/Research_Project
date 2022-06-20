#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 19:18:08 2022

@author: enricofoglia
"""

import numpy as np
import matplotlib.pyplot as plt
import control  # https://python-control.readthedocs.io/en/0.9.1/
from Theodorsen_control import *

a = 1/2  # pitch axis wrt to 1/2-chord
b = 1  # half-chord length of the airfoil
airfoil = AirfoilGeometry(a=a, b=b)  # default values of C_1 and C_2 used

# the balanced truncation Theodorsen function approximation
theodorsen_function_sys = theodorsen_function_balanced_truncation_ss()

theodorsen_full_sys = unsteady_lift_ss(airfoil, theodorsen_function_sys)

t = np.linspace(0, 50, 1000)
amp_scale = 0.2
N = 1

alpha_vec, IC_alpha = sinusoidalInputs(t, amp_scale, N, second_derivative=True)
h_vec, IC_h = sinusoidalInputs(t, amp_scale, N, second_derivative=False)

u_MISO = np.vstack((h_vec[-1], alpha_vec[-1]))
X0 = np.array([0,0,0,0, IC_h[0], IC_alpha[0], IC_alpha[1]])
output = control.forced_response(
    theodorsen_full_sys, 
    T=t, 
    U=u_MISO,
    X0 = X0
    )
data_both = TheodorsenTimeResponse(output, inputs='both')
data_both.phase_plot(state='alpha_e')
data_both.io_plot()
#data_both.state_plot()
#data_both.theodorsen_state_plot()



fig, ax = plt.subplots(3,1)
ax[0].plot(t, alpha_vec[0])
ax[1].plot(t, alpha_vec[1])
ax[2].plot(t, alpha_vec[2])
ax[2].set_xlabel('time')
ax[0].set_ylabel(r'$\alpha$') 
ax[1].set_ylabel(r'$\dot\alpha$') 
ax[2].set_ylabel(r'$\ddot\alpha$') 

plt.show()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 12 18:44:32 2022

@author: enricofoglia
"""

import numpy as np
import pysindy as ps
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp as integral
from Theodorsen_control import *

def u_fun(t):
    amplitude_alpha, amplitude_h, omega, phi = np.loadtxt('theodorsen_params.dat', dtype = float)
    b = 1
    U_inf = 1
    u = np.zeros((1,5))
    alpha_dotdot = amplitude_alpha*np.exp(1j*omega*b/U_inf*t) # angular acceleration [-]
    alpha_dot = 1/(1j*omega*b/U_inf)*alpha_dotdot # angular velocity [-]
    alpha = -1/(omega*b/U_inf)**2*alpha_dotdot # angle of attack [-]
    
    h_dotdot = amplitude_h*np.exp(1j*(omega*b/U_inf*t + phi)) # vertical acceleration, normalized with the semichord [-]
    h_dot = 1/(1j*omega*b/U_inf)*h_dotdot # vertical velocity [-]
    u = np.real(np.array([alpha_dotdot,
                          alpha_dot,
                          alpha,
                          h_dotdot,
                          h_dot]))
    return  u
    
a = 1/2  # pitch axis wrt to 1/2-chord
b = 1  # half-chord length of the airfoil
airfoil = AirfoilGeometry(a=a, b=b)  # default values of C_1 and C_2 used

# the balanced truncation Theodorsen function approximation
theodorsen_function_sys = theodorsen_function_balanced_truncation_ss()

# generation of unsteady lift state-space systems

# state-space system with both α" and h" as inputs
theodorsen_full_sys = unsteady_lift_ss(airfoil, theodorsen_function_sys)
# state-space system with both α" as input
theodorsen_alpha_sys = unsteady_lift_ss(
    airfoil, theodorsen_function_sys, inputs='alpha')
# state-space system with both h" as input
theodorsen_h_sys = unsteady_lift_ss(
    airfoil, theodorsen_function_sys, inputs='h')

t = np.linspace(0, 100, int(1e05))


omega = 1
amplitude = 0.1
phase = np.pi/2
u_alpha = amplitude*np.sin(omega*t + phase) + 0.8*amplitude*np.sin(3*omega*t + phase) + 1.2*amplitude*np.sin(4*omega*t + phase)


T = 5
phase = T/2
amplitude = 0.1/T
# square wave
u_h = np.array([amplitude if np.floor((ti+phase)/T+T/2) %
               2 == 0 else -amplitude for ti in t])
output = control.forced_response(
    theodorsen_full_sys, T=t, U=np.vstack((u_h, u_alpha)))
data_both = TheodorsenTimeResponse(output, inputs='alpha')
data_both.phase_plot(state='alpha_e')

# TODO: the pdf has to be done on the input, not the output.

nb_bins = 1000
fig, ax = plt.subplots()
ax.hist(data_both.C_L, nb_bins, density = True)
ax.set_xlabel(r'$C_L$')
ax.set_ylabel(r'$f(C_L)$')
ax.set_title(r'$C_L$ p.d.f')

plt.show()





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
    

theodorsen_data = np.loadtxt('theodorsen_data.dat', dtype=float)

t = theodorsen_data[:,0]
dt = t[1] - t[0]
X = theodorsen_data[:,1]
X = np.array([X]).T
U = theodorsen_data[:,2:]
U = np.reshape(U, (len(t),5))


fig1, ax1 = plt.subplots()

for deg in range(2,3):
    optimizer = ps.optimizers.stlsq.STLSQ(threshold = 0.1, alpha = 1e-05, max_iter = 100)
    library = ps.feature_library.polynomial_library.PolynomialLibrary(degree = deg)
    model = ps.SINDy(optimizer = optimizer, 
   				     feature_library = library,
   				     feature_names = ['CL', '(a)\"', '(a)\'', 'a', '(h)\"', '(h)\'']) # default paramaters:
                                           					   						   # differentiation method: centered differences
   
    model.fit(X, u = U, t = dt)
    model.print()
    X0 = np.array([X[0]])[0]
    model_x = model.simulate(X0, t, u = u_fun)
    
    end_plot =int( 5 / dt)
    
    ax1.plot(t[0:end_plot], model_x[0:end_plot])
    ax1.plot(t[0:end_plot], X[0:end_plot], linestyle = '--', color = 'k')
    ax1.set_xlabel('time tb/U')
    ax1.set_ylabel(r'$C_L$')
    ax1.set_title(r'Reconstructed $C_L$')
    ax1.grid(True)
    ax1.legend(['SINDy model', 'Original data'])
plt.show() 
   
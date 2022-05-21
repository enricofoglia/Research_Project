#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 13 12:40:42 2022

@author: enricofoglia
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv, yn # Bessel functions

def hankel2(n,k):
    return jv(n,k) - 1j*yn(n,k)

def Theodorsen(k):
    return hankel2(1,k)/(hankel2(1,k) + hankel2(0,k)*1j)

def u_fun(amplitude_alpha, amplitude_h, omega, t, b = 1, U_inf = 1, phi = 0.1):
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
    return u.T

def C_L_sinus(amplitude_alpha, amplitude_h, omega, x_pitch, t, b = 1, U_inf = 1, C1 = np.pi, C2 = 2*np.pi, phi = 0.1):
    # all distances are normalized with respect to the semichord b
    # all times are normalized with respect to the convection time b/U_inf
    # it is, as a first approximation, assumed that alpha(0) = alpha_dot(0) = h_dot(0) = 0
    # TODO: implement initial conditions
    
    a = 2*x_pitch-1 # # pitch axis with respect to 1/2 chord, normalized with the semichord [-]
    
    alpha_dotdot = amplitude_alpha*np.exp(1j*omega*b/U_inf*t) # angular acceleration [-]
    alpha_dot = 1/(1j*omega*b/U_inf)*alpha_dotdot # angular velocity [-]
    alpha = -1/(omega*b/U_inf)**2*alpha_dotdot # angle of attack [-]
    
    h_dotdot = amplitude_h*np.exp(1j*(omega*b/U_inf*t + phi)) # vertical acceleration, normalized with the semichord [-]
    h_dot = 1/(1j*omega*b/U_inf)*h_dotdot # vertical velocity [-]
    
    return np.real(C1*(h_dotdot + alpha_dot - a*alpha_dotdot) + C2*(alpha + h_dot + alpha_dot*(0.5 - a))*Theodorsen(omega*b/U_inf))

if __name__ == '__main__':
    
    PLOT_FLAG = input('Do you want to plot the results? [y/n]: ')
    while not (PLOT_FLAG in 'yn'):
        print('Invalid input.\n')
        PLOT_FLAG = input('Do you want to plot the results? [y/n]: ')

    SAVE_FLAG = input('Do you want to save the data to a file? [y/n]: ')
    while not(SAVE_FLAG in 'yn'):
        print('Invalid input.\n')
        SAVE_FLAG = input('Do you want to plot the results? [y/n]: ')

    
    omega = np.pi 
    f = omega / (2*np.pi) # angular frequency = 0.5 Hz
    dt = 1 / f / 200 # 100 data points every half period
    t_range = np.arange(0, 100, dt)
    
    amplitude_alpha = 0.1 # amplitude in angular oscillations in radians
    amplitude_h = 0.1 # amplitude in vertical displacement normalised with semichord
    x_pitch = 0.5 # position of oscillation centre
    phi = 0.5
    
    CL = C_L_sinus(amplitude_alpha, amplitude_h, omega, x_pitch, t_range, phi = phi)
    u = u_fun(amplitude_alpha, amplitude_h, omega, t_range, phi = phi)
    
    # -------------------------------------------------------------------------
    #                              SAVING TO FILE
    # -------------------------------------------------------------------------
    if SAVE_FLAG == 'y':
        theodorsen_data = np.column_stack((t_range, CL, u))
        np.savetxt('theodorsen_data.dat', theodorsen_data) 
        parameters = np.array([amplitude_alpha, amplitude_h, omega, phi])
        np.savetxt('theodorsen_params.dat', parameters)

    # -------------------------------------------------------------------------
    #                                 GRAPHICS
    # -------------------------------------------------------------------------
    if PLOT_FLAG == 'y':
        T = 1/f
        end_plot =int( 5 * T / dt)
        
        fig, ax = plt.subplots()
        ax.plot(t_range[0:end_plot+1], CL[0:end_plot+1])
        ax.plot(t_range[0:end_plot+1], u[0:end_plot+1,0], color = 'orangered')
        ax.plot(t_range[0:end_plot+1], u[0:end_plot+1,3], color = 'tab:orange')
        ax.set_xlabel('time tb/U ')
        ax.set_ylabel(r'$C_L$')
        ax.set_title(r'Unsteady Lift coefficient $C_L$')
        ax.legend([r'$C_L$', r'$\ddot{\alpha}$', r'$\ddot{h}$'])
        ax.grid(True)
        
        
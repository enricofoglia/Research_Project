#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 08:30:47 2022

@author: Enrico Foglia

Compute the exact Wagner function phi(t) = L(t)/L0.

The functions come from the paper S. Dawson and S. Brunton "Improved approximations to the Wagner 
function using sparse identification of nonlinear dynamics", 2021.

In particular the equation shown in the article to be the best behaved is performing terribly 
(large_times = False), while the one that should work for large t is well behaved for small t but 
fails somewhat for larger t. It is probably do to the integration approach (I have tried quad, that 
is the the best performing and allows for infinite domain, and quadrature).

webplotdigitalizer

"""
import numpy as np
from scipy.special import jv, yn # Bessel functions
from scipy.integrate import quad

def hankel2(n,k):
    return jv(n,k) - 1j*yn(n,k)

def Wagner(t, large_times = False): 
    if large_times == False:
        G= lambda k: (jv(1,k)*(jv(1,k)+yn(0,k))+yn(1,k)*(yn(1,k)-jv(0,k)))/((jv(1,k) + yn(0,k))**2 + (yn(1,k) - jv(0,k))**2)
        phi = 1/2 + 2/np.pi*quad(lambda k: 1/k*(G(k)-1/2)*np.sin(k*t), 0, 100, limit = int(100*t)+50)[0]
    else:    
        G= lambda k: (yn(1,k)*yn(0,k)+jv(1,k)*jv(0,k))/((jv(1,k) + yn(0,k))**2 + (yn(1,k) - jv(0,k))**2)
        phi = 1 - 2/np.pi*quad(lambda k: 1/k*G(k)*np.cos(k*t), 0, 100, limit = int(100*t))[0]
    return phi

def Theodorsen(k):
    return hankel2(1,k)/(hankel2(1,k) + hankel2(0,k)*1j)

def C_L_sinus(amplitude_alpha, amplitude_h, omega, x_pitch, t, b = 1, U_inf = 1, C1 = np.pi, C2 = 2*np.pi):
    # all distances are normalized with respect to the semichord b
    # all times are normalized with respect to the convection time b/U_inf
    # it is, as a first approximation, assumed that alpha(0) = alpha_dot(0) = h_dot(0) = 0
    # TODO: implement initial conditions
    
    a = 2*x_pitch-1 # # pitch axis with respect to 1/2 chord, normalized with the semichord [-]
    
    alpha_dotdot = amplitude_alpha*np.exp(1j*omega*b/U_inf*t) # angular acceleration [-]
    alpha_dot = 1/(1j*omega*b/U_inf)*alpha_dotdot # angular velocity [-]
    alpha = -1/(omega*b/U_inf)**2*alpha_dotdot # angle of attack [-]
    
    phi = 0.1 # phase shift velocity / angle of attack
    h_dotdot = amplitude_h*np.exp(1j*(omega*b/U_inf*t + phi)) # vertical acceleration, normalized with the semichord [-]
    h_dot = 1/(1j*omega*b/U_inf)*h_dotdot # vertical velocity [-]
    
    return np.real(C1*(h_dotdot + alpha_dot - a*alpha_dotdot) + C2*(alpha + h_dot + alpha_dot*(0.5 - a))*Theodorsen(omega*b/U_inf))

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    PLOT_FLAG = input('Do you want to plot the results? [y/n]: ')
    if not (PLOT_FLAG in 'yn'):
        print('Invalid input.\n')
        PLOT_FLAG = input('Do you want to plot the results? [y/n]: ')

    SAVE_FLAG = input('Do you want to save the data to a file? [y/n]: ')
    if not(SAVE_FLAG in 'yn'):
        print('Invalid input.\n')
        SAVE_FLAG = input('Do you want to plot the results? [y/n]: ')

    amplitude_alpha = 0.1
    amplitude_h = 0.1
    omega = 1e-1*2*np.pi # frequency of the exitation = 0.1 Hz
    x_pitch = 0.6 # position of the axis of rotation
    

    tt = np.linspace(0,100, 1000)
    b = 1
    U_inf = 1
    h_dotdot = np.real(amplitude_h*np.exp(1j*omega*b/U_inf*tt)) # vertical acceleration, normalized with the semichord [-]
    h_dot = 1e3*np.real(1/(1j*omega*b/U_inf)*h_dotdot) # vertical velocity [-]
    
    # data for plotting Theodorsen function
    CL = C_L_sinus(amplitude_alpha, amplitude_h, omega, x_pitch, tt)
    k = np.logspace(-3,2,1000)

    # data to plot Wagner function
    if PLOT_FLAG == 'y':
        T = np.logspace(-1,4, num = 100)
        W = np.zeros(len(T))
        for i in range(len(T)):
            t = T[i]
            if t<=100:
                W[i] = Wagner(t, large_times = False)
            else:
                W[i] = Wagner(t, large_times = True)   
        wagner_data = np.column_stack((T,W))
        np.savetxt('wagner_data_loglog.dat', wagner_data) 
    
    # generate data for wagner_SINDy.py
    if SAVE_FLAG == 'y':
        T = np.linspace(0,500, 2000) # make sure to have a 
        W = np.zeros(len(T))
        for i in range(len(T)):
            t = T[i]
            if t<=100:
                W[i] = Wagner(t, large_times = False)
            else:
                if i%100 == 0:
                    print(f'Passed {i} time steps\n')
                W[i] = Wagner(t, large_times = False)   

        wagner_data = np.column_stack((T,W))
        np.savetxt('wagner_data.dat', wagner_data) #19/04/2022: saving to data file to be used in SINDy module 
    
    # Plotting
    if PLOT_FLAG == 'y':
        fig, ax = plt.subplots(figsize=(10,4), tight_layout = True)
        ax.plot(tt,CL)
        ax.plot(tt, h_dotdot)
        ax.set_xlabel(r'$\tau = tb/U_{\infty}$')
        ax.set_ylabel(r'$C_L(\tau)$')
        ax.set_title(f'$C_L$ for a sinusoidal velocity imput with reduced frequency f = {omega/2/np.pi}')
        ax.grid(True)
        ax.legend(['$C_L$', '$\ddot{h}$'])
        
        fig2, ax2 = plt.subplots()
        ax2.plot(k, np.real(Theodorsen(k)))
        ax2.plot(k, np.imag(Theodorsen(k)))
        ax2.set_xlabel(r'$k = \omega b/U_{\infty}$')
        ax2.set_xscale('log')
        ax2.set_ylabel('C(k)')
        ax2.set_title('Real and imaginary part of the Theodorsen function C(k)')
        ax2.legend([r'$\mathfrak{R}[C(k)]$',r'$\mathfrak{I}[C(k)]$'])
        ax2.grid(True)
        
        # plt.show()

        fig, ax = plt.subplots()
        ax.plot(T,np.ones(len(W))-W)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.grid(True)
        ax.set_xlabel('time')
        ax.set_ylabel('$1-\phi (t)$')
        ax.set_title('Wagner function')
        plt.show()
    
    
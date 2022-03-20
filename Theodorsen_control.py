"""
Created on Wed Mar 16 22:49:48 2022

@author: Maciej Michałów

This script demonstrates a functional approximated implementation of the original
Theodorsen model, using the Python control systems library and the Python Control
Systems Library(https://python-control.readthedocs.io/en/0.9.1/).

The system is modelled using a transfer function approach; all equations have been
adapted from S. L. Brunton and C. W. Rowley, ‘Empirical state-space representations
for Theodorsen’s lift model’, Journal of Fluids and Structures, vol. 38, pp. 174–186,
april 2013, doi: 10.1016/j.jfluidstructs.2012.10.005.

TODO: This script should be converted to a module with functions for generating data
based on given parameters.

"""

import numpy as np
import matplotlib.pyplot as plt
import control  # https://python-control.readthedocs.io/en/0.9.1/

# geometric parameters (TODO check the relationships and dimension of all of them)
c = 1  # chord length of the airfoil [m]
b = c/2  # half-chord length of airfoil [m]
x_pitch = 0.3  # pitch axis wrt
a = x_pitch-b  # pitch axis with respect to 1/2 chord [m?]

# aerodynamic parameters
nu = 2e-5
C_1 = np.pi  # [1/rad]
C_2 = 2*np.pi  # [1/rad]
U_inf = 1  # [m/s]
Re = c*U_inf/nu  # Reynolds number (just informative)

# Theodorsen transfer function (balanced truncation approximation)
C_Theodorsen = control.tf(
    [0.5, 0.703, 0.2393, 0.01894, 2.318e-4], [1, 1.158, 0.3052, 0.02028, 2.325e-4])

# Auxiliary tf for ease of writing
s = control.tf('s')

# Transfer function of C_L wrt alpha"
C_L_alpha_bis = C_1*(1/s - a) + C_2*(1/s**2+1/s*(1/2-a)) * \
    C_Theodorsen

# Transfer function of C_L wrt h"
C_L_h_bis = C_1 + C_2/s*C_Theodorsen

# Bode plot of the Theodorsen transfer function
mag, phase, omega = control.bode_plot(
    C_Theodorsen, dB=True, omega_limits=[5e-3, 1e2])
plt.suptitle('$C(\overline{s})$')
plt.show()

# Bode plot of the α" transfer function
mag, phase, omega = control.bode_plot(
    C_L_alpha_bis, dB=True, omega_limits=[5e-3, 1e3])
plt.suptitle(
    'Transfer function of $C_L$ wrt $\ddot{\\alpha}}$ (variable $\overline{s}$); ' + '$x/c = {:.1f}$'.format(x_pitch))
plt.show()

# Bode plot of the h" transfer function
mag, phase, omega = control.bode_plot(
    C_L_h_bis, dB=True, omega_limits=[5e-3, 1e2])
plt.suptitle(
    'Transfer function of $C_L$ wrt $\ddot{h}$ (variable $\overline{s}$)')
plt.show()

# sine response - pitching
amplitude = 0.01  # [deg/s^2]
frequency = 1e2  # [Hz]
t_series = np.linspace(0, 100, 1000)  # [s]
u_series = amplitude*np.pi/180.*np.sin(2*np.pi*frequency*t_series)
response = control.forced_response(C_L_alpha_bis, t_series, u_series)

plt.plot(response.time, response.outputs)
plt.title(
    'Response to a {:.1e}'.format(frequency) + '[Hz] sine $\\ddot{\\alpha}$ input of amplitude ' + '{:.1e} $[deg/s^2]$'.format(amplitude))
plt.xlabel('t [s]')
plt.ylabel('$C_L$ $[-]$')
plt.show()

# sine response - plunging
amplitude = 0.01  # [m/s^2]
frequency = 1e2  # [Hz]
t_series = np.linspace(0, 100, 1000)  # [s]
u_series = amplitude*np.sin(2*np.pi*frequency*t_series)
response = control.forced_response(C_L_h_bis, t_series, u_series)

plt.plot(response.time, response.outputs)
plt.title(
    'Response to a {:.1e}'.format(frequency) + '[Hz] sine $\\ddot{h}$ input of amplitude ' + '{:.1e} $[h/s^2]$'.format(amplitude))
plt.xlabel('t [s]')
plt.ylabel('$C_L$ $[-]$')
plt.show()

# impulse response - pitching
amplitude = np.pi/36  # [deg/s^2]
t_series = np.linspace(0, 20, 1000)
response = control.impulse_response(amplitude*C_L_alpha_bis, T=t_series)

plt.plot(response.time, response.outputs)
plt.title(
    'Response to an impulse $\\ddot{\\alpha}$ input of amplitude ' + '{:.1e} $[deg/s^2]$'.format(amplitude))
plt.xlabel('t [s]')
plt.ylabel('$C_L$ $[-]$')
plt.show()

# impulse response - plunging
amplitude = 0.1  # [m/s^2]
t_series = np.linspace(0, 20, 1000)
response = control.impulse_response(amplitude*C_L_h_bis, T=t_series)

plt.plot(response.time, response.outputs)
plt.title(
    'Response to an impulse $\\ddot{h}$ input of amplitude ' + '{:.1e} $[m/s^2]$'.format(amplitude))
plt.xlabel('t [s]')
plt.ylabel('$C_L$ $[-]$')
plt.show()

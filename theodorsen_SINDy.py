'''The following script demonstrates a preliminary (but rather succesful) attempt
at recreating the linear Theodorsen model approximation using SINDy. It is assumed
that all of the system states are available as measurements, which is not really
physical, but can be addressed later, e.g. by using a state observer.'''

import matplotlib.pyplot as plt
import numpy as np
import control
import Theodorsen_control as theodorsen
import pysindy

# GEOMETRY

a = 1.  # pitch axis wrt to 1/2-chord
b = 1.  # half-chord length of the airfoil
# default values of C_1 and C_2 used
airfoil = theodorsen.AirfoilGeometry(a=a, b=b)

# THEDORSEN MODEL

# the balanced truncation Theodorsen function approximation
theodorsen_function_sys = theodorsen.theodorsen_function_balanced_truncation_ss()

# state-space system with both α" and h" as inputs
theodorsen_full_sys = theodorsen.unsteady_lift_ss(
    airfoil, theodorsen_function_sys)

# INPUT SIGNALS

# harmonic alpha" signal
t = np.linspace(0, 50, 1000)
omega = 1
amplitude_alpha = 0.01
phase = np.pi/2
u_alpha = amplitude_alpha*np.sin(omega*t + phase)

# square wave h" signal
T_h = 5
phase_h = T_h/2
amplitude_h = 0.01/T_h
u_h = np.array([amplitude_h if np.floor((ti + phase_h)/T_h) %
                2 == 0 else -amplitude_h for ti in t])

u_MISO = np.vstack((u_h, u_alpha))

# TIME RESPONSE

output = control.forced_response(
    theodorsen_full_sys, T=t, U=u_MISO)

# TIME RESPONSE POSTPROCESSING

data = theodorsen.TheodorsenTimeResponse(
    output, inputs='both', sys=theodorsen_full_sys)

# SYSTEM IDENTIFCATION

# the optimizer has no thresholding for now, so it's basically least squares:
optimizer = pysindy.optimizers.stlsq.STLSQ(threshold=0)
# the model is fitted with 1st order polynomials, since we're reproducing a linear system;
# in theory, 0-th order terms are not necessary, but their coefficients are small as of now
model = pysindy.SINDy(optimizer=optimizer, feature_library=pysindy.PolynomialLibrary(degree=1))
# in theory this should work better, but at the moment the results are wrong; TODO check why:
# model.fit(data.x.T, t=data.t, x_dot=data.x_dot.T, u=u_MISO.T)
model.fit(data.x.T, t=data.t, u=u_MISO.T)
print('Identified model:')
model.print()

# VALIDATION

# for now comparison is made with the same inputs as the training data;
# # later we will need some proper validation procedures
x_model = model.simulate(x0=np.zeros((7, )), t=data.t, u=u_MISO.T)

# plot of the Theodorsen function states
for i in range(4):
    plt.subplot(4, 1, i+1)
    if i == 0:
        plt.title('States of the Theodorson function approximating model')
    plt.plot(data.t[:-1], data.x_theodorsen[i,:-1], '-', label='$x_{}$ model'.format(i))
    plt.plot(data.t[:-1], x_model[:,i], '--', label='$x_{}$ SINDy'.format(i))
    if i < 3:
        plt.tick_params('x', labelbottom=False)
    plt.ylabel('$x_{}$'.format(i))
    plt.legend()
plt.xlabel('t [-]')
plt.show()

# plot of the "physical" states
# h'
plt.subplot(3, 1, 1)
plt.title('Physical states of the system')
plt.tick_params('x', labelbottom=False)
plt.plot(data.t[:-1], data.h_dot[:-1], '-', label='model'.format(i))
plt.plot(data.t[:-1], x_model[:,4], '--', label='SINDy'.format(i))
plt.ylabel(r'$\dot{h}$')
plt.legend()
# alpha
plt.subplot(3, 1, 2)
plt.tick_params('x', labelbottom=False)
plt.plot(data.t[:-1], data.alpha[:-1], '-', label='model'.format(i))
plt.plot(data.t[:-1], x_model[:,5], '--', label='SINDy'.format(i))
plt.ylabel(r'$\alpha$')
plt.legend()
# alpha'
plt.subplot(3, 1, 3)
plt.plot(data.t[:-1], data.alpha_dot[:-1], '-', label='model'.format(i))
plt.plot(data.t[:-1], x_model[:,6], '--', label='SINDy'.format(i))
plt.xlabel('t [-]')
plt.ylabel(r'$\dot{\alpha}$')
plt.legend()
plt.show()
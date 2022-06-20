'''The following script demonstrates a preliminary (but rather succesful) attempt
at recreating the linear Theodorsen model approximation using SINDy. It is assumed
that all of the system states are available as measurements, which is not really
physical, but can be addressed later, e.g. by using a state observer.'''

import matplotlib.pyplot as plt
import numpy as np
import control
import Theodorsen_control as theodorsen
import pysindy
import signals
import sklearn as sk

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

# alpha" signal
t = np.linspace(0, 20, 1000)
u_alpha = signals.linear_chirp(t, omega_init=50, omega_end=0.1, amplitude=0.01)
# u_alpha = signals.prbs(t, dt=1, min=-0.001, max=0.001)
# u_alpha = signals.square_wave(t, T=10, phase=5, amplitude=0.001)
# u_alpha = signals.white_noise(t, sigma=0.01, mean=0)
# u_alpha = signals.white_noise_averaged(
#     t, sigma=0.001, mean=0, averaging_radius=5)

# h" signal
# u_h = signals.linear_chirp(t, omega_init=37, omega_end=0.4, amplitude=0.01)
u_h = signals.prbs(t, dt=0.1, min=-0.01, max=0.01)
# u_h = signals.square_wave(t, T=14, phase=-7, amplitude=0.001)
# u_h = signals.white_noise(t, sigma=0.01, mean=0)
# u_h = signals.white_noise_averaged(t, sigma=0.02, mean=0, averaging_radius=3)

u_MISO = np.vstack((u_h.T, u_alpha.T))

# TIME RESPONSE

output = control.forced_response(
    theodorsen_full_sys, T=t, U=u_MISO)

# TIME RESPONSE POSTPROCESSING

data = theodorsen.TheodorsenTimeResponse(
    output, inputs='both', sys=theodorsen_full_sys)

data.io_plot()

# SYSTEM IDENTIFCATION

AB = np.zeros((7, 10))
AB[:7, 1:8] = theodorsen_full_sys.A
AB[:7, 8:] = theodorsen_full_sys.B

# the optimizer has no thresholding for now, so it's basically least squares:
optimizer = pysindy.optimizers.stlsq.STLSQ(
    threshold=1e-2, initial_guess=AB)
# the model is fitted with 1st order polynomials, since we're reproducing a linear system;
# in theory, 0-th order terms are not necessary, but their coefficients are small as of now
model = pysindy.SINDy(optimizer=optimizer,
                      feature_library=pysindy.PolynomialLibrary(degree=1))
# in theory this should work better, but at the moment the results are wrong; TODO check why:
# model.fit(data.x.T, t=data.t, x_dot=data.x_dot.T, u=u_MISO.T)
model.fit(data.x.T, t=data.t, u=u_MISO.T)
print('Identified model:')
model.print()

# VALIDATION

# for now comparison is made with the same inputs as the training data;
# # later we will need some proper validation procedures

# simulating the output of the identified model
x_model = model.simulate(x0=np.zeros((7, )), t=data.t, u=u_MISO.T)

# plot of the Theodorsen function states
for i in range(4):
    plt.subplot(4, 1, i+1)
    if i == 0:
        plt.title('States of the Theodorson function approximating model')
    plt.plot(data.t[:-1], data.x_theodorsen[i, :-1],
             '-', label='$x_{}$ model'.format(i))
    plt.plot(data.t[:-1], x_model[:, i], '--', label='$x_{}$ SINDy'.format(i))
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
plt.plot(data.t[:-1], data.h_dot[:-1], '-', label='model')
plt.plot(data.t[:-1], x_model[:, 4], '--', label='SINDy')
plt.ylabel(r'$\dot{h}$')
plt.legend()
# alpha
plt.subplot(3, 1, 2)
plt.tick_params('x', labelbottom=False)
plt.plot(data.t[:-1], data.alpha[:-1], '-', label='model')
plt.plot(data.t[:-1], x_model[:, 5], '--', label='SINDy')
plt.ylabel(r'$\alpha$')
plt.legend()
# alpha'
plt.subplot(3, 1, 3)
plt.plot(data.t[:-1], data.alpha_dot[:-1], '-', label='model')
plt.plot(data.t[:-1], x_model[:, 6], '--', label='SINDy')
plt.xlabel('t [-]')
plt.ylabel(r'$\dot{\alpha}$')
plt.legend()
plt.show()
# C_L
# C_L_SINDy computed using the true C and D, because this usecase of SINDy does not estimate them
C_L_SINDy = (theodorsen_full_sys.C @ x_model.T).T + \
    (theodorsen_full_sys.D @ data.u[:, :-1]).T
rmse = sk.metrics.mean_squared_error(data.C_L[:-1], C_L_SINDy, squared=False)
plt.subplot(4, 1, 1)
plt.plot(data.t[:-1], data.C_L[:-1], '-', label='model')
plt.plot(data.t[:-1], C_L_SINDy, '--', label='SINDy')
plt.title('RMSE = {:.2e}'.format(rmse))
plt.ylabel('$C_L$')
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(data.t[:-1], C_L_SINDy - data.C_L[:-1], '-')
plt.ylabel('$\Delta C_L$')
plt.legend()

plt.subplot(4, 1, 3)
C_L_x_SINDy = (theodorsen_full_sys.C @ x_model.T).T
Du = (theodorsen_full_sys.D @ data.u).T
plt.plot(data.t[:-1], data.C_L[:-1] - Du[:-1], '-', label='model')
plt.plot(data.t[:-1], C_L_x_SINDy, '--', label='SINDy')
plt.ylabel('$C_L - Du$')
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(data.t[:-1], Du[:-1], '-')
plt.ylabel('$Du$')
plt.xlabel('t [-]')
plt.show()

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

# TRAINING DATA


class TrajectoryDatasets:
    '''Collection of unsteady lift time series (with states, inputs etc).'''

    def __init__(self, theodorsen_sys):
        self.x = []
        self.t = []
        self.u = []
        self.C_L = []
        self.data = []
        self.theodorsen_sys = theodorsen_sys

    def add_dataset(self, t, u_alpha, u_h, plot=True):
        '''Create a dataset with given time and inputs.'''
        u_MISO = np.vstack((u_h.T, u_alpha.T))

        # time response
        output = control.forced_response(
            self.theodorsen_sys, T=t, U=u_MISO)

        # time response postprocessing
        data = theodorsen.TheodorsenTimeResponse(
            output, inputs='both', sys=self.theodorsen_sys)

        if plot:
            data.io_plot()

        # new results-
        self.t.append(t)
        self.u.append(u_MISO)
        self.C_L.append(data.C_L)
        self.x.append(data.x)
        self.data.append(data)

    def C_L_without_Du(self, num=None):
        '''Compute C_L(t)-Du(t).'''
        if num is not None:
            Du = self.theodorsen_sys.D @ self.u[num]
            C_L_Du = self.C_L[num] - Du
        else:
            C_L_Du = []
            for i, C_L in enumerate(self.C_L):
                Du = self.theodorsen_sys.D @ self.u[i]
                C_L_Du.append(C_L - Du)
        return C_L_Du


datasets = TrajectoryDatasets(theodorsen_full_sys)

t_end = 20
t = np.linspace(0, t_end, 1000)

# TEST CASES
# these scenarios were tuned by hand in order to give a good accuracy

# CASE 1
u_alpha = signals.linear_chirp(t, omega_init=10, omega_end=0.1, amplitude=0.01)
u_h = signals.prbs(t, dt=0.1, min=-0.01, max=0.01)
datasets.add_dataset(t, u_alpha, u_h, plot=True)

# CASE 2
u_alpha = signals.square_wave(t, T=4, phase=2, amplitude=0.003)
u_h = signals.white_noise_averaged(t, sigma=0.01, mean=0, averaging_radius=5)
datasets.add_dataset(t, u_alpha, u_h, plot=True)

# CASE 3
u_alpha = signals.white_noise_averaged(
    t, sigma=0.003, mean=0, averaging_radius=5)
u_h = signals.square_wave(t, T=3.2, phase=1.6, amplitude=0.003)
datasets.add_dataset(t, u_alpha, u_h, plot=True)

# CASE 4
u_alpha = signals.prbs(t, dt=0.2, min=-0.01, max=0.01)
u_h = signals.linear_chirp(t, omega_init=10, omega_end=0.2, amplitude=0.01)
datasets.add_dataset(t, u_alpha, u_h, plot=True)

# CASE 5
t_long = np.linspace(0, 100, 1000)
u_alpha = signals.linear_chirp(
    t_long, omega_init=20, omega_end=0.1, amplitude=0.006)
u_h = signals.square_wave(t_long, T=6, phase=3, amplitude=0.01)
datasets.add_dataset(t_long, u_alpha, u_h, plot=True)

# SYSTEM IDENTIFCATION


def transpose_all(list_of_arrays):
    return [a.T for a in list_of_arrays]


# initial guess for the optimizer (perfect)
AB = np.zeros((7, 10))
AB[:7, 1:8] = theodorsen_full_sys.A
AB[:7, 8:] = theodorsen_full_sys.B


# optimizer settings
# alpha = 0 seems to work much better than the default
optimizer = pysindy.optimizers.stlsq.STLSQ(
    threshold=1e-2, alpha=0.0)  # , initial_guess=AB)

# the model is fitted with 1st order polynomials, since we're reproducing a linear system;
# in theory, 0-th order terms are not necessary, but their coefficients are small as of now
model = pysindy.SINDy(optimizer=optimizer,
                      feature_library=pysindy.PolynomialLibrary(degree=1))

model.fit(transpose_all(datasets.x), t=datasets.t,
          u=transpose_all(datasets.u), multiple_trajectories=True)
print('Identified model:')
model.print()

# VALIDATION

u_alpha = signals.square_wave(t, T=3, phase=1.5, amplitude=0.01)
u_h = signals.square_wave(t, T=2.1, phase=-1.1, amplitude=0.01)
u_MISO = np.vstack((u_h.T, u_alpha.T))

# time response
output = control.forced_response(
    theodorsen_full_sys, T=t, U=u_MISO)

# time response postprocessing
data = theodorsen.TheodorsenTimeResponse(
    output, inputs='both', sys=theodorsen_full_sys)

# simulating the output of the identified model
x_model = model.simulate(x0=np.zeros((7, )), t=t, u=u_MISO.T)

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
plt.tick_params('x', labelbottom=False)
# alpha
plt.subplot(3, 1, 2)
plt.tick_params('x', labelbottom=False)
plt.plot(data.t[:-1], data.alpha[:-1], '-', label='model')
plt.plot(data.t[:-1], x_model[:, 5], '--', label='SINDy')
plt.ylabel(r'$\alpha$')
plt.legend()
plt.tick_params('x', labelbottom=False)
# alpha'
plt.subplot(3, 1, 3)
plt.plot(data.t[:-1], data.alpha_dot[:-1], '-', label='model')
plt.plot(data.t[:-1], x_model[:, 6], '--', label='SINDy')
plt.xlabel('t [-]')
plt.ylabel(r'$\dot{\alpha}$')
plt.legend()
plt.show()

# plot of C_L
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
plt.tick_params('x', labelbottom=False)

plt.subplot(4, 1, 2)
plt.plot(data.t[:-1], C_L_SINDy - data.C_L[:-1], '-')
plt.ylabel('$\Delta C_L$')
plt.tick_params('x', labelbottom=False)

plt.subplot(4, 1, 3)
C_L_x_SINDy = (theodorsen_full_sys.C @ x_model.T).T
Du = (theodorsen_full_sys.D @ data.u).T
plt.plot(data.t[:-1], data.C_L[:-1] - Du[:-1], '-', label='model')
plt.plot(data.t[:-1], C_L_x_SINDy, '--', label='SINDy')
plt.ylabel('$C_L - Du$')
plt.legend()
plt.tick_params('x', labelbottom=False)

plt.subplot(4, 1, 4)
plt.plot(data.t[:-1], Du[:-1], '-')
plt.ylabel('$Du$')
plt.xlabel('t [-]')
plt.show()

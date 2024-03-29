"""
Created on Mon May 16 2022

@author: Maciej Michałów

This library provides a functional approximated implementation of the original
Theodorsen model, using the Python control systems library and the Python Control
Systems Library(https://python-control.readthedocs.io/en/0.9.1/).

The system is modelled using a state-space model approach; all equations have been
adapted from S. L. Brunton and C. W. Rowley, ‘Empirical state-space representations
for Theodorsen’s lift model’, Journal of Fluids and Structures, vol. 38, pp. 174–186,
april 2013, doi: 10.1016/j.jfluidstructs.2012.10.005.

At the bottom, an example is provided."""

import numpy as np
import matplotlib.pyplot as plt
import control  # https://python-control.readthedocs.io/en/0.9.1/


class AirfoilGeometry:
    '''Class representing a given thin airfoil geometry.'''

    def __init__(self, a, b, C_1=np.pi, C_2=2*np.pi):
        '''Input:
        a - pitch axis wrt to 1/2-chord
        b - half-chord length of the airfoil
        C_1  - added-mass coefficient
        C_2  - steady-state lift slope (dC_L/dα)'''
        self.a = a
        self.b = b
        self.C_1 = C_1
        self.C_2 = C_2


def theodorsen_function_balanced_truncation_ss():
    '''Return a state-space approximation of the Theodorsen function.

        Output:
        sys - a control.StateSpace representing the Theodorsen function approximation

        The state-space model is of the form x' = Ax + Bu, y = Cx + Du.

        The matrices have been copied from:
        S. L. Brunton and C. W. Rowley, ‘Empirical state-space representations for Theodorsen’s lift model’,
        Journal of Fluids and Structures, vol. 38, pp. 174–186, april 2013, doi: 10.1016/j.jfluidstructs.2012.10.005.


        For reference on the contro.StateSpace class, see:
        https://python-control.readthedocs.io/en/0.9.1/generated/control.StateSpace.html#control.StateSpace'''
    dim = 4
    A = np.zeros((dim, dim))
    A[0, :] = np.array([-1.158, -0.3052, -0.02028, -2.325e-4])
    A[1:4, 0:3] = np.eye(3)
    B = np.array([1., 0., 0., 0.])
    C = np.array([0.124, 0.08667, 0.008805, 1.156e-4])
    D = np.array(0.5)
    sys = control.ss(A, B, C, D)
    return sys


def unsteady_lift_ss(airfoil, theodorsen_sys, inputs='both', minimal=False):
    '''Return a state-space approximation of the Theodorsen non-dimensionalised unsteady lift model.

    Input:
    airfoil - an AirfoilGeometry object with dimensions and aerodynamic coefficients of the airfoil
    theodorsen_sys - a state-space model of the Theodorsen function
    inputs - the inputs to the system: either 'h', 'alpha' or (by default) 'both'
    minimal - if True, uses the effective angle of attack as a state instead of AoA and h'

    Output:
    sys - a control.StateSpace representing the Theodorsen function approximation

    The state-space model is of the form x' = Ax + Bu, y = Cx + Du.

    The matrices have been copied from:
    S. L. Brunton and C. W. Rowley, ‘Empirical state-space representations for Theodorsen’s lift model’,
    Journal of Fluids and Structures, vol. 38, pp. 174–186, april 2013, doi: 10.1016/j.jfluidstructs.2012.10.005.

    For reference on the contro.StateSpace class, see:
    https://python-control.readthedocs.io/en/0.9.1/generated/control.StateSpace.html#control.StateSpace'''

    dim_theodorsen = theodorsen_sys.nstates

    if minimal:
        # state-space model with effective AoA as state
        A = np.hstack((theodorsen_sys.A, theodorsen_sys.B *airfoil.C_2,
            theodorsen_sys.B*airfoil.C_2*(1-2*airfoil.a)/2))
        A = np.vstack((A, np.zeros((2, dim_theodorsen+2))))
        A[-2, -1] = 1
        B = np.zeros((dim_theodorsen+2, 2))
        B[dim_theodorsen, 0] = 1
        B[-1, -1] = 1
        C = np.hstack((theodorsen_sys.C, theodorsen_sys.D*airfoil.C_2,
            airfoil.C_1 + theodorsen_sys.D*airfoil.C_2*(1-2*airfoil.a)/2))
        D = np.array([airfoil.C_1, -airfoil.C_1*airfoil.a])
    else:
        # state-space model with AoA and h' as states
        A = np.hstack((theodorsen_sys.A, theodorsen_sys.B*airfoil.C_2, theodorsen_sys.B *
                    airfoil.C_2, theodorsen_sys.B*airfoil.C_2*(1-2*airfoil.a)/2))
        A = np.vstack((A, np.zeros((3, dim_theodorsen+3))))
        A[-2, -1] = 1
        B = np.zeros((dim_theodorsen+3, 2))
        B[dim_theodorsen, 0] = 1
        B[-1, -1] = 1
        C = np.hstack((theodorsen_sys.C, theodorsen_sys.D*airfoil.C_2, theodorsen_sys.D*airfoil.C_2,
                    airfoil.C_1 + theodorsen_sys.D*airfoil.C_2*(1-2*airfoil.a)/2))
        D = np.array([airfoil.C_1, -airfoil.C_1*airfoil.a])

    if inputs == 'both':
        # LTI model of the Theodorsen model with h" and α" inputs
        sys = control.ss(A, B, C, D)
    elif inputs == 'alpha':
        # LTI model of the Theodorsen model limited to α" input
        sys = control.ss(A, B[:, 1], C, D[1])
    elif inputs == 'h':
        # LTI model of the Theodorsen model limited to h" input
        sys = control.ss(A, B[:, 0], C, D[0])

    return sys


class TheodorsenTimeResponse:
    '''Class for postprocessing of Theodorsen model output data.'''

    def __init__(self, output, inputs='both', sys=None):
        '''
        Input:
        output - a control.TimeResponseData object, generated by a simulation
        inputs - the inputs to the system: either 'h', 'alpha' or (by default) 'both
        sys - the LTI system used to generate the data (for computing state derivatives)'''

        self.t = output.time
        self.x = output.states
        self.x_theodorsen = output.states[0:4, :]
        self.h_dot = output.states[4]
        self.alpha = output.states[5]
        self.alpha_dot = output.states[6]
        self.alpha_e = self.alpha + self.h_dot
        self.C_L = output.outputs.T
        self.inputs = inputs
        self.u = output.inputs

        if self.inputs == 'h':
            self.h_ddot = output.inputs
        elif self.inputs == 'alpha':
            self.alpha_ddot = output.inputs
        elif self.inputs == 'both':
            self.h_ddot = output.inputs[0]
            self.alpha_ddot = output.inputs[1]

        if sys is not None:
            self.sys = sys
            self.x_dot = np.zeros(self.x.shape)
            for i in range(len(self.t)):
                self.x_dot[:, i] = self.sys.A @ \
                    self.x[:, i] + self.sys.B @ self.u[:, i]

    def state_plot(self):
        '''Plot the C_L response together with physicaly meanignful states.'''

        # C_L
        plt.subplot(3, 1, 1)
        plt.plot(self.t, self.C_L)
        plt.title('Time response of unsteady lift - states')
        plt.ylabel(r'$C_L [-]$')
        plt.tick_params('x', labelbottom=False)
        plt.grid()

        # angles and h' (equivalent to an angle)
        plt.subplot(3, 1, 2)
        plt.plot(self.t, self.h_dot, label=r'$\dot{h} [m/s]$')
        plt.plot(self.t, self.alpha, label=r'$\alpha [rad]$')
        plt.plot(self.t, self.alpha_e, label=r'$\alpha_e [rad]$')
        plt.ylabel('angle [rad]')
        plt.tick_params('x', labelbottom=False)
        plt.grid()
        plt.legend()

        # alpha'
        plt.subplot(3, 1, 3)
        plt.plot(self.t, self.alpha_dot)
        plt.ylabel(r'$\dot{\alpha} [rad/s]$')
        plt.xlabel('$t$')
        plt.legend()
        plt.grid()
        plt.show()

    def theodorsen_state_plot(self):
        '''Plot the Theodorsen function model states.'''
        for i in range(4):
            plt.subplot(4, 1, i+1)
            if i == 0:
                plt.title(
                    'States of the Theodorson function approximating model')
            plt.plot(self.t, self.x_theodorsen[i, :], '-')
            if i < 3:
                plt.tick_params('x', labelbottom=False)
            plt.ylabel('$x_{}$'.format(i))
            plt.grid()
        plt.xlabel('t [-]')
        plt.show()

    def phase_plot(self, state='alpha_e'):
        '''Plot the unsteady lift as a function of one of the state parameters.

        Inputs:
        state - the state used for the x-axis: either 'h_dot', 'alpha' or (by default) 'alpha_e'''

        states = {'alpha_e': self.alpha_e,
                  'alpha': self.alpha, 'h_dot': self.h_dot}
        labels = {'alpha_e': '$\\alpha_e$',
                  'alpha': '$\\alpha$', 'h_dot': '$\dot{h}$'}
        plt.plot(states[state], self.C_L)
        plt.title('Phase plot of unsteady lift')
        plt.xlabel(labels[state])
        plt.ylabel('$C_L$')
        plt.grid()
        plt.show()

    def io_plot(self):
        '''Plot the C_L response together with the inputs.'''

        if self.inputs != 'both':
            plt.subplot(2, 1, 1)
        else:
            plt.subplot(3, 1, 1)

        plt.plot(self.t, self.C_L)
        plt.title('Time response of unsteady lift - inputs')
        plt.ylabel(r'$C_L [-]$')
        plt.tick_params('x', labelbottom=False)
        plt.grid()

        if self.inputs == 'h' or self.inputs == 'alpha':
            plt.subplot(2, 1, 2)
        elif self.inputs == 'both':
            plt.subplot(3, 1, 2)

        if self.inputs == 'h' or self.inputs == 'both':
            plt.plot(self.t, self.h_ddot)
            plt.ylabel(r'$\ddot{h} [rad/s^2]$')
            plt.grid()

        if self.inputs == 'both':
            plt.tick_params('x', labelbottom=False)
            plt.subplot(3, 1, 3)

        if self.inputs == 'alpha' or self.inputs == 'both':
            plt.plot(self.t, self.alpha_ddot)
            plt.ylabel(r'$\ddot{\alpha} [rad/s^2]$')
            plt.grid()

        plt.xlabel('$t$')
        plt.show()
        
def sinusoidalInputs(
        t, # time vector
        amp_scale, # maximum amplitude
        N, # number of terms
        second_derivative = True 
        ):
    coeff = amp_scale * np.random.rand(N,1)
    f = 0
    df = 0
    if second_derivative: ddf = 0
    
    for i in range(1, N+1):
        f += coeff[i-1] * np.sin(i * t)
        df += coeff[i-1] * i * np.cos(i * t)
        if second_derivative:
            ddf -= coeff[i-1] * i**2 * np.sin(i * t)
    
    if second_derivative:
        return [f, df, ddf], np.array([0, df[0], 0])
    else:
        return [f, df], np.array([0, df[0]])
        

# example script using the library
if __name__ == '__main__':
    a = 1/2  # pitch axis wrt to 1/2-chord
    b = 1  # half-chord length of the airfoil
    airfoil = AirfoilGeometry(a=a, b=b)  # default values of C_1 and C_2 used

    # the balanced truncation Theodorsen function approximation
    theodorsen_function_sys = theodorsen_function_balanced_truncation_ss()

    # Bode plot of the approximated Theodorsen function
    control.bode_plot(theodorsen_function_sys, dB=True,
                      omega_limits=[1e-2, 1e2])
    plt.show()

    # generation of unsteady lift state-space systems

    # state-space system with both α" and h" as inputs
    theodorsen_full_sys = unsteady_lift_ss(airfoil, theodorsen_function_sys)
    # state-space system with both α" as input
    theodorsen_alpha_sys = unsteady_lift_ss(
        airfoil, theodorsen_function_sys, inputs='alpha')
    # state-space system with both h" as input
    theodorsen_h_sys = unsteady_lift_ss(
        airfoil, theodorsen_function_sys, inputs='h')

    # bode plots of the single-input, single-output systems
    control.bode_plot(theodorsen_alpha_sys, dB=True, omega_limits=[1e-3, 1e3])
    plt.show()
    control.bode_plot(theodorsen_h_sys, dB=True, omega_limits=[1e-3, 1e3])
    plt.show()

    # response to an h" impulse (analogous to a Wagner function)
    t = np.linspace(0, 50, 1000)
    output = control.impulse_response(theodorsen_h_sys, T=t)
    data_h_impulse = TheodorsenTimeResponse(output, inputs='h')
    data_h_impulse.state_plot()
    data_h_impulse.theodorsen_state_plot()

    # response to a harmonic alpha" signal
    omega = 1
    amplitude = 0.1
    phase = np.pi/2
    u_alpha = amplitude*np.sin(omega*t + phase)
    output = control.forced_response(
        theodorsen_alpha_sys, 
        T=t, 
        U=u_alpha)
    data_alpha_sine = TheodorsenTimeResponse(output, inputs='alpha')
    data_alpha_sine.phase_plot(state='alpha')
    data_alpha_sine.io_plot()

    # response to harmonic alpha" and square h" signals
    T = 5
    phase = T/2
    amplitude = 0.1/T
    # square wave
    u_h = np.array([amplitude if np.floor((ti+phase)/T+T/2) %
                   2 == 0 else -amplitude for ti in t])
    u_MISO = np.vstack((u_h, u_alpha))
    output = control.forced_response(
        theodorsen_full_sys, 
        T=t, 
        U=u_MISO)
    data_both = TheodorsenTimeResponse(output, inputs='both')
    data_both.phase_plot(state='alpha_e')
    data_both.io_plot()
    data_both.state_plot()
    data_both.theodorsen_state_plot()

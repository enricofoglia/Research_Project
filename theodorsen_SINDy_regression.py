'''The following script demonstrates a preliminary (but rather succesful) attempt
at recreating the linear Theodorsen model approximation using SINDy. It is assumed
that all of the system states are available as measurements, which is not really
physical, but can be addressed later, e.g. by using a state observer.'''

import matplotlib.pyplot as plt
import numpy as np
import control
import Theodorsen_control as theodorsen
import pysindy as ps
from sklearn.metrics import r2_score
from PolynomialChaos import *

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

# CONSTRUCTION OF DATA MATRIX
X_train = np.stack([data.alpha_e[:501], data.alpha_dot[:501], data.alpha_ddot[:501], data.h_ddot[:501]], axis = -1) # physical states of the system
CL_train = data.C_L[:501] # output of the system

X_test = np.stack([data.alpha_e[501:], data.alpha_dot[501:], data.alpha_ddot[501:], data.h_ddot[501:]], axis = -1)
CL_test = data.C_L[501:]

# ORTHOGONAL BASIS

 # Generation of aPC library
expansionDegree = 1

aPC = PolynomialChaos(
     X_train,
     expansionDegree = expansionDegree,
     numberOfInputs = 4)
aPC.ComputeCoefficients(threshold = 1e-4, normalize = True)
coefficients = aPC.coefficients
AlphaMatrix = aPC.AlphaMatrix
 
LibraryList = GenerateLibraryList(
     expansionDegree=expansionDegree,
     coefficients = coefficients,
     AlphaMatrix = AlphaMatrix)

# SETTING UP SINDy 
optimizer = ps.optimizers.stlsq.STLSQ(threshold = 0.0, alpha = 1e-05, max_iter = 50)
library = ps.feature_library.polynomial_library.PolynomialLibrary(degree = expansionDegree) # standard polynomial library
library = ps.feature_library.custom_library.CustomLibrary(LibraryList)

model = ps.SINDy(optimizer = optimizer, 
			     feature_library = library,
			     feature_names = ['alpha_e', 'alpha_dot', 'alpha_ddot', 'h_ddot']) # default paramaters:
t = data.t
model.fit(X_train, t = t[1] - t[0], x_dot = CL_train)

model.print()
CL_SINDy = model.model.predict(X_train)
CL_SINDy_test = model.model.predict(X_test)

# PRINTING OUT LOUD
print('\n------------- PRINTS --------------\n')
print(f'R score train: {r2_score(CL_SINDy, CL_train)}')
print(f'R score test: {r2_score(CL_SINDy_test, CL_test)}')

# THE COLORS UAO

fig, ax = plt.subplots(1,2, constrained_layout=True)
ax[0].plot(t[:501], CL_train)
ax[0].set_xlabel('Adimensional time [-]')
ax[0].set_ylabel(r'$C_l$')
ax[0].set_title('Collected data')
ax[0].grid(True)

ax[1].plot(t[:501], CL_SINDy)
ax[1].set_xlabel('Adimensional time [-]')
ax[1].set_ylabel(r'$C_l$')
ax[1].set_title('SINDy fitted model')
ax[1].grid(True)

fig, ax = plt.subplots()
ax.plot(t[501:], CL_test)
ax.plot(t[501:], CL_SINDy_test)
ax.set_xlabel('Adimensional time [-]')
ax.set_ylabel(r'$C_l$')
ax.set_title('Comparison Actual vs Predicted')
ax.legend(['Actual Data', 'SINDy data'])
ax.grid(True)


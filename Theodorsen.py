#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 16:50:01 2022

@author: Enrico Foglia

"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt 

class StateSpace:
    def __init__(self,A, B, C, D):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
    
    def dxdt(self, x, t, u):
        return np.matmul(self.A,x) + np.matmul(self.B, u(t))
    
    def solve(self, x_0, t, u):
        self.state = odeint(self.dxdt, x_0, t, args = (u,))
        self.output = np.matmul(self.C, self.state.T) + np.matmul(self.D, u(t))
        return self.output
    

#%% Theodorsen Balanced Truncation r = 4
A_tilde = np.array([[-1.158, -0.3052, -0.02028, -2.325E-4],[1, 0,0,0],[0, 1, 0, 0],[0, 0, 1, 0]])
B_tilde = np.array([[1, 0,0,0]]).T
C_tilde = np.array([[0.124, 0.08667, 0.008805, 1.156E-4]])
D_tilde = np.array([[0.5]])
def u(t):
    return np.array([0])

t = np.linspace(0,10)
Th = StateSpace(A_tilde, B_tilde, C_tilde, D_tilde)
x0 = np.array([1,1,1,1]).T
Cl = Th.solve(x0, t, u)
fig, ax = plt.subplots()
ax.plot(t, Cl.T)



#%% Forced spring-mass-damper system

# csi = 0.2
# omega = 1
# A = np.array([[-2*csi*omega, -omega**2],[1,0]])
# B = np.array([[1,0]]).T  # Careful about the dimentions!
# C = np.array([[1,0],[0,1]])
# D = np.array([[0,0]]).T  # Careful about the dimentions!

# def u(t):
#     return np.array([np.sin(t)]) # We had here some problems with the dimentions

# t = np.linspace(0,10)
# SS = StateSpace(A,B,C,D)
# x = SS.solve(np.array([0,1]).T, t, u)

# figure, ax = plt.subplots()
# ax.plot(t,x)

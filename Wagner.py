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

"""
import numpy as np
from scipy.special import jv, yn # Bessel functions
from scipy.integrate import quad

def Wagner(t, large_times = False): 
    if large_times == False:
        G= lambda k: (jv(1,k)*(jv(1,k)+yn(0,k))+yn(1,k)*(yn(1,k)-jv(0,k)))/((jv(1,k) + yn(0,k))**2 + (yn(1,k) - jv(0,k))**2)
        phi = 1/2 + 2/np.pi*quad(lambda k: 1/k*(G(k)-1/2)*np.sin(k*t), 0, 10, limit = int(100*t)+50)[0]
    else:    
        G= lambda k: (yn(1,k)*yn(0,k)+jv(1,k)*jv(0,k))/((jv(1,k) + yn(0,k))**2 + (yn(1,k) - jv(0,k))**2)
        phi = 1 - 2/np.pi*quad(lambda k: 1/k*G(k)*np.cos(k*t), 0, 10, limit = int(100*t))[0]
    return phi


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    
    W = []
    T = np.logspace(-1,3, num = 100)
    for t in T:
        if t<=100:
            W.append(Wagner(t, large_times = False))
        else:
            W.append(Wagner(t, large_times = True))
        
    fig, ax = plt.subplots()
    ax.plot(T,np.ones(len(W))-W)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.grid(True)
    ax.set_xlabel('time')
    ax.set_ylabel('$1-\phi (t)$')
    ax.set_title('Wagner function')
    plt.show()
    
    # Here a couple lines of code to see that the part of the integral that I'm
    # cutting is not that important after all.
    
    # t = 1e4
    # G= lambda k: (yn(1,k)*yn(0,k)+jv(1,k)*jv(0,k))/((jv(1,k) + yn(0,k))**2 + (yn(1,k) - jv(0,k))**2)
    # k = np.linspace(1e-10,20,int(100*t))
    # f = lambda k: 1/k*G(k)*np.cos(k*t)
    # kInf = 10
    # kSup = 100
    # integral, error = quad(f, kInf, kSup, limit = int(100*t))
    # print(f'The integral from k = {kInf} to k = {kSup} is:\nI = {integral} (error = {error})')    
    # fig,ax = plt.subplots()
    # ax.plot(k,f(k))
    # ax.set_xlabel('k')
    # ax.set_ylabel('f(k)')
    # ax.grid(True)
    
    
    
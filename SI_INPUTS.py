#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 16:12:10 2022

@author: enrico foglia

Compute various inputs for System Identification

"""
import numpy as np


def prbs31(seed, n_bits = 32):
    mask = 2**n_bits-1
    for i in range(n_bits):
        next_bit = ~((seed>>30) ^ (seed>>27))&0x01
        seed = ((seed<<1) | next_bit) & mask
    return format(seed, 'b')

def PRBS_input(t, n_steps = 1):
    PRBS = prbs31(
        seed = 2**13,
        n_bits = len(t))
    signal = np.zeros(len(t))
    for i in range(len(t)):
        if PRBS[i] == '1': next_bit = 1
        else: next_bit = -1
    
        if i % n_steps == 0:
            new_bit = PRBS[i]
            if PRBS[i] == '1': new_bit = 1
            else: new_bit = -1
        signal[i] = new_bit
        
    return signal

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    t = np.linspace(0, 100, 1000)
    signal = PRBS_input(t, n_steps = 20)
    plt.plot(t, signal)
    
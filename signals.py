import numpy as np
import scipy.interpolate


def linear_chirp(t, omega_init, omega_end, amplitude):
    '''Chirp signal with linearly progressing frequency.
    
    Inputs:
    t - time [s]
    omega_init - frequency at t=0 [rad/s]
    omega_end - frequency at the end [rad/s]
    amplitude - multiplicative amplitude (scalar or array of size t.shape)'''
    t_end = t[-1]
    return amplitude*np.sin((omega_end-omega_init) /
               (2*t_end)*t**2 + omega_init*t)


def square_wave(t, T, phase, amplitude):
    '''Square wave signal with equal upper and lower portions.
    
    Inputs:
    t - time [s]
    T - length of one half-pulse (total period is 2*T) [t]
    amplitude - multiplicative amplitude (scalar or array of size t.shape)'''
    return amplitude*np.array([1. if np.floor((ti + phase)/T) %
                               2 == 0 else -1. for ti in t])


def sine_wave(t, omega, phase, amplitude):
    '''Sine wave.
    
    Inputs:
    t - time [s]
    omega - frequency [rad/s]
    phase - phase angle [rad]
    amplitude - multiplicative amplitude (scalar or array of size t.shape)'''
    return amplitude*np.sin(omega*t + phase)


def white_noise(t, sigma, mean=0.):
    '''Discrete white noise sampled from a gaussian distribution.
    
    Inputs:
    t - time
    sigma - standard deviation
    mean - mean value'''
    return np.random.normal(mean, sigma, size=len(t))


def _moving_average(array, n=3):
    return np.array([np.average(array[np.max([0, i-n]):np.min([len(array)-1, i+n])]) for i in range(len(array))])


def white_noise_averaged(t, sigma, mean=0., averaging_radius=3):
    '''Discrete white noise smoothened by a moving average.
    
    Inputs:
    t - time
    sigma - standard deviation
    mean - mean value
    averaging_radius - radius of averaging window'''
    return _moving_average(np.random.normal(mean, sigma, size=len(t)), n=averaging_radius)


def _LFSR31(N, seed):
    m = 31
    c = np.zeros(m)
    c[0] = 1
    c[-1] = 1
    c[27] = 1
    output = np.zeros((N, 1))
    state = seed
    for i in range(0, N):
        next_bit = np.mod(np.sum(c*state), 2)
        output[i] = state[m-1]
        state = np.concatenate((np.array([next_bit]), state[0:m-1]))
    return output


def prbs(t, dt, min=0., max=1., seed=None):
    '''Pseudo-random binary signal based on PRBS31.

    
    
    Inputs:
    t - time
    dt - minimum time between jumps
    min - lower value
    max - upper value
    seed - seed for the LSFR'''
    N = int(t[-1]/dt)

    t_jumps = np.linspace(t[0], t[-1], N)

    if seed is None:
        seed = np.random.randint(low=0, high=2, size=31)

    binary_signal = np.where(_LFSR31(N, seed) > 0.5, max, min)
    interp_function = scipy.interpolate.interp1d(
        t_jumps, binary_signal, kind='previous', fill_value=0., axis=0)
    return interp_function(t)

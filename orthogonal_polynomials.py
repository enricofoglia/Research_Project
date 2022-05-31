#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 17:04:46 2022

@author: Enrico Foglia

Calculate orthogonal basis functions
TODO: implement higher dimensional data

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from scipy import stats
import sympy as sym

class OrthogonalPolynomials():
    
    def __init__(
            self, 
            X, 
            degree = 5, 
            ):
        self.X = X
        self.degree = degree
        
    def get_X_dimensions(self):
        ''' 
        Returns shape of the input matrix by assuming that the number of samples 
        (n) is greater than the number of state variables (m)
        '''
        nx, mx = self.X.shape
        if nx > mx: 
            n, m = nx, mx
        else:
            m, n = nx, mx
            self.X = self.X.T
            
        return n, m
        
    def pmf(self, plot = False):
        '''
        Generate histogram (probability mass function) for the distribution of 
        the data. 
        Gives the possibility to plot it if necessary
        '''
        n, m = self.get_X_dimensions()
        bins = []
        for i in range(m):
            bins.append(len(np.histogram_bin_edges(self.X[:,i], 'fd'))-1) # optimal numbre of bins using Freedman Diaconis Estimator
        
        self.pmf, self.bin_edges = np.histogramdd(self.X, bins, density = True)
        if plot:
           
            if m > 1:
                raise ValueError(
                    'Can only plot one dimensional histograms'
                    )
            else:
                fig, ax = plt.subplots()
                ax.hist(self.X, self.bin_edges[0])
                ax.set_xlabel('X')
                ax.set_ylabel('p(X)')
                ax.set_title('Probability mass function of X')
                plt.show()
        
        return self.pmf, self.bin_edges
    
    def interpolate_pmf(self):
        '''
        Interpolate histogram on bin edges
        '''
        interp_pmf =  stats.rv_histogram((self.pmf, self.bin_edges[0]))
        return interp_pmf.pdf(self.bin_edges[0])
        
    
    def orthogonal_polynomial(self):
        '''
        Calculates the coefficients for the orthogonal polynomial basis
        At the moment supports only single variable polynomials.
        To implement the weight function
        '''
        
        def alpha(self, pi):
            x0 = sym.symbols("x0")
            integrand_num = sym.lambdify(x0, pi[1] * pi[1] * x0, "numpy")
            integrand_den = sym.lambdify(x0, pi[1] * pi[1], "numpy")
            numerator = simpson(integrand_num(self.bin_edges[0]) * self.interpolate_pmf(), self.bin_edges[0])
            denominator = simpson(integrand_den(self.bin_edges[0]) * self.interpolate_pmf(), self.bin_edges[0])
            return numerator / denominator
        
        def beta(self, pi):
            x0 = sym.symbols("x0")
            integrand_num = sym.lambdify(x0, pi[1] * pi[1], "numpy")
            integrand_den = sym.lambdify(x0, pi[0] * pi[0], "numpy")
            numerator = simpson(integrand_num(self.bin_edges[0]) * self.interpolate_pmf(), self.bin_edges[0])
            if pi[0] == x0**0:
                integrand_den = lambda x: x**0
            denominator = simpson(integrand_den(self.bin_edges[0]) * self.interpolate_pmf(), self.bin_edges[0])
            return numerator / denominator
        
        n, m = self.get_X_dimensions()
        x = []
        for i in range(m): x.append(sym.symbols(f'x{i}')) 
        
        pi = [x[0]**0, x[0]]

        for deg in range(2,self.degree+1):
            alpha_k = alpha(self,pi)
            beta_k = beta(self,pi)
            pi_new = (x[0]-alpha_k) * pi[1] - beta_k * pi[0]
            pi[0] = pi[1]
            pi[1] = pi_new
            
        self.coefficients = sym.Poly(pi[1]).all_coeffs()
        return sym.lambdify(x[0],pi[1])
        

if __name__ == '__main__':
    np.random.seed(43)
    x = np.random.randn(10000, 1)
    a = np.max(x)
    b = np.min(x)
    X = OrthogonalPolynomials(1/(b-a)*(2*x-a-b), degree = 2)
    
    hist, bin_edges = X.pmf(plot = True)
    ortho_func = X.orthogonal_polynomial()
    print(X.coefficients)

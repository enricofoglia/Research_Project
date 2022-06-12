#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 10:42:55 2022

@author: Enrico Foglia

Implement aPC (arbitrary Polynomial Chaos) from the matlab code:
    Sergey Oladyshkin (2022). aPC Matlab Toolbox: Data-driven Arbitrary Polynomial Chaos
    (https://www.mathworks.com/matlabcentral/fileexchange/72014-apc-matlab-toolbox-data-driven-arbitrary-polynomial-chaos),
    MATLAB Central File Exchange. Retrieved June 10, 2022.
    
Based on the paper:
    [1] Oladyshkin, S., & Nowak, W. (2012). Data-driven uncertainty quantification 
    using the arbitrary polynomial chaos expansion. Reliability Engineering & 
    System Safety, 106, 179-190.
    
Create library object to be used with SINDy
"""

import numpy as np
from MultivariatePolynomialIndex import MultivariatePolynomialIndex
import sympy as sym

class PolynomialChaos():
    '''
    Basic class for the Polynomial Chaos Expansion
    '''
    def __init__(self, 
                 distribution,
                 expansionDegree,
                 numberOfInputs):
        self.expansionDegree = expansionDegree
        self.distribution = distribution
        self.numberOfInputs = numberOfInputs
        
    def ComputeMoments(self, distribution_1D):
        '''
        Compute the statistical moments of a distribution (1D) up to the 
        2*expansionDegree (2*expansionDegree-1 would be sufficient, the last 
        could be useful for a furteher versions that implement normalization in 
        order to have orthonormal base)
        '''
        numberOfSamples = len(distribution_1D)
        DistributionMoments = np.array([np.sum(distribution_1D**i)/numberOfSamples for i in range((self.expansionDegree+1)*2)])
        return DistributionMoments
        
    def MomentMatrix(self, distribution_1D, polynomialDegree):
        '''
        Generate the moment matrix to compute the coefficients for a polynomial 
        of degree polynomialDegree, as explained in the reference paper [1]
        '''
        d = polynomialDegree + 1
        Hankel = np.zeros((d,d)) # moments matrix initialization
        moments = self.ComputeMoments(distribution_1D)
        for i in range(polynomialDegree+1):
            for j in range(polynomialDegree+1):
                if i < polynomialDegree:
                    Hankel[i,j] = moments[i+j]
                else:
                    Hankel[i,-1] = 1
        return Hankel
    
    def aPC_OneDimensional(self, distribution_1D):
        '''
        Computes and returns the coefficient matrix for a 1D distribution from the 0-degree
        polynomial up to the one of degree expansionDegree.
        '''
        d = self.expansionDegree + 1
        coefficients = np.zeros((d,d))
        for i in range(d):
            H = self.MomentMatrix(distribution_1D,i)
            v = np.zeros(i+1)
            v[-1] = 1
            coefficients[0:i+1,i] = np.linalg.solve(H,v)
        # coefficients = np.reshape(coefficients,(d,d,1))
        return coefficients

    def ComputeCoefficients(self):
        '''
        Computes the coefficient for the PC expansion (in general multidimensional).
        Makes use of the MultivariatePolynomialsIndex function to generate the 
        Alpha matrix useful for the construction of the base.
        The coefficient tensor and the Alpha matrix are enough to fully characterize
        the PC expansion
        
        The coefficient tensor has three dimensions, as:
            - the first represents the order of the sigle term (from 0 to expansionDegree)
            - the second the total degree of the polynomial (from 0 to expansionDegree)
            - the third the variable (from 1 to numberOfInputs)
        '''
        d = self.expansionDegree + 1
        if self.numberOfInputs == 1:
            self.coefficients = np.reshape(self.aPC_OneDimensional(self.distribution), (d,d,1))
            self.AlphaMatrix = np.array([range(d)]).T
        else:
            self.coefficients = np.zeros((d,d, numberOfInputs))
            for i in range(numberOfInputs):
                self.coefficients[:,:,i] = self.aPC_OneDimensional(self.distribution[:,i])
            self.AlphaMatrix = MultivariatePolynomialIndex(numberOfInputs, d-1)

            
def GenerateLibraryList(
        expansionDegree,
        coefficients,
        AlphaMatrix
        ):
    '''
    Given the Alpha matrix and the coefficient tensor conputes a list of functions
    ready to be transformed into a SINDy library.
    '''
    M , numberOfInputs = AlphaMatrix.shape # M = total number of terms in the expansion
    x = []
    for i in range(numberOfInputs): x.append(sym.symbols(f'x{i}')) # list of symbolic variables
    LibraryList = []
    for i in range(M): # order
        index = AlphaMatrix[i,:]
        MultivariatePolynomial = 1
        for j in range(numberOfInputs): # variable
            coeff = coefficients[:, index[j], j] 
            coeff = np.flip(coeff) # The MultivariatePolynomials function gives the coefficients from 0 to max_deg, while Poly starts from max_deg and goes to 0
            Polynomial1D = sym.Poly(coeff, x[j])
            MultivariatePolynomial = MultivariatePolynomial * Polynomial1D # multivaried polynomial object
            MultivariatePolynomial = MultivariatePolynomial.as_expr() 
            
        LibraryList.append(sym.lambdify(x, MultivariatePolynomial, 'numpy'))
        
    return LibraryList
        

if __name__ == '__main__':
    np.random.seed(43)
    n = 1000
    data = np.zeros((n,3))
    data_uniform = np.linspace(-1,1,n)
    data_uniform = np.array([data_uniform])
    data[:,0] = data_uniform
    data[:,1] = np.random.randn(n)
    data[:,2] = np.random.randn(n)
    
    expansionDegree = 5
    numberOfInputs = 3
    
    aPC = PolynomialChaos(data, expansionDegree, numberOfInputs)
    aPC.ComputeCoefficients()
    coefficients = aPC.coefficients
    A = aPC.AlphaMatrix
    
    LibraryList = GenerateLibraryList(5, coefficients, A)
    
    
    
    
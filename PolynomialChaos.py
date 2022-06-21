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
from math import sqrt
from math import factorial

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
        numberOfSamples = distribution_1D.shape[0]
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
    
    def aPC_OneDimensional(self, distribution_1D, threshold = 0.0, normalize = True):
        '''
        Computes and returns the coefficient matrix for a 1D distribution from the 0-degree
        polynomial up to the one of degree expansionDegree.
        '''
        d = self.expansionDegree + 1
        coefficients = np.zeros((d,d))
        coefficients[0,0] = 1
        for i in range(1,d):
            H = self.MomentMatrix(distribution_1D,i)
            v = np.zeros(i+1)
            v[-1] = 1
            coeff = np.linalg.solve(H,v)
            if not normalize: 
                coeff= self._sparsePolynomials(coeff, threshold)
                coefficients[0:i+1,i] = coeff
            else:
                norm = self.PolynomialNorm(coeff, distribution_1D)
                threshold = norm * threshold
                coeff= self._sparsePolynomials(coeff, threshold)
                coefficients[0:i+1,i] = coeff / sqrt(np.abs(norm)) # the abs is necessary since very small norms can be computed as negative
        # coefficients = np.reshape(coefficients,(d,d,1))
        return coefficients
    
    def _sparsePolynomials(self, coeff, threshold):
        '''
        Retain only those terms that are bigger then a threshold
        '''
        c = coeff
        big_ind = np.abs(c) >= threshold
        c[~big_ind] = 0
        return c
    
    def PolynomialNorm(self, coefficients, distribution_1D):
        '''
        Compute the l-2 norm of a polynomial given the distribution of the inputs 
        and the array of coefficients
        '''
        coeff = np.outer(coefficients, coefficients)
        moments = self.ComputeMoments(distribution_1D)
        for i in range(len(coefficients)):
            for j in range(len(coefficients)):
                coeff[i,j] = coeff[i,j] * moments[i+j]
        return np.sum(coeff)

    def ComputeCoefficients(self, threshold = 0.0, normalize = True):
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
            self.coefficients = np.reshape(self.aPC_OneDimensional(self.distribution,
                                                                   threshold = threshold,
                                                                   normalize = normalize), (d,d,1))
            
            self.AlphaMatrix = np.array([range(d)]).T
        else:
            self.coefficients = np.zeros((d,d, self.numberOfInputs))
            for i in range(self.numberOfInputs):
                self.coefficients[:,:,i] = self.aPC_OneDimensional(self.distribution[:,i],
                                                                   threshold = threshold,
                                                                   normalize = normalize)
                

            self.AlphaMatrix = MultivariatePolynomialIndex(self.numberOfInputs, d-1)
            
    def _get_feature_names(self, namelist = None):
        N = self.AlphaMatrix.shape[1]
        
        if namelist == None: # generate default names
            namelist = [f'x{i}' for i in range(N)]
        self.feature_names = []    
        num = 0
        for row in self.AlphaMatrix:
            inds = np.where(row)[0]
            if len(inds):
                coeff = self.coefficients[:,row[inds], inds]
                # print('----- feature: ------')
                # print(coeff)
                name = f"f{num}: "
                for i in range(len(inds)):
                    varname = namelist[inds[i]]
                    exps = np.where(coeff[:,i])[0]
                    varcoeffs = coeff[exps,i]
                    name += '('
                    name += " + ".join('%0.3f%s^%d' % (coef, varname, exp)
                              if exp > 1
                              else '%0.3f%s' % (coef, varname) if exp == 1
                              else '%0.3f' % coef
                              for coef, exp in zip(varcoeffs, exps)
                              )
                    name += ')'
                    
            else:
                name = "f0: 1"
            self.feature_names.append(name)
            num += 1
        
        return self.feature_names
    
    def printFeatureNames(self, nameList = None):
        try:
            for name in self.feature_names:
                print(name)
        except:
            names = self._get_feature_names(nameList)
            for name in names:
                print(name)
            
        
            
def GenerateLibraryList(
        expansionDegree,
        coefficients,
        AlphaMatrix,
        intercept = True
        ):
    '''
    Given the Alpha matrix and the coefficient tensor conputes a list of functions
    ready to be transformed into a SINDy library.
    '''
    M , numberOfInputs = AlphaMatrix.shape # M = total number of terms in the expansion
    x = []
    for i in range(numberOfInputs): x.append(sym.symbols(f'x{i}')) # list of symbolic variables
    
    LibraryList = []
    if intercept:
        LibraryList.append(lambda *x: 1)
    
    for i in range(1, M): # order
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
    n = 10000
    data = np.zeros((n,3))
    data_uniform = np.linspace(-1,1,n)
    data_uniform = np.array([data_uniform])
    data[:,0] = data_uniform
    data[:,1] = np.random.randn(n)
    data[:,2] = np.random.randn(n)
    
    expansionDegree = 2
    numberOfInputs = 1
    
    aPC = PolynomialChaos(data_uniform.T, expansionDegree, numberOfInputs)
    aPC.ComputeCoefficients(threshold = 0.0, normalize=True)
    coefficients = aPC.coefficients
    A = aPC.AlphaMatrix
    # features_names = aPC._get_feature_names()
    aPC.printFeatureNames()
    
    LibraryList = GenerateLibraryList(expansionDegree, coefficients, A)
    
    
    
    
    
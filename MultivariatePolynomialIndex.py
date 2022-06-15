#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 14:55:25 2022

@author: Enrico Foglia

This code provide the combinatory information for multivariate polynomial chaos 
expansion. The code is the translation from Matlab to Python of the one from
Sergey Oladyshkin (http://www.iws.uni-stuttgart.de), original code can be found at:
Sergey Oladyshkin (2022). aPC Matlab Toolbox: Data-driven Arbitrary Polynomial Chaos 
(https://www.mathworks.com/matlabcentral/fileexchange/72014-apc-matlab-toolbox-data-driven-arbitrary-polynomial-chaos), 
MATLAB Central File Exchange. Retrieved June 12, 2022.

A difference in the construction of UniqueDegreeCombinations has as an effect 
that the SortedDegreeCombinations is a bit different form the original code. It
 is, however, only an "aesthetic" difference, and the resulting indices are just
 a slight recombination of the original ones.
 
"""

import numpy as np
from math import factorial

def MultivariatePolynomialIndex(
        N, # number of variables
        d  # degree of the polynomial expansion 
        ):
    M = int(factorial(N+d) / (factorial(N)*factorial(d))) # total number of inputs
    
    PossibleDegree = [list(range(d+1)) for i in range(N)]
    UniqueDegreeCombinations = np.array(np.meshgrid(*PossibleDegree))
    UniqueDegreeCombinations = UniqueDegreeCombinations.reshape((N,-1))
    UniqueDegreeCombinations = UniqueDegreeCombinations.T
    
    DegreeWeight = np.zeros(len(UniqueDegreeCombinations))
    for i in range(len(UniqueDegreeCombinations)):
        #DegreeWeight[i] = 0
        # for j in range(N):
        #     DegreeWeight[i] += UniqueDegreeCombinations[i,j]
        DegreeWeight[i] = sum(UniqueDegreeCombinations[i,:])
            
    ID = np.argsort(DegreeWeight)
    SortDegreeCombination = UniqueDegreeCombinations[ID,:]
    
    return SortDegreeCombination[0:M,:]

if __name__ == '__main__':
    N = 2
    d = 2
    AlphaMatrix = MultivariatePolynomialIndex(N,d)

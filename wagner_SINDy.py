'''
	Created on: 19/04/2022
	Author: Enrico Foglia

	First attempt at system discovery using PySINDy on the data generated with the function
	Wagner.py. Ideally the results should match those obtained by prof. S. Brunton in 
	[1] "Improved approximations to the Wagner function using sparse identification of nonlinear 
	dynamics", 2021.
'''

import numpy as np
import pysindy as ps
import matplotlib.pyplot as plt

def clean_data(data, tol = 1e-5):
    '''
    The functions cleans numerical errors in the integration, based on the hypothesis that
    these appear in the last phases of the transient phase, for t>>1, and that in this zone 
    the derivative is almost zero.

    TODO: should be refined, it's a bit crude at the moment

    INPUT:
    data: n x 2 np.array;
        The data to be cleaned, first column being time and the second L(t)
    tol: float;
        Tolerance on the value of the derivative.

    OUTPUT:
    data: n x 2 np.array;
            Cleaned data.
    '''
    dt = data[1,0] - data[0,0] # fixed timestep
    derivative = (data[1:,1] - data[0:-1,1]) / dt
    # Find end of first rising time
    rising_tol = 1e-5
    end_rise = 0
    while derivative[end_rise] > rising_tol:
        end_rise += 1
    
    outlier_index = np.where(np.abs(derivative) > tol)[0]
    outlier_index = outlier_index[np.where(outlier_index > end_rise)]
    # print(outlier_index)
    
    # find nearest non-corrupted data
    # Since numerical errors can come in adiacent points, it is important to find the 
    # nearest uncorrupted data

    diff = outlier_index[1:] - outlier_index[0:-1]
    # print(diff)
    def count_ones(diff):
        count = 1
        counts = []
        for i in range(len(diff)):
            if diff[i] == 1:
                count += 1
            else:
                counts.append(count)
                count = 1
        counts.append(count)
        return counts
    count = count_ones(diff)
    
    non_corrupted_index = np.zeros((len(count),2), dtype=np.int64)
    non_corrupted_index[0,0] = outlier_index[0]-1
    j = 0
    for i in range(len(diff)):
        if diff[i] != 1:
            non_corrupted_index[j,1] = outlier_index[i]+1
            j += 1
            non_corrupted_index[j,0] = outlier_index[i+1]-1
    non_corrupted_index[j,1] = outlier_index[-1]+1
    # print(non_corrupted_index)
    
    # compute the mean with nearest clean data

    mean_index = []
    for i in range(len(count)):
        for j in range(count[i]):
            mean_index.append(non_corrupted_index[i,:])
    # print(mean_index)
        
    
    for i in range(len(outlier_index)):
        outlier = outlier_index[i]
        indices = mean_index[i]
        lower_index = indices[0]
        upper_index = indices[1]
        data[outlier, 1] = (data[lower_index, 1] + data[upper_index, 1]) / 2
   
    # The plotting helps with evaluating the performance of the function
    fig, ax = plt.subplots()
    ax.plot(data[0:-1, 0], derivative)
    ax.set_title('Derivative')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('L\'(t)')
    ax.set_yscale('log')
    ax.grid(True)
    
    fig, ax = plt.subplots()
    ax.plot(data[:,0],1 - data[:,1])
    ax.set_xlabel('Time')
    ax.set_ylabel('1-L(t)')
    ax.set_title('Cleaned data')
    ax.set_yscale('log')
    ax.grid(True)

    plt.show()
    return data

# importing data
wagner_data = np.loadtxt('wagner_data.dat', dtype=float)
cleaned_wagner_data = clean_data(wagner_data)

t = cleaned_wagner_data[:,0] 
x = 1 - cleaned_wagner_data[:,1] # normalized, steady-state-subtracted lift (see reference [1])
x = np.array([x]).T

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()

for deg in range(2,9):
	# fitting model
	optimizer = ps.optimizers.stlsq.STLSQ(threshold = 0.1, alpha = 1e-05, max_iter = 10)
	library = ps.feature_library.polynomial_library.PolynomialLibrary(degree = deg)
	model = ps.SINDy(optimizer = optimizer, 
				     feature_library = library,
				     feature_names = ['phi']) # default paramaters:
					   						  # differentiation method: centered differences

	model.fit(x, t = t[1] - t[0])
	model.print()

	x0 = np.array([0.5])
	model_x = model.simulate(x0, t)

	err = np.abs(x - model_x) / x

	# plotting
	ax1.plot(t, model_x)
	ax1.set_xlabel('t [s]')
	ax1.set_ylabel(r'L(t) - $L_0$ / $L_0$')
	ax1.set_title('Fitted Unsteady Lift')
	ax1.set_xscale('log')
	ax1.set_yscale('log')
	ax1.grid(True)

	ax2.plot(t, err)
	ax2.set_xlabel('t [s]')
	ax2.set_ylabel(r'$\frac{L(t) - \hat{L}(t)}{1 - L(t)}$')
	ax2.set_yscale('log')
	ax2.set_title('Normalized Error')
	ax2.grid(True)

ax1.plot(t, x, 'k--')

ax1.legend(['r = 2','r = 3','r = 4','r = 5','r = 6','r = 7','r = 8', 'Analytical'])
ax2.legend(['r = 2','r = 3','r = 4','r = 5','r = 6','r = 7','r = 8'])

plt.show()

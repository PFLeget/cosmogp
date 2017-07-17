""" Test of differents implementations of GP."""

import cosmogp
import george
import sklearn.gaussian_process.kernels as skl_kernels
from sklearn.gaussian_process.gpr import GaussianProcessRegressor as skl_gpr
import numpy as np
import pylab as plt
import time



def generate_data_test(number_of_data,number_of_point,
                       kernel_amplitude=1.,correlation_length=1.,
                       white_noise=0,seed=1):
    """
    Generation of simulated data for GP.
    Generating random gaussian fields in 1 dimensions (y depend 
    only from the x axis). Used of Squared exponential kernel.
    """
    np.random.seed(seed)
    
    x_axis = np.linspace(0,5,number_of_points)

    x = np.zeros(number_of_data,number_of_point)
    y = np.zeros(number_of_data,number_of_point)

    kernel = cosmogp.rbf_kernel_1d(x, [kernel_amplitude,correlation_length],
                                   white_noise, floor=0.0, y_err=None)

    det_kernel = np.linalg.det(kernel)

    for i in range(number_of_data):
        y[i] = np.random.multivariate_normal(np.zeros(len(x_axis)), kernel)
        x[i] = x_axis

    return x,y,kernel,det_kernel


def cosmogp_gp(x,y,xnew,kernel='rbf1d',
               search_hyperparameter=False):

    timeA = time.time()
    gp = cosmogp.gaussian_process(all_y[0],all_grid[0]) 
    gp.find_hyperparameters()
    new_grid = np.linspace(-10,30,60) 
    gp.get_prediction(new_binning=new_grid)
    timeB = time.time()
    
    bp = cosmogp.build_pull([all_y[i]],[np.zeros_like(all_y[i])],
                            [grid],grid,np.zeros_like(grid),gp.hyperparameters)
    bp.compute_pull(diFF=None)
    
    return ynew,pull,std,time


def george_gp(x,y,xnew,kernel='rbf1d',
              search_hyperparameter=False):

    if kernel is 'rbf1d':
        from george.kernels import ExpSquaredKernel as kernel_george

    kernel = kernel_george(1.0)

    timeA = time.time()
    
    gp = george.GP(kernel)
    gp.compute(x, yerr)
    gp.lnlikelihood(y)

    t = np.linspace(0, 10, 500)
    mu, cov = gp.predict(y, t)

    timeB = time.time()
    
    return ynew,pull,std,time


def scikitlearn_gp(x,y,xnew,kernel='rbf1d',
                   search_hyperparameter=False):

    #TO DO 
    
    return ynew,pull,std,time



class test_gaussian_process:

    def __init__(self,kernel='rbf1d',search_hyperparameter=False):
        """
        Test to compare different Gaussian Process codes.
        Three codes are tested.
        - Scikit learn implementation of GP.
        - George GP code 
        - cosmogp code (done for SNIa intepolation of SED)

        kernel: string, type of kernel 

        search_hyperparameter: boolean, optimization or 
        not of hyperparameter. Per default, no optimization 
        are done. 
        """

        self.kernel = kernel
        self.search_hyperparameter = search_hyperparameter
        
        self.N = np.array([10,100,1000,10000])
        

    def run_test(self):
        
        print 'TO DO'


    def plot_test_result(self):

        print 'TO DO'





if __name__=='__main__':
            
    test_gp = test_gaussian_process(kernel='rbf1d',search_hyperparameter=False)
    test_gp.run_test()
    test_gp.plot_test_result()

        
        
    
    
        

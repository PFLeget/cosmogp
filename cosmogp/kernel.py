#############################################
#
# implementation of different kind of kernel 
#
#############################################


import numpy as N

def RBF_kernel_1D(x,hyperparameter,nugget,floor=0.00,y_err=None):

    """
    1D RBF kernel

    K(x_i,x_j) = sigma^2 exp(-0.5 ((x_i-x_j)/l)^2) 
               + (y_err[i]^2 + nugget^2 + floor^2) delta_ij

    
    x : 1D numpy array or 1D list. Grid of observation.
    For SNIa it would be observation phases. 

    hyperparameter : dic. dictionnary that have as key sigma 
    and l wich are the two parameters of th RBF kernel.
    hyperparameter.keys() should return {'sigma':float,
    'l': float}

    sigma : float. Kernel amplitude hyperparameter. 
    It explain the standard deviation from the mean function.

    l : float. Kernel correlation length hyperparameter. 
    It explain at wich scale one data point afect the 
    position of an other.

    nugget : float. Diagonal dispertion that you can add in 
    order to explain intrinsic variability not discribe by 
    the RBF kernel.

    floor : float. Diagonal error that you can add to your 
    RBF Kernel if you know the value of your intrinsic dispersion. 

    y_err : 1D numpy array or 1D list. Error from data 
    observation. For SNIa, it would be the error on the 
    observed flux/magnitude.


    output : Cov. 2D numpy array, shape = (len(x),len(x))
    """
    
    if y_err is None:
        y_err = N.zeros_like(x)
    
    Cov = N.zeros((len(x),len(x)))
    
    for i in range(len(x)):
        for j in range(len(x)):
            Cov[i,j] = (hyperparameter['sigma']**2)*N.exp(-0.5*((x[i]-x[j])/hyperparameter['l'])**2)
            
            if i==j:
                Cov[i,j] += y_err[i]**2+floor**2+nugget**2

    return Cov





        

    

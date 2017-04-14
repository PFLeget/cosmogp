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

    sigma = hyperparameter[0]
    l = hyperparameter[1]

    sigma -->  Kernel amplitude hyperparameter. 
    It explain the standard deviation from the mean function.

    l --> Kernel correlation length hyperparameter. 
    It explain at wich scale one data point afect the 
    position of an other.


    
    x : 1D numpy array or 1D list. Grid of observation.
    For SNIa it would be observation phases. 

    hyperparameter : 1D numpy array or 1D list. 

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
            Cov[i,j] = (hyperparameter[0]**2)*N.exp(-0.5*((x[i]-x[j])/hyperparameter[1])**2)
            
            if i==j:
                Cov[i,j] += y_err[i]**2+floor**2+nugget**2

    return Cov



def RBF_kernel_2D(x,hyperparameter,nugget,floor=0.00,y_err=None):

    """
    2D RBF kernel

    K(x_i,x_j) = sigma^2 exp(-0.5 (x_i-x_j)^t L (x_i-x_j)) 
               + (y_err[i]^2 + nugget^2 + floor^2) delta_ij

    sigma = hyperparameter[0]

    L = numpy.array(([hyperparameter[1]**2,hyperparameter[3]],
                   [hyperparameter[3],hyperparameter[2]**2]))
    
    sigma --> Kernel amplitude hyperparameter. 
    It explain the standard deviation from the mean function.

    L --> Kernel correlation length hyperparameter. 
    It explain at wich scale one data point afect the 
    position of an other.

    x : 2D numpy array or 2D list. Grid of coordinate.
    For WL it would be pixel coordinate. 

    hyperparameter : 1D numpy array or 1D list. 

    nugget : float. Diagonal dispertion that you can add in 
    order to explain intrinsic variability not discribe by 
    the RBF kernel.

    floor : float. Diagonal error that you can add to your 
    RBF Kernel if you know the value of your intrinsic dispersion. 

    y_err : 1D numpy array or 1D list. Error from data 
    observation. For WL, it would be the error on the 
    parameter that you want to interpolate in the focal plane.


    output : Cov. 2D numpy array, shape = (len(x),len(x))
    """
    
    if y_err is None:
        y_err = N.zeros(len(x))
    
    Cov = N.zeros((len(x),len(x)))
    L=N.array(([hyperparameter[1]**2,hyperparameter[3]],
               [hyperparameter[3],hyperparameter[2]**2]))
    
    for i in range(len(x)):
        for j in range(len(x)):
            delta_x = x[i]-x[j]
            delta_x_t = delta_x.reshape((len(delta_x),1))
            Cov[i,j] = (hyperparameter[0]**2)*N.exp(-0.5*delta_x.dot(N.dot(L,delta_x_t)))
            
            if i==j:
                Cov[i,j] += y_err[i]**2+floor**2+nugget**2

    return Cov






        

    

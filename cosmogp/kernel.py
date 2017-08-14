"""implementation of different kind of kernel."""

import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform

def init_rbf(x,y):

    number_point = np.zeros(len(y))
    L_min = np.zeros(len(y))
    L_max = np.zeros(len(y))
    sigma = np.zeros(len(y))

    for i in range(len(y)):
        L_min = np.min(x[i])
        L_max = np.max(x[i])
        number_point[i] = len(x[i])
        sigma = np.std(y[i])

    d = np.mean(np.sqrt((L_max-L_min)**2/number_point))
    L = np.mean(L_max-L_min)

    return np.mean(sigma), np.mean([d,L])


def rbf_kernel_1d(x, hyperparameter, new_x=None,
                  nugget=0., floor=0.00, y_err=None):
    """
    1D RBF kernel.

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
        y_err = np.zeros_like(x)

    if new_x is None:
        x2 = x
        add_error = True 
    else:
        x2 = new_x
        add_error = False

    A = x - x2[:,None]
    cov = (hyperparameter[0]**2) * np.exp(-0.5 * ((A * A) / (hyperparameter[1]**2)))
    
    if add_error:
        cov += (np.eye(len(y_err)) * (y_err**2 + floor**2 + nugget**2))

    return cov


def rbf_kernel_2d(x, hyperparameter, new_x=None,
                  nugget=0., floor=0.00, y_err=None):
    """
    2D RBF kernel.

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
        y_err = np.zeros(len(x))

#    if new_x is None:

#    else:


    inv_l = np.array(([hyperparameter[2]**2,-hyperparameter[3]],
                     [-hyperparameter[3],hyperparameter[1]**2]))
    
    inv_l *= 1./(((hyperparameter[1]**2)*(hyperparameter[2]**2))-hyperparameter[3]**2)
    
    #A = (x[:,0]-x2[:,0][:,None])
    #B = (x[:,1]-x2[:,1][:,None])

    #cov = (hyperparameter[0]**2)*np.exp(-0.5*(A * A * inv_l[0,0] + 2. * A * B * inv_l[0,1] + B * B * inv_l[1,1]))

    if new_x is not None:
        print 'pouet'
        add_error = False
        dist = cdist(x, new_x, metric='mahalanobis', VI=inv_l)
        cov = (hyperparameter[0]**2)*np.exp(-0.5*dist**2)
        cov = cov.T
    else:
        add_error = True
        dists = pdist(x, metric='mahalanobis', VI=inv_l)
        cov = np.exp(-0.5 * dists**2)
        cov = squareform(cov)
        np.fill_diagonal(cov, 1)
    
    if add_error:
        cov += (np.eye(len(y_err))*(y_err**2+floor**2+nugget**2))

    print np.sum(cov)

    return cov





        

    

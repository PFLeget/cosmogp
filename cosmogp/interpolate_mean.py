"""cubic spline interpolation for average function."""


import scipy.interpolate as inter


def interpolate_mean_1d(old_binning,mean_function,new_binning):    
    """
    Function to interpolate 1D mean function on the new grid 
    Interpolation is done using cubic spline from scipy 

    old_binning : 1D numpy array or 1D list. Represent the 
    binning of the mean function on the original grid. Should not 
    be sparce. For SNIa it would be the phases of the Mean function 

    mean_function : 1D numpy array or 1D list. The mean function 
    used inside Gaussian Process, observed at the Old binning. Would 
    be the average Light curve for SNIa. 

    new_binning : 1D numpy array or 1D list. The new grid where you 
    want to project your mean function. For example, it will be the 
    observed SNIa phases.

    output : mean_interpolate,  Mean function on the new grid (New_binning)

    """

    cubic_spline = inter.InterpolatedUnivariateSpline(old_binning,mean_function)
      
    mean_interpolate = cubic_spline(new_binning)

    return mean_interpolate


def interpolate_mean_2d(old_binning,mean_function,new_binning):    
    """
    Function to interpolate 2D mean function on the new grid 
    Interpolation is done using cubic spline from scipy. 

    old_binning : 2D numpy array or 2D list. Represent the 
    binning of the mean function on the original grid. Should not 
    be sparce. For Weak-lensing it would be the pixel coordinqtes 
    for the Mean function.

    mean_function : 2D numpy array or 2D list. The mean function 
    used inside Gaussian Process, observed at the Old binning. Would 
    be the average value of PSF size for example in Weak-lensing. 

    new_binning : 2D numpy array or 2D list. The new grid where you 
    want to project your mean function. For example, it will be the 
    galaxy position for Weak-Lensing.

    output : mean_interpolate,  Mean function on the new grid (New_binning)

    """
    

    tck = inter.bisplrep(old_binning[:,0],old_binning[:,1],mean_function,task=1)
    
    mean_interpolate = N.zeros(len(new_binning))
    
    for i in range(len(new_binning)):
        mean_interpolate[i] = interpolate.bisplev(old_binning[i,0],old_binning[i,1],tck)
                                            

    return mean_interpolate 

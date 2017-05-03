"""implementation of different kind of kernel."""

import numpy as N

#TO DO : each times I should implement the H matrix which will be the
# the covariance matrix and the new binning (to implement in 1D and 2D)

def rbf_kernel_1d(x, hyperparameter, nugget, floor=0.00, y_err=None,SPEED=False):
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
        y_err = N.zeros_like(x)

    if not SPEED :
        Cov = N.zeros((len(x),len(x)))
        
        for i in range(len(x)):
            for j in range(len(x)):
                Cov[i,j] = (hyperparameter[0]**2)*N.exp(-0.5*((x[i]-x[j])/hyperparameter[1])**2)
            
                if i==j:
                    Cov[i,j] += y_err[i]**2+floor**2+nugget**2
    else:
        A = x-x[:,None]

        Cov = (hyperparameter[0]**2)*N.exp(-0.5*((A*A)/(hyperparameter[1]**2)))
        Cov += (N.eye(len(y_err))*(y_err**2+floor**2+nugget**2))
        
    return Cov


def compute_rbf_1d_ht_matrix(new_x,old_x,hyperparameters,as_the_same_grid=False,SPEED=False):
    
    HT=[]

    if not SPEED:
        for sn in range(len(old_x)): 
            if as_the_same_grid:
                New_binning=new_x[sn]
            else:
                New_binning=new_x
            
            HT.append(N.zeros((len(New_binning),len(old_x[sn]))))
            for i in range(len(New_binning)):
                for j in range(len(old_x[sn])):
                    HT[sn][i,j]=(hyperparameters[0]**2)*N.exp(-0.5*((New_binning[i]-old_x[sn][j])/hyperparameters[1])**2)

    else:
        for sn in range(len(old_x)): 
            if as_the_same_grid:
                New_binning=new_x[sn]
            else:
                New_binning=new_x

            #A = New_binning-old_x[sn][:,None]
            A = old_x[sn]-New_binning[:,None]
            HT.append((hyperparameters[0]**2)*N.exp(-0.5*((A*A)/(hyperparameters[1]**2))))
        

    return HT


def rbf_kernel_2d(x,hyperparameter,nugget,floor=0.00,y_err=None,SPEED=False):
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
        y_err = N.zeros(len(x))

    if not SPEED:
        Cov = N.zeros((len(x),len(x)))
        L=N.array(([hyperparameter[1]**2,hyperparameter[3]],
                   [hyperparameter[3],hyperparameter[2]**2]))
    
        Inv_L=N.linalg.inv(L)

        for i in range(len(x)):
            for j in range(len(x)):
                delta_x = x[i]-x[j]
                delta_x_t = delta_x.reshape((len(delta_x),1))
                Cov[i,j] = (hyperparameter[0]**2)*N.exp(-0.5*delta_x.dot(N.dot(Inv_L,delta_x_t)))
                    
                if i==j:
                    Cov[i,j] += y_err[i]**2+floor**2+nugget**2

    else:
        Inv_L=N.array(([hyperparameter[2]**2,-hyperparameter[3]],
                       [-hyperparameter[3],hyperparameter[1]**2]))
    
        Inv_L*=1./(((hyperparameter[1]**2)*(hyperparameter[2]**2))-hyperparameter[3]**2)
    
        A = (x[:,0]-x[:,0][:,None])
        B = (x[:,1]-x[:,1][:,None])

        Cov = (hyperparameter[0]**2)*N.exp(-0.5*(A*A*Inv_L[0,0] + 2.*A*B*Inv_L[0,1] + B*B*Inv_L[1,1]))
        Cov += (N.eye(len(y_err))*(y_err**2+floor**2+nugget**2))
    
    return Cov


def compute_rbf_2d_ht_matrix(new_x,old_x,hyperparameters,as_the_same_grid=False,SPEED=False):

    HT=[]
    if not SPEED:

        L=N.array(([hyperparameters[1]**2,hyperparameters[3]],
                   [hyperparameters[3],hyperparameters[2]**2]))

        Inv_L=N.linalg.inv(L)
    
        for sn in range(len(old_x)): 
            if as_the_same_grid:
                New_binning=new_x[sn]
            else:
                New_binning=new_x

            HT.append(N.zeros((len(New_binning),len(old_x[sn]))))
            for i in range(len(New_binning)):
                for j in range(len(old_x[sn])):
                    delta_x = New_binning[i]-old_x[sn][j]
                    delta_x_t = delta_x.reshape((len(delta_x),1))
                    HT[sn][i,j]=(hyperparameters[0]**2)*N.exp(-0.5*delta_x.dot(N.dot(Inv_L,delta_x_t)))
                
    else:
        for sn in range(len(old_x)): 
            if as_the_same_grid:
                New_binning = new_x[sn]
            else:
                New_binning = new_x
        
            Inv_L = N.array(([hyperparameters[2]**2,-hyperparameters[3]],
                             [-hyperparameters[3],hyperparameters[1]**2]))
    
            Inv_L *= 1./(((hyperparameters[1]**2)*(hyperparameters[2]**2))-hyperparameters[3]**2)
    
            A = (old_x[sn][:,0]-New_binning[:,0][:,None])
            B = (old_x[sn][:,1]-New_binning[:,1][:,None])

            HT.append((hyperparameters[0]**2)*N.exp(-0.5*(A*A*Inv_L[0,0] + 2.*A*B*Inv_L[0,1] + B*B*Inv_L[1,1])))

    


    return HT





        

    

""" Compute the pull and the residual of GP."""

import numpy as np
import cosmogp
from scipy.stats import norm as normal


class build_pull:

    def __init__(self,y,x,hyperparameters,
                 nugget=0.,y_err=None,y_mean=None,
                 x_axis_mean=None,kernel='RBF1D'):

        """
        Build the pull of GP interpolation. 
        
        Basically it will return you the pull and 
        the residual for a given kernel and a given 
        set of hyperparameter(s). 
        """

        self.y = y
        self.x = x
        self.hyperparameters = hyperparameters

        self.y_err = y_err
        self.y_mean = y_mean
        self.x_axis_mean = x_axis_mean

        self.kernel = kernel
        self.nugget = nugget
        
        self.n_object = len(y)

        self._pull = []
        self.pull = []
        self.residual = []

        self.pull_average = None
        self.pull_std = None

        
    def compute_pull(self,diff=None,svd=True,
                     substract_mean=False):

        """
        Function that compute the pull.

        Will return pull and residual
        """
        
        for sn in range(self.n_object):

            pred = np.zeros(len(self.y[sn]))
            pred_var = np.zeros(len(self.y[sn]))
            
            for t in range(len(self.y[sn])):
                
                FILTRE = np.array([True]*len(self.y[sn]))
                FILTRE[t] = False
                
                if self.y_err is None:
                    yerr = np.zeros_like(self.y[sn])
                else:
                    yerr = self.y_err[sn]
                    
                gpp = cosmogp.gaussian_process(self.y[sn][FILTRE],
                                               self.x[sn][FILTRE],
                                               y_err=yerr[FILTRE],
                                               Mean_Y=self.y_mean,
                                               Time_mean=self.x_axis_mean,
                                               kernel=self.kernel)
                
                if diff or substract_mean:
                    if diff is None:
                        gpp.substract_Mean(diff=None)
                    else:
                        gpp.substract_Mean(diff=[diff[sn]])
                                
                gpp.hyperparameters = self.hyperparameters
                gpp.nugget = self.nugget
                gpp.get_prediction(new_binning=self.x[sn],SVD=svd)
                
                pred[t] = gpp.Prediction[0][t]
                pred_var[t] = abs(gpp.covariance_matrix[0][t,t])

            self.gpp = gpp
            
            self.pred_var = pred_var
            pull = (pred-self.y[sn])/np.sqrt(yerr**2+pred_var+self.nugget**2)
            res = pred-self.y[sn]
            self._pull.append(pull)
            
            for t in range(len(self.y[sn])):
                self.pull.append(pull[t])
                self.residual.append(res[t])

        self.pull_average,self.pull_std = normal.fit(self.pull)


    def plot_result(self,binning=60):

        """
        Plot the pull distribution. 

        Basically it will present the 
        fit of the pull distribution as 
        a normal law. 

        binning: int or 1d numpy array. The
        same parameter as in histogramme function
        of matplotlib. 
        """

        import pylab as plt
        
        plt.figure()
        
        plt.hist(self.pull,bins=binning,normed=True)

        xmin, xmax = plt.xlim()
        _max = max([abs(xmin),abs(xmax)])
        plt.xlim(-_max,_max)
        xmin, xmax = plt.xlim()
        xaxis = np.linspace(xmin, xmax, 100)
        pdf = normal.pdf(xaxis, self.pull_average, self.pull_std)
        
        plt.plot(xaxis, pdf, 'r', linewidth=3)

        title = r"Fit results: $\mu$ = $%.2f\pm%.2f$, $\sigma$ = $%.2f\pm%.2f$" % (self.pull_average,
                                                                                   self.pull_std/np.sqrt(len(self.pull)),
                                                                                   self.pull_std,
                                                                                   self.pull_std/np.sqrt(2*len(self.pull)))
        
        plt.title(title)
        plt.ylabel('Number of points (normed)')
        plt.xlabel('Pull')


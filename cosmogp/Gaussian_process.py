"""gaussian process interpolator."""

import numpy as np
from scipy.optimize import fmin
import copy
try: from svd_tmv import computeSVDInverse as svd
except: from cosmogp import svd_inverse as svd
try: from svd_tmv import computeLDLInverse as chol
except: from cosmogp import cholesky_inverse as chol


def log_likelihood_gp(y, y_err, Mean_Y, Time, kernel,
                      hyperparameter, nugget,SVD=True):
    """
    Log likehood to maximize in order to find hyperparameter.

    The key point is that all matrix inversion are
    done by SVD decomposition + (if needed) pseudo-inverse.
    Slow but robust

    y : 1D numpy array or 1D list. Observed data at the
    observed grid (Time). For SNIa it would be light curve

    y_err : 1D numpy array or 1D list. Observed error from
    data on the observed grid (Time). For SNIa it would be
    error on data light curve points.

    Mean_Y : 1D numpy array or 1D list. Average function
    that train Gaussian Process. Should be on the same grid 
    as observation (y). For SNIa it would be average light
    curve.

    Time : 1D numpy array or 1D list. Grid of observation.
    For SNIa, it would be light curves phases.

    sigma : float. Kernel amplitude hyperparameter.
    It explain the standard deviation from the mean function.

    l : float. Kernel correlation length hyperparameter.
    It explain at wich scale one data point afect the
    position of an other.

    nugget : float. Diagonal dispertion that you can add in
    order to explain intrinsic variability not discribe by
    the RBF kernel.

    output : float. log_likelihood
    """

    NT = len(Time)
    kernel_matrix = kernel(Time,hyperparameter,nugget,y_err=y_err)
    y_ket = y.reshape(len(y),1)
    Mean_Y_ket = Mean_Y.reshape(len(Mean_Y),1)

    if SVD : #svd decomposition
        inv_kernel_matrix,log_det_kernel_matrix = svd(kernel_matrix,return_logdet=True)
    else : #cholesky decomposition
        inv_kernel_matrix,log_det_kernel_matrix = chol(kernel_matrix,return_logdet=True)

    log_likelihood = (-0.5*(np.dot((y-Mean_Y),np.dot(inv_kernel_matrix,(y_ket-Mean_Y_ket)))))

    log_likelihood += np.log((1./(2*np.pi)**(NT/2.)))
    log_likelihood -= 0.5*log_det_kernel_matrix

    return log_likelihood


class Gaussian_process:

    def __init__(self, y, Time, kernel='RBF1D',
                 y_err=None, Mean_Y=None, Time_mean=None):
        """
        Gaussian process interpolator.

        For a given data or a set of data (with associated error(s))
        and assuming a given average function of your data, this
        class will provide you the interpolation of your data on
        a new grid where your average funtion is difine. It provides
        also covariance matrix of the interpolation.
        
        y : list of numpy array. Data that you want to interpolate.
        Each numpy array represent one of your data that you want to
        interpolate. For SNIa it would represent different light curves
        observed at different phases

        y_err : list of numpy array with the same structure as y.
        Error of y.

        Time : list of numpy array with the same structure as y. Observation phase
        of y. Each numpy array could have different size, but should correspond to
        y. For SNIa it would represent the differents epoch observation of differents
        light curves.

        Time_mean : numpy array with same shape as Mean_Y. Grid of the
        choosen average function. Don't need to be similar as Time.

        Mean_Y : numpy array. Average function of your data. Not reasonable 
        choice of average function will provide bad result, because interpolation
        from Gaussian Process use the average function as a prior.

        example :

        gp = Gaussian_process(y,y_err,Time,Time_mean,Mean_Y)
        gp.find_hyperparameters(sigma_guess=0.5,l_guess=8.)
        gp.get_prediction(new_binning=np.linspace(-12,42,19))

        output :

        GP.Prediction --> interpolation on the new grid
        GP.covariance_matrix --> covariance matrix from interpoaltion on the
        on the new grid
        GP.hyperparameters --> Fitted hyperparameters


        optional :
        If you think that you have a systematic difference between your data
        and your data apply this function before to fit hyperparameter or
        interpolation. If you think to remove a global constant for each data, put it
        in the diff option

        gp.substract_Mean(diff=None)
        """

        kernel_choice = ['RBF1D', 'RBF2D']

        assert kernel in kernel_choice, '%s is not in implemented kernel' %(kernel)

        if kernel == 'RBF1D':
            from cosmogp import rbf_kernel_1d as kernel
            from cosmogp import interpolate_mean_1d as interpolate_mean
            from cosmogp import compute_rbf_1d_ht_matrix as compute_ht
            from cosmogp import init_rbf as init_hyperparam

            self.kernel = kernel
            self.compute_HT_matrix = compute_ht
            self.interpolate_mean = interpolate_mean
            sigma,L = init_hyperparam(Time,y)
            self.hyperparameters = np.array([sigma, L])

        if kernel == 'RBF2D':
            from cosmogp import rbf_kernel_2d as kernel
            from cosmogp import interpolate_mean_2d as interpolate_mean
            from cosmogp import compute_rbf_2d_ht_matrix as compute_ht
            from cosmogp import init_rbf as init_hyperparam

            self.kernel = kernel
            self.compute_HT_matrix = compute_ht
            self.interpolate_mean = interpolate_mean
            sigma,L = init_hyperparam(Time,y)
            self.hyperparameters = np.array([sigma, L, L, 0.])

        self.y = y
        self.N_sn = len(y)
        self.Time = Time
        self.nugget = 0.

        self.SUBSTRACT_MEAN = False
        self.CONTEUR_MEAN = 0
        self.diff = np.zeros(self.N_sn)

        if y_err is not None:
            self.y_err = y_err
        else:
            if len(self.y) == 1:
                self.y_err = [np.zeros(len(self.y[0]))]
            else:
                self.y_err = []
                for i in range(len(self.y)):
                    self.y_err.append(np.zeros_like(self.y[i]))

        if Mean_Y is not None:
            self.Mean_Y = Mean_Y
        else:
            self.Mean_Y = np.zeros_like(self.y[0])

        if Time_mean is not None:
            self.Time_mean = Time_mean
        else:
            self.Time_mean = self.Time[0]

        #self.substract_Mean()
        #self.CONTEUR_MEAN+=1


    def substract_Mean(self, diff=None):
        """
        Substract the mean function.
        in order to avoid systematic difference between
        average function and the data
        """

        self.SUBSTRACT_MEAN = True
        self.Mean_Y_in_BINNING_Y = []
        self.TRUE_mean = copy.deepcopy(self.Mean_Y)

        for sn in range(self.N_sn):
            
            MEAN_Y = self.interpolate_mean(self.Time_mean, self.Mean_Y, self.Time[sn])
            
            if diff is None:
                self.Mean_Y_in_BINNING_Y.append(MEAN_Y + np.mean(self.y[sn] - MEAN_Y))
                self.y[sn] -= (MEAN_Y + np.mean(self.y[sn] - MEAN_Y))
            else:
                self.diff = diff
                self.Mean_Y_in_BINNING_Y.append(MEAN_Y + diff[sn])
                self.y[sn] -= (MEAN_Y + diff[sn])

        self.Mean_Y=np.zeros(len(self.Time_mean))


    def compute_log_likelihood(self, Hyperparameter, svd_log=True):
        """
        Function to compute the log likelihood.
        compute the global likelihood for all your data
        for a set of hyperparameters
        """

        if self.fit_nugget:
            Nugget = Hyperparameter[-1]
            hyperparameter = Hyperparameter[:-1]
        else:
            Nugget = 0
            hyperparameter = Hyperparameter

        log_likelihood = 0

        for sn in range(self.N_sn):
            Mean_Y = self.interpolate_mean(self.Time_mean, self.Mean_Y, self.Time[sn])
            log_likelihood += log_likelihood_gp(self.y[sn], self.y_err[sn], Mean_Y,
                                                self.Time[sn], self.kernel,
                                                hyperparameter, Nugget, SVD=svd_log)

        self.log_likelihood = log_likelihood


    def find_hyperparameters(self, hyperparameter_guess=None, nugget=False, SVD=True):
        """
        Search hyperparameter using a maximum likelihood.
        Maximize with optimize.fmin for the moment
        """

        if hyperparameter_guess is not None :
            assert len(self.hyperparameters) == len(hyperparameter_guess), 'should be same len'
            self.hyperparameters = hyperparameter_guess

        def _compute_log_likelihood(Hyper, svd_log=SVD):
            """
            Likelihood computation.
            Used for minimization
            """

            self.compute_log_likelihood(Hyper, svd_log=svd_log)

            return -self.log_likelihood[0]

        initial_guess = []

        for i in range(len(self.hyperparameters)):
            initial_guess.append(self.hyperparameters[i])

        if nugget:
            self.fit_nugget = True
            initial_guess.append(1.)
        else:
            self.fit_nugget = False

        hyperparameters = fmin(_compute_log_likelihood, initial_guess, disp=False)

        for i in range(len(self.hyperparameters)):
                self.hyperparameters[i] = np.sqrt(hyperparameters[i]**2)

        if self.fit_nugget:
            self.nugget = np.sqrt(hyperparameters[-1]**2)


    def compute_kernel_matrix(self):
        """
        Compute kernel.
        Compute the kernel function 
        """

        self.kernel_matrix = []

        for sn in range(self.N_sn):

            self.kernel_matrix.append(self.kernel(self.Time[sn], self.hyperparameters,
                                      self.nugget, y_err=self.y_err[sn]))

            
    def get_prediction(self, new_binning=None, COV=True, SVD=True):
        """
        Compute your interpolation.

        new_binning : numpy array Default = None. It will
        provide you the interpolation on the same grid as
        the data. Useful to compute pull distribution.
        Store with a new grid in order to get interpolation
        ouside the old grid. Will be the same for all the data

        COV : Boolean, Default = True. Return covariance matrix of
        interpolation.
        """

        if self.SUBSTRACT_MEAN and self.CONTEUR_MEAN != 0:

            self.substract_Mean(diff=self.diff)
            self.CONTEUR_MEAN = 0

        if new_binning is None :
            self.as_the_same_time = True
            self.new_binning = self.Time
        else:
            self.as_the_same_time = False
            self.new_binning = new_binning
            
        self.compute_kernel_matrix()
        self.HT = self.compute_HT_matrix(self.new_binning, self.Time,
                                         self.hyperparameters,
                                         as_the_same_grid=self.as_the_same_time)
        self.Prediction = []

        for i in range(self.N_sn):

            if not self.as_the_same_time:
                self.Prediction.append(np.zeros(len(self.new_binning)))
            else:
                self.Prediction.append(np.zeros(len(self.new_binning[i])))

        if not self.as_the_same_time:
            self.New_mean = self.interpolate_mean(self.Time_mean, self.Mean_Y,
                                                  self.new_binning)

        self.inv_kernel_matrix = []

        for sn in range(self.N_sn):
            
            if self.as_the_same_time:
                self.New_mean = self.interpolate_mean(self.Time_mean,self.Mean_Y,self.new_binning[sn])

            self.inv_kernel_matrix.append(np.zeros((len(self.Time[sn]), len(self.Time[sn]))))
            Mean_Y = self.interpolate_mean(self.Time_mean, self.Mean_Y, self.Time[sn])
            Y_ket = (self.y[sn] - Mean_Y).reshape(len(self.y[sn]), 1)

            if SVD: #SVD deconposition for kernel_matrix matrix
                inv_kernel_matrix = svd(self.kernel_matrix[sn])
            else: #choleski decomposition
                inv_kernel_matrix = chol(self.kernel_matrix[sn])

            self.inv_kernel_matrix[sn] = inv_kernel_matrix
            
            self.Prediction[sn] += (np.dot(self.HT[sn],np.dot(inv_kernel_matrix,Y_ket))).T[0]
            self.Prediction[sn] += self.New_mean

        if COV:
            self.get_covariance_matrix()

        if self.SUBSTRACT_MEAN and self.CONTEUR_MEAN == 0:

            for sn in range(self.N_sn):

                if self.as_the_same_time:
                    True_mean = self.interpolate_mean(self.Time_mean, self.TRUE_mean,
                                                      self.new_binning[sn])
                else:
                    True_mean = self.interpolate_mean(self.Time_mean, self.TRUE_mean,
                                                      self.new_binning)

                self.Prediction[sn] += True_mean + self.diff[sn]
                self.y[sn] += self.Mean_Y_in_BINNING_Y[sn]

            self.Mean_Y = copy.deepcopy(self.TRUE_mean)
            self.CONTEUR_MEAN += 1


    def get_covariance_matrix(self):
        """
        Compute error of interpolation.
        Will compute the error on the new grid
        and the covariance between it.
        """

        self.covariance_matrix = []

        for sn in range(self.N_sn):

            self.covariance_matrix.append(-np.dot(self.HT[sn], np.dot(self.inv_kernel_matrix[sn], self.HT[sn].T)))
            
            if self.as_the_same_time:
                self.covariance_matrix[sn] += self.kernel(self.new_binning[sn],
                                                          self.hyperparameters, 0)
            else:
                self.covariance_matrix[sn]+=self.kernel(self.new_binning,
                                                        self.hyperparameters, 0)


    def plot_prediction(self,sn, Error=False, TITLE=None,
                        y1_label='Y', y2_label='Y-<Y>', x_label='X'):
        """
        Plot result for a given data.
        """

        from matplotlib import pyplot as plt
        import matplotlib.gridspec as gridspec

        if not self.as_the_same_time:
            Time_predict = self.new_binning
        else:
            Time_predict = self.new_binning[sn]

        plt.figure(figsize=(8,8))

        gs = gridspec.GridSpec(2, 1,height_ratios=[2,1])
        plt.subplots_adjust(hspace = 0.01)

        plt.subplot(gs[0])
        CST_top = np.mean(self.Prediction[sn])

        Y_err = np.sqrt(np.diag(self.covariance_matrix[sn]))
        plt.scatter(self.Time[sn], self.y[sn]-CST_top, c='r', label='Data')
        plt.plot(Time_predict, self.Prediction[sn]-CST_top,'b',
                 label='Prediction')

        if Error:
            plt.errorbar(self.Time[sn], self.y[sn]-CST_top, linestyle='',
                         yerr=self.y_err[sn], ecolor='red', alpha=0.9,
                         marker='.', zorder=0)
            plt.fill_between(Time_predict,
                             self.Prediction[sn] - CST_top - Y_err,
                             self.Prediction[sn] - CST_top + Y_err,
                             color='b', alpha=0.7)

        plt.ylabel(y1_label)
        if TITLE:
            plt.title(TITLE)

        plt.legend()
        plt.xticks([],[])
        plt.ylim(np.min(self.Prediction[sn] - CST_top) - 1,
                 np.max(self.Prediction[sn] - CST_top) + 1)
        plt.xlim(np.min(self.Time[sn]), np.max(self.Time[sn]))
        plt.gca().invert_yaxis()

        plt.subplot(gs[1])

        Mean_Y = self.interpolate_mean(self.Time_mean,
                                       self.Mean_Y, self.Time[sn])
        Mean_Y_new_binning = self.interpolate_mean(self.Time_mean,
                                                   self.Mean_Y, Time_predict)
        CST_bottom = self.diff[sn]

        plt.scatter(self.Time[sn], self.y[sn] - Mean_Y - CST_bottom, c='r')
        plt.plot(Time_predict,
                 self.Prediction[sn] - Mean_Y_new_binning - CST_bottom, 'b')
        if Error:
            plt.errorbar(self.Time[sn], self.y[sn] - Mean_Y - CST_bottom,
                         linestyle='', yerr=self.y_err[sn], ecolor='red',
                         alpha=0.9, marker='.', zorder=0)
            plt.fill_between(Time_predict,
                             self.Prediction[sn] - Mean_Y_new_binning - Y_err - CST_bottom,
                             self.Prediction[sn] - Mean_Y_new_binning + Y_err - CST_bottom,
                             color='b', alpha=0.7)

        plt.plot(Time_predict, np.zeros(len(self.Prediction[sn])), 'k')

        plt.xlim(np.min(self.Time[sn]), np.max(self.Time[sn]))
        plt.ylim(np.min(self.Prediction[sn] - CST_bottom - Mean_Y_new_binning) - 0.5,
                 np.max(self.Prediction[sn] - CST_bottom - Mean_Y_new_binning) + 0.5)
        plt.ylabel(y2_label)
        plt.xlabel(x_label)

        
class gaussian_process(Gaussian_process):


    def __init__(self, y, Time, kernel='RBF1D',
                 y_err=None, Mean_Y=None, Time_mean=None):
        """
        Run gp for one object.
        """
        if y_err is not None:
            y_err = [y_err]

        Gaussian_process.__init__(self, [y], [Time], kernel=kernel,
                                  y_err=y_err, Mean_Y=Mean_Y, Time_mean=Time_mean)


class gaussian_process_nobject(Gaussian_process):


    def __init__(self,y, Time, kernel='RBF1D',
                 y_err=None, Mean_Y=None, Time_mean=None):
        """
        Run gp for n object. 
        """
        Gaussian_process.__init__(self, y, Time, kernel=kernel,
                                  y_err=y_err, Mean_Y=Mean_Y, Time_mean=Time_mean)




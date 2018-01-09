"""gaussian process interpolator."""

import numpy as np
from scipy.optimize import fmin
import copy
try: from svd_tmv import computeSVDInverse as svd
except: from cosmogp import svd_inverse as svd
try: from svd_tmv import computeLDLInverse as chol
except: from cosmogp import cholesky_inverse as chol
from cosmogp import return_mean


def log_likelihood_gp(y, x_axis, kernel, hyperparameter, nugget,
                      y_err=None, y_mean=None, svd_method=True):
    """
    Log likehood to maximize in order to find hyperparameter.

    The key point is that all matrix inversion are
    done by svd decomposition + (if needed) pseudo-inverse.
    Slow but robust

    y : 1D numpy array or 1D list. Observed data at the
    observed grid (x_axis). For SNIa it would be light curve

    x_axis : 1D numpy array or 1D list. Grid of observation.
    For SNIa, it would be light curves phases.

    y_err : 1D numpy array or 1D list. Observed error from
    data on the observed grid (x_axis). For SNIa it would be
    error on data light curve points.

    Mean_Y : 1D numpy array or 1D list. Average function
    that train Gaussian Process. Should be on the same grid
    as observation (y). For SNIa it would be average light
    curve.

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
    if y_mean is None:
        y_mean = np.zeros_like(y)

    number_points = len(x_axis)
    kernel_matrix = kernel(x_axis, hyperparameter, nugget=nugget, y_err=y_err)
    y_ket = y.reshape(len(y), 1)
    y_mean_colomn = np.ones(len(y))*y_mean
    y_mean_colomn = y_mean_colomn.reshape(len(y_mean_colomn), 1)

    if svd_method:  #svd decomposition
        inv_kernel_matrix, log_det_kernel_matrix = svd(kernel_matrix,
                                                       return_logdet=True)
    else:  #cholesky decomposition
        inv_kernel_matrix, log_det_kernel_matrix = chol(kernel_matrix,
                                                        return_logdet=True)
        #inv_kernel_matrix = np.linalg.inv(kernel_matrix)
        #log_det_kernel_matrix = np.sum(np.log(np.linalg.eigvals(kernel_matrix)))

    log_likelihood = (-0.5 * (np.dot((y - y_mean),
                                     np.dot(inv_kernel_matrix,
                                            (y_ket - y_mean_colomn)))))

    log_likelihood += -(number_points / 2.) * np.log((2 * np.pi))
    log_likelihood -= 0.5 * log_det_kernel_matrix

    return log_likelihood


class Gaussian_process:
    "Gaussian process regressor."

    def __init__(self, y, Time, kernel='RBF1D',
                 y_err=None, diff=None, Mean_Y=None,
                 Time_mean=None, substract_mean=False):
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
            from cosmogp import init_rbf as init_hyperparam

            self.kernel = kernel
            sigma, L = init_hyperparam(Time,y)
            self.hyperparameters = np.array([sigma, L])

        if kernel == 'RBF2D':
            from cosmogp import rbf_kernel_2d as kernel
            from cosmogp import init_rbf as init_hyperparam

            self.kernel = kernel
            sigma, L = init_hyperparam(Time,y)
            self.hyperparameters = np.array([sigma, L, L, 0.])

        self.y = y
        self.N_sn = len(y)
        self.Time = Time
        self.nugget = 0.

        if y_err is not None:
            self.y_err = y_err
        else:
            if len(self.y) == 1:
                self.y_err = [np.zeros(len(self.y[0]))]
            else:
                self.y_err = []
                for i in range(len(self.y)):
                    self.y_err.append(np.zeros_like(self.y[i]))

        self.Mean_Y = Mean_Y
        self.Time_mean = Time_mean
        self.substract_mean = substract_mean
        
        if diff is None:
            self.diff=np.array([None]*self.N_sn)
        else:
            self.diff = diff

        if self.substract_mean or self.Mean_Y is not None:
            self.y0 = []
            for i in range(self.N_sn):
                self.y0.append(return_mean(self.y[i], self.Time[i], mean_y=self.Mean_Y,
                                           mean_xaxis=self.Time_mean, diff=self.diff[i]))
        else:
            self.y0 = np.zeros(self.N_sn)

        self.as_the_same_time = True


    def compute_log_likelihood(self, Hyperparameter, svd_method=True):
        """
        Function to compute the log likelihood.
        compute the global likelihood for all your data
        for a set of hyperparameters
        """

        if self.fit_nugget:
            Nugget = Hyperparameter[-1]
            hyperparameter = Hyperparameter[:-1]
        else:
            Nugget = self.nugget
            hyperparameter = Hyperparameter

        log_likelihood = 0

        for sn in range(self.N_sn):

            log_likelihood += log_likelihood_gp(self.y[sn], self.Time[sn], self.kernel,
                                                hyperparameter, Nugget, y_err=self.y_err[sn],
                                                y_mean=self.y0[sn], svd_method=svd_method)

        self.log_likelihood = log_likelihood


    def find_hyperparameters(self, hyperparameter_guess=None, nugget=False, svd_method=True):
        """
        Search hyperparameter using a maximum likelihood.
        Maximize with optimize.fmin for the moment
        """

        if hyperparameter_guess is not None :
            assert len(self.hyperparameters) == len(hyperparameter_guess), 'should be same len'
            self.hyperparameters = hyperparameter_guess

        def _compute_log_likelihood(Hyper, svd_method=svd_method):
            """
            Likelihood computation.
            Used for minimization
            """

            self.compute_log_likelihood(Hyper, svd_method=svd_method)

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
                                      nugget=self.nugget, y_err=self.y_err[sn]))


    def get_prediction(self, new_binning=None, COV=True, svd_method=True):
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

        if new_binning is None :
            self.as_the_same_time = True
            self.new_binning = self.Time
        else:
            self.as_the_same_time = False
            self.new_binning = new_binning

        self.compute_kernel_matrix()

        self.Prediction = []

        for i in range(self.N_sn):

            if not self.as_the_same_time:
                self.Prediction.append(np.zeros(len(self.new_binning)))
            else:
                self.Prediction.append(np.zeros(len(self.new_binning[i])))

        self.inv_kernel_matrix = []

        for sn in range(self.N_sn):

            if self.substract_mean or self.Mean_Y is not None:
            
                if self.as_the_same_time:
                    new_y0 = self.y0[sn]
                else:
                    new_y0 = return_mean(self.y[sn], self.Time[sn], new_x=self.new_binning,
                                         mean_y=self.Mean_Y, mean_xaxis=self.Time_mean, diff=self.diff[sn])
            else:
                new_y0 = 0.
            self.warning_pf = new_y0
            self.inv_kernel_matrix.append(np.zeros((len(self.Time[sn]), len(self.Time[sn]))))
            Y_ket = (self.y[sn] - self.y0[sn]).reshape(len(self.y[sn]), 1)

            if svd_method: #SVD deconposition for kernel_matrix matrix
                inv_kernel_matrix = svd(self.kernel_matrix[sn])
            else: #choleski decomposition
                inv_kernel_matrix = chol(self.kernel_matrix[sn])
                #inv_kernel_matrix = np.linalg.inv(self.kernel_matrix[sn])

            self.inv_kernel_matrix[sn] = inv_kernel_matrix

            if not self.as_the_same_time:
                new_grid = self.new_binning
            else:
                new_grid = self.new_binning[i]
            
            H = self.kernel(self.Time[sn], self.hyperparameters, new_x=new_grid)

            self.Prediction[sn] += (np.dot(H,np.dot(inv_kernel_matrix,Y_ket))).T[0]
            self.Prediction[sn] += new_y0

        if COV:
            self.get_covariance_matrix()

    def get_covariance_matrix(self):
        """
        Compute error of interpolation.
        Will compute the error on the new grid
        and the covariance between it.
        """

        self.covariance_matrix = []

        for sn in range(self.N_sn):

            if self.as_the_same_time:
                new_grid = self.new_binning[sn]
            else:
                new_grid = self.new_binning

            H = self.kernel(self.Time[sn], self.hyperparameters, new_x=new_grid)
            K = self.kernel(new_grid, self.hyperparameters, nugget=self.nugget)
            
            self.covariance_matrix.append(-np.dot(H, np.dot(self.inv_kernel_matrix[sn], H.T)))

            self.covariance_matrix[sn] += K



class gaussian_process(Gaussian_process):


    def __init__(self,y, Time, kernel='RBF1D',
                 y_err=None, diff=None, Mean_Y=None,
                 Time_mean=None, substract_mean=False):
        """
        Run gp for one object.
        """
        if y_err is not None:
            y_err = [y_err]

        Gaussian_process.__init__(self, [y], [Time], kernel=kernel,
                                  y_err=y_err, Mean_Y=Mean_Y, Time_mean=Time_mean,
                                  diff=diff, substract_mean=substract_mean)


class gaussian_process_nobject(Gaussian_process):


    def __init__(self,y, Time, kernel='RBF1D',
                 y_err=None, diff=None, Mean_Y=None,
                 Time_mean=None, substract_mean=False):
        """
        Run gp for n object.
        """
        Gaussian_process.__init__(self, y, Time, kernel=kernel,
                                  y_err=y_err, diff=diff, Mean_Y=Mean_Y,
                                  Time_mean=Time_mean, substract_mean=substract_mean)




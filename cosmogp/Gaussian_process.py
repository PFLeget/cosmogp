"""gaussian process interpolator."""

import numpy as N
from scipy.optimize import fmin
import copy

try: from svd_tmv import computeSVDInverse as svd
except: from cosmogp import svd_inverse as svd

try: from svd_tmv import computeLDLInverse as chol
except: from cosmogp import cholesky_inverse as chol


def Log_Likelihood_GP(y, y_err, Mean_Y, Time, kernel, hyperparameter, nugget,SVD=True):
    
    """
    Log likehood to maximize in order to find hyperparameter.

    The key point is that all matrix inversion are 
    done by SVD decomposition + (if needed) pseudo-inverse.
    Slow but robust 

    y : 1D numpy array or 1D list. Observed data at the 
    observed grid (Time). For SNIa it would be light curve

    y_err : 1D numpy array or 1D list. Observed error fron 
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

    output : float. Log_Likelihood
    
    """

    NT = len(Time)
    K = kernel(Time,hyperparameter,nugget,y_err=y_err)
    y_ket = y.reshape(len(y),1)
    Mean_Y_ket = Mean_Y.reshape(len(Mean_Y),1)
    
    if SVD : #svd decomposition 
        inv_K,log_det_K = svd(K,return_logdet=True)
    else : #cholesky decomposition
        inv_K,log_det_K = chol(K,return_logdet=True)

    Log_Likelihood = (-0.5*(N.dot((y-Mean_Y),N.dot(inv_K,(y_ket-Mean_Y_ket)))))

    Log_Likelihood += N.log((1./(2*N.pi)**(NT/2.)))
    Log_Likelihood -= 0.5*log_det_K
    
    #if N.sum(Filtre)!=len(Filtre):
    #     Log_Likelihood -= 0.5*(len(Filtre)-N.sum(Filtre))*N.log(10**-15)
    
    return Log_Likelihood


class Gaussian_process:
    
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
    gp.get_prediction(new_binning=N.linspace(-12,42,19))

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

    def __init__(self,y,Time,kernel='RBF1D',y_err=None,Mean_Y=None,Time_mean=None):

        kernel_choice=['RBF1D','RBF2D']
        assert kernel in kernel_choice, '%s is not in implemented kernel' %(kernel)

        if kernel == 'RBF1D':
            from cosmogp import rbf_kernel_1d as kernel
            from cosmogp import interpolate_mean_1d as interpolate_mean
            from cosmogp import compute_rbf_1d_ht_matrix as compute_ht
            from cosmogp import init_rbf as init_hyperparam
            
            self.kernel=kernel
            self.compute_HT_matrix= compute_ht
            self.interpolate_mean = interpolate_mean
            sigma,L = init_hyperparam(Time,y)
            self.hyperparameters=N.array([sigma, L])

        if kernel == 'RBF2D':
            from cosmogp import rbf_kernel_2d as kernel
            from cosmogp import interpolate_mean_2d as interpolate_mean
            from cosmogp import compute_rbf_2d_ht_matrix as compute_ht
            from cosmogp import init_rbf as init_hyperparam
            
            self.kernel = kernel
            self.compute_HT_matrix = compute_ht
            self.interpolate_mean = interpolate_mean
            sigma,L = init_hyperparam(Time,y)
            self.hyperparameters = N.array([sigma, L, L, 0.])
            
        self.y=y
        self.N_sn=len(y)
        self.Time=Time
        self.nugget=0.

        
        self.SUBSTRACT_MEAN=False
        self.CONTEUR_MEAN=0
        self.diff=N.zeros(self.N_sn)

            
        if y_err is not None:
            self.y_err=y_err
        else:
            if len(self.y)==1:
                self.y_err=[N.zeros(len(self.y[0]))]
            else:
                self.y_err=[]
                for i in range(len(self.y)):
                    self.y_err.append(N.zeros_like(self.y[i]))

        if Mean_Y is not None:
            self.Mean_Y=Mean_Y
        else:
            self.Mean_Y=N.zeros_like(self.y[0])
                    
        if Time_mean is not None:
            self.Time_mean=Time_mean
        else:
            self.Time_mean=self.Time[0]

        #self.substract_Mean()
        #self.CONTEUR_MEAN+=1
            
    def substract_Mean(self,diff=None):
        """
        in order to avoid systematic difference between 
        average function and the data 

        """
        
        self.SUBSTRACT_MEAN=True
        self.Mean_Y_in_BINNING_Y=[]
        self.TRUE_mean=copy.deepcopy(self.Mean_Y)
        for sn in range(self.N_sn):
            MEAN_Y=self.interpolate_mean(self.Time_mean,self.Mean_Y,self.Time[sn])
            if diff is None:
                self.Mean_Y_in_BINNING_Y.append(MEAN_Y+N.mean(self.y[sn]-MEAN_Y))
                self.y[sn]-=(MEAN_Y+N.mean(self.y[sn]-MEAN_Y))
            else:
                self.diff=diff
                self.Mean_Y_in_BINNING_Y.append(MEAN_Y+diff[sn])
                self.y[sn]-=(MEAN_Y+diff[sn])
        self.Mean_Y=N.zeros(len(self.Time_mean))


    def compute_Log_Likelihood(self,Hyperparameter,svd_log=True):
        """
        compute the global likelihood for all your data 
        for a set of hyperparameters 
        """
        if self.fit_nugget:
            Nugget=Hyperparameter[-1]
            hyperparameter=Hyperparameter[:-1]
        else:
            Nugget=0
            hyperparameter=Hyperparameter
            
        Log_Likelihood=0
        for sn in range(self.N_sn):
            Mean_Y=self.interpolate_mean(self.Time_mean,self.Mean_Y,self.Time[sn])
            Log_Likelihood+=Log_Likelihood_GP(self.y[sn],self.y_err[sn],Mean_Y,self.Time[sn],self.kernel,hyperparameter,Nugget,SVD=svd_log)

        self.Log_Likelihood=Log_Likelihood



    def find_hyperparameters(self,hyperparameter_guess=None,nugget=False,SVD=True):
        """
        Search hyperparameter using a maximum likelihood.
        Maximize with optimize.fmin for the moment 

        """

        if hyperparameter_guess is not None :
            assert len(self.hyperparameters) == len(hyperparameter_guess), 'should be same len' 
            self.hyperparameters = hyperparameter_guess

        def _compute_Log_Likelihood(Hyper,svd_log=SVD):

            self.compute_Log_Likelihood(Hyper,svd_log=svd_log)
            
            return -self.Log_Likelihood[0]     

        initial_guess=[]
        for i in range(len(self.hyperparameters)):
            initial_guess.append(self.hyperparameters[i])
        if nugget:
            self.fit_nugget=True
            initial_guess.append(1.)
        else:
            self.fit_nugget=False
            
        hyperparameters=fmin(_compute_Log_Likelihood,initial_guess,disp=False)
        
        for i in range(len(self.hyperparameters)):
                self.hyperparameters[i]=N.sqrt(hyperparameters[i]**2)
        
        if self.fit_nugget:
            self.nugget=N.sqrt(hyperparameters[-1]**2)

            
    def compute_covariance_matrix_K(self):

        self.K=[]
        for sn in range(self.N_sn):
            self.K.append(self.kernel(self.Time[sn],self.hyperparameters,self.nugget,y_err=self.y_err[sn]))
        

    def get_prediction(self,new_binning=None,COV=True,SVD=True):
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
        if self.SUBSTRACT_MEAN and self.CONTEUR_MEAN!=0:

            self.substract_Mean(diff=self.diff)
            self.CONTEUR_MEAN=0

        if new_binning is None :
            self.as_the_same_time=True
            self.new_binning=self.Time
        else:
            self.as_the_same_time=False
            self.new_binning=new_binning
        self.compute_covariance_matrix_K()
        self.HT=self.compute_HT_matrix(self.new_binning,self.Time,
                                       self.hyperparameters,as_the_same_grid=self.as_the_same_time)
        self.Prediction=[]

        for i in range(self.N_sn):
            if not self.as_the_same_time:
                self.Prediction.append(N.zeros(len(self.new_binning)))
            else:
                self.Prediction.append(N.zeros(len(self.new_binning[i])))

        if not self.as_the_same_time:
            self.New_mean= self.interpolate_mean(self.Time_mean,self.Mean_Y,self.new_binning)

        self.inv_K=[]
        for sn in range(self.N_sn):
            if self.as_the_same_time:
                self.New_mean= self.interpolate_mean(self.Time_mean,self.Mean_Y,self.new_binning[sn])
            self.inv_K.append(N.zeros((len(self.Time[sn]),len(self.Time[sn]))))
            Mean_Y=self.interpolate_mean(self.Time_mean,self.Mean_Y,self.Time[sn])
            Y_ket=(self.y[sn]-Mean_Y).reshape(len(self.y[sn]),1)

            if SVD: #SVD deconposition for K matrix
                inv_K = svd(self.K[sn])
            else: #choleski decomposition
                inv_K = chol(self.K[sn])
                
            self.inv_K[sn]=inv_K
            
            self.Prediction[sn]+=(N.dot(self.HT[sn],N.dot(inv_K,Y_ket))).T[0]
            self.Prediction[sn]+=self.New_mean
            

        if COV:
            self.get_covariance_matrix()

        if self.SUBSTRACT_MEAN and self.CONTEUR_MEAN==0:

            for sn in range(self.N_sn):
                if self.as_the_same_time:
                    True_mean=self.interpolate_mean(self.Time_mean,self.TRUE_mean,self.new_binning[sn])
                else:
                    True_mean=self.interpolate_mean(self.Time_mean,self.TRUE_mean,self.new_binning)
                self.Prediction[sn]+=True_mean+self.diff[sn]
                self.y[sn]+=self.Mean_Y_in_BINNING_Y[sn]
            self.Mean_Y=copy.deepcopy(self.TRUE_mean)
            self.CONTEUR_MEAN+=1


    def get_covariance_matrix(self):
        
        self.covariance_matrix=[]

        for sn in range(self.N_sn):
            
            self.covariance_matrix.append(-N.dot(self.HT[sn],N.dot(self.inv_K[sn],self.HT[sn].T)))
            
            if self.as_the_same_time:
                self.covariance_matrix[sn]+=self.kernel(self.new_binning[sn],self.hyperparameters,0)
            else:
                self.covariance_matrix[sn]+=self.kernel(self.new_binning,self.hyperparameters,0)
            

    def plot_prediction(self,sn,Error=False,TITLE=None,y1_label='Y',y2_label='Y-<Y>',x_label='X'):

        from matplotlib import pyplot as P 
        import matplotlib.gridspec as gridspec

        if not self.as_the_same_time:
            Time_predict=self.new_binning
        else:
            Time_predict=self.new_binning[sn]

        P.figure(figsize=(8,8))

        gs = gridspec.GridSpec(2, 1,height_ratios=[2,1])
        P.subplots_adjust(hspace = 0.01)
        #P.subplots_adjust(hspace=0.001)
        P.subplot(gs[0])
        CST_top=N.mean(self.Prediction[sn])

        Y_err=N.sqrt(N.diag(self.covariance_matrix[sn]))
        P.scatter(self.Time[sn],self.y[sn]-CST_top,c='r',label='Data')
        P.plot(Time_predict,self.Prediction[sn]-CST_top,'b',label='Prediction')

        if Error:
            P.errorbar(self.Time[sn],self.y[sn]-CST_top, linestyle='', yerr=self.y_err[sn],ecolor='red',alpha=0.9,marker='.',zorder=0)
            P.fill_between(Time_predict,self.Prediction[sn]-CST_top-Y_err,self.Prediction[sn]-CST_top+Y_err,color='b',alpha=0.7 )

        P.ylabel(y1_label)
        if TITLE:
            P.title(TITLE)

        P.legend()
        P.xticks([-50,150],['toto','tata'])
        P.ylim(N.min(self.Prediction[sn]-CST_top)-1,N.max(self.Prediction[sn]-CST_top)+1)
        P.xlim(N.min(self.Time[sn]),N.max(self.Time[sn]))
        P.gca().invert_yaxis()

        P.subplot(gs[1])
        
        Mean_Y=self.interpolate_mean(self.Time_mean,self.Mean_Y,self.Time[sn])
        Mean_Y_new_binning=self.interpolate_mean(self.Time_mean,self.Mean_Y,Time_predict)
        CST_bottom=self.diff[sn]#N.mean(self.Prediction[sn]-Mean_Y_new_binning)

        P.scatter(self.Time[sn],self.y[sn]-Mean_Y-CST_bottom,c='r')
        P.plot(Time_predict,self.Prediction[sn]-Mean_Y_new_binning-CST_bottom,'b')
        if Error:
            P.errorbar(self.Time[sn],self.y[sn]-Mean_Y-CST_bottom, linestyle='', yerr=self.y_err[sn],ecolor='red',alpha=0.9,marker='.',zorder=0)
            P.fill_between(Time_predict,self.Prediction[sn]-Mean_Y_new_binning-Y_err-CST_bottom,self.Prediction[sn]-Mean_Y_new_binning+Y_err-CST_bottom,color='b',alpha=0.7 )

        P.plot(Time_predict,N.zeros(len(self.Prediction[sn])),'k')
        
        P.xlim(N.min(self.Time[sn]),N.max(self.Time[sn]))
        P.ylim(N.min(self.Prediction[sn]-CST_bottom-Mean_Y_new_binning)-0.5,N.max(self.Prediction[sn]-CST_bottom-Mean_Y_new_binning)+0.5)
        P.ylabel(y2_label)
        P.xlabel(x_label)

        
class gaussian_process(Gaussian_process):
    
    
    def __init__(self,y,Time,kernel='RBF1D',y_err=None,Mean_Y=None,Time_mean=None):

        if y_err is not None:
            y_err=[y_err]
        
        Gaussian_process.__init__(self,[y],[Time],kernel=kernel,y_err=y_err,Mean_Y=Mean_Y,Time_mean=Time_mean)


class gaussian_process_nobject(Gaussian_process):

    
    def __init__(self,y,Time,kernel='RBF1D',y_err=None,Mean_Y=None,Time_mean=None):

        Gaussian_process.__init__(self,y,Time,kernel=kernel,y_err=y_err,Mean_Y=Mean_Y,Time_mean=Time_mean)




import numpy as N
import scipy as S
import pylab as P
import matplotlib.gridspec as gridspec
import cPickle
import iminuit as minuit
from scipy import linalg  
import scipy.interpolate as inter
import copy
from scipy.stats import norm as NORMAL_LAW


def interpolate_mean(old_binning,mean_function,new_binning):
    
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



def RBF_kernel_1D(Time,sigma,l,nugget,floor=0.00,y_err=None):

    """
    1D RBF kernel

    K(t_i,t_j) = sigma^2 exp(-0.5 ((t_i-t_j)/l)^2) 
               + (y_err[i]^2 + nugget^2 + floor^2) delta_ij

    
    Time : 1D numpy array or 1D list. Grid of observation.
    For SNIa it would be observation phases. 

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


    output : Cov. 2D numpy array, shape = (len(Time),len(Time))

    """

    if y_err is None:
        y_err = N.zeros_like(Time)
    
    Cov = N.zeros((len(Time),len(Time)))
    
    for i in range(len(Time)):
        for j in range(len(Time)):
            Cov[i,j] = (sigma**2)*N.exp(-0.5*((Time[i]-Time[j])/l)**2)
            
            if i==j:
                Cov[i,j] += y_err[i]**2+floor**2+nugget**2

    return Cov



def Log_Likelihood_GP(y,y_err,Mean_Y,Time,sigma,l,nugget):

    """
    Log likehood to maximize in order to find hyperparameter
    with 1D RBF kernel. 

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
    K = RBF_kernel_1D(Time,sigma,l,nugget,y_err=y_err)
    y_ket = y.reshape(len(y),1)
    Mean_Y_ket = Mean_Y.reshape(len(Mean_Y),1)
    #SVD deconposition for K matrix
    U,s,V = N.linalg.svd(K)
    # Pseudo-inverse 
    Filtre = (s>10**-15)
    if N.sum(Filtre)!=len(Filtre):
         print 'Pseudo-inverse decomposition :', len(Filtre)-N.sum(Filtre)
    inv_K = N.dot(V.T[:,Filtre],N.dot(N.diag(1./s[Filtre]),U.T[Filtre]))
    Log_Likelihood = (-0.5*(N.dot((y-Mean_Y),N.dot(inv_K,(y_ket-Mean_Y_ket)))))

    Log_Likelihood += N.log((1./(2*N.pi)**(NT/2.)))
    Log_Likelihood -= 0.5*N.sum(N.log(s[Filtre]))
    if N.sum(Filtre)!=len(Filtre):
         Log_Likelihood -= 0.5*(len(Filtre)-N.sum(Filtre))*N.log(10**-15)
    
    return Log_Likelihood



class Gaussian_process:

    """
    
    Gaussian process interpolator 

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

    def __init__(self,y,y_err,Time,Time_mean,Mean_Y):

        self.y=y
        self.N_sn=len(y)
        self.y_err=y_err
        self.Time=Time
        self.Mean_Y=Mean_Y
        self.Time_mean=Time_mean
        self.SUBSTRACT_MEAN=False
        self.hyperparameters={}
        self.CONTEUR_MEAN=0

    def substract_Mean(self,diff=None):

        """
        in order to avoid systematic difference between 
        average function and the data 

        """
        
        self.SUBSTRACT_MEAN=True
        self.Mean_Y_in_BINNING_Y=[]
        self.TRUE_mean=copy.deepcopy(self.Mean_Y)
        for sn in range(self.N_sn):
            MEAN_Y=Interpolate_mean(self.Time_mean,self.Mean_Y,self.Time[sn])
            if diff is None:
                self.diff=N.zeros(self.N_sn)
                self.Mean_Y_in_BINNING_Y.append(MEAN_Y+N.mean(self.y[sn]-MEAN_Y))
                self.y[sn]-=(MEAN_Y+N.mean(self.y[sn]-MEAN_Y))
            else:
                self.diff=diff
                self.Mean_Y_in_BINNING_Y.append(MEAN_Y+diff[sn])
                self.y[sn]-=(MEAN_Y+diff[sn])
        self.Mean_Y=N.zeros(len(self.Time_mean))



    def compute_Log_Likelihood(self,sigma,l):

        """
        compute the global likelihood for all your data 
        for a set of hyperparameters 

        """

        Nugget=0
        Log_Likelihood=0
        for sn in range(self.N_sn):
            Mean_Y=Interpolate_mean(self.Time_mean,self.Mean_Y,self.Time[sn])
            Log_Likelihood+=Log_Likelihood_GP(self.y[sn],self.y_err[sn],Mean_Y,self.Time[sn],sigma,l,Nugget)
        print 'sigma : ', sigma, ' l: ', l, ' Log_like: ', Log_Likelihood[0]
        self.Log_Likelihood=Log_Likelihood
            

    def init_hyperparameter_sigma(self):

        """
        initialize first guess for hyperparameter sigma
        
        """
        
        self.sigma_init=0.
        W=0
        for sn in range(self.N_sn):
            Mean_Y=Interpolate_mean(self.Time_mean,self.Mean_Y,self.Time[sn])
            W+=len(self.Time[sn])
            self.sigma_init+=(1./len(self.Time[sn]))*N.sum((Mean_Y-self.y[sn])**2)

        self.sigma_init/=W
        self.sigma_init=N.sqrt(self.sigma_init)
         
         

    def init_hyperparameter_l(self):

        """
        initialize first guess for hyperparameter l
        
        """
        

        residuals=N.zeros((self.N_sn,len(self.Mean_Y)))
         
        for sn in range(self.N_sn):

            y_interpolate=Interpolate_mean(self.Time[sn],self.y[sn],self.Time_mean)
            residuals[sn]=y_interpolate-self.Mean_Y

        self.init_hyperparameter_sigma()     

        self.Cov_matrix=(1./self.N_sn)*N.dot(residuals.T,residuals)
        self.L_matrix_guess=N.zeros(N.shape(self.Cov_matrix))
        for i in range(len(self.Time_mean)):
            for j in range(len(self.Time_mean)):
                self.L_matrix_guess[i,j]=N.sqrt(0.5*((self.Time_mean[i]-self.Time_mean[j])**2/(abs(2.*N.log(self.sigma_init)-self.Cov_matrix[i,j]))))

        self.Filtre=(self.L_matrix_guess!=0.)

        self.l_init=N.mean(self.L_matrix_guess[self.Filtre])



    def find_hyperparameters(self,sigma_guess=None,l_guess=None):

        """
        Search hyperparameter using a maximum likelihood

        maximize with iminuit for the moment 

        sigma_guess : Default = None and will used a specific function 
        to find it. Could be initialize with a float if you have a good 
        expectation.  

        l_guess : Default = None and will used a specific function 
        to find it. Could be initialize with a float if you have a good 
        expectation.  

        """

        if sigma_guess is None :
             self.init_hyperparameter_sigma()
             sigma_guess=self.sigma_init

        if l_guess is None :
             self.init_hyperparameter_l()
             l_guess=self.l_init

        def _compute_Log_Likelihood(sigma,l):


            self.compute_Log_Likelihood(sigma,l)
            #print 'sigma :', sigma, 'l :', l ,' Log L:', -self.Log_Likelihood[0] 
            
            return -self.Log_Likelihood[0]     

        Find_hyper=minuit.Minuit(_compute_Log_Likelihood, sigma=sigma_guess,l=l_guess)
        
        Find_hyper.migrad()
        #Find_hyper.simplex()
        
        self.hyperparameters=Find_hyper.values
        self.hyperparameters_Covariance=Find_hyper.covariance
        self.hyperparameters['sigma']=N.sqrt(self.hyperparameters['sigma']**2)
        self.hyperparameters['l']=N.sqrt(self.hyperparameters['l']**2)



    def map_Log_Likelihood(self,window_sig=10.,window_l=10.):

        """
        plot the log likelihood nearby the solution in order to see 
        if you are in a global maximum or a local maximum.

        """

        self.find_hyperparameters()

        sig=self.hyperparameters['sigma']
        L_corr=self.hyperparameters['l']
        

        SIGMAA=N.linspace(sig-window_sig,sig+window_l,100)
        ll=N.linspace(L_corr-window_l,L_corr+window_l,100)

        SIGMA, l = N.meshgrid(SIGMAA,ll)
        self.Map_log_l=N.zeros((len(SIGMA),len(l)))

        for i in range(len(SIGMA)):
            for j in range(len(l)):
                self.compute_Log_Likelihood(SIGMA[i,j],l[i,j])
                self.Map_log_l[i,j]=self.Log_Likelihood[0]
                print ''
                print i , j
                print SIGMA[i,j],l[i,j]
                print self.Log_Likelihood[0]

        P.pcolor(SIGMA, l, self.Map_log_l)

          
    def compute_covariance_matrix_K(self):

        self.K=[]
        for sn in range(self.N_sn):
            self.K.append(RBF_kernel_1D(self.Time[sn],self.hyperparameters['sigma'],self.hyperparameters['l'],0.,y_err=self.y_err[sn]))
        
    def compute_HT_matrix(self,NEW_binning):
        
        self.HT=[]

        for sn in range(self.N_sn): 
            if self.as_the_same_time:
                New_binning=NEW_binning[sn]
            else:
                New_binning=NEW_binning

            self.HT.append(N.zeros((len(New_binning),len(self.Time[sn]))))
            for i in range(len(New_binning)):
                for j in range(len(self.Time[sn])):
                    self.HT[sn][i,j]=(self.hyperparameters['sigma']**2)*N.exp(-0.5*((New_binning[i]-self.Time[sn][j])/self.hyperparameters['l'])**2)
            

    def get_prediction(self,new_binning=None,COV=True):

        """

        Compute your interpolation

        new_binning : numpy array Default = None. It will 
        provide you the interpolation on the same grid as 
        the data. Useful to compute pull distribution. 
        Store with a new grid in order to get interpolation 
        ouside the old grid. Will be the same for all the data 

        COV : Boolean, Default = True. Return covariance matrix of 
        interpoaltion. 

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
        self.compute_HT_matrix(self.new_binning)
        #self.Prediction=N.zeros((self.N_sn,len(self.new_binning)))
        self.Prediction=[]

        for i in range(self.N_sn):
            if not self.as_the_same_time:
                self.Prediction.append(N.zeros(len(self.new_binning)))
            else:
                self.Prediction.append(N.zeros(len(self.new_binning[i])))

        if not self.as_the_same_time:
            self.New_mean= Interpolate_mean(self.Time_mean,self.Mean_Y,self.new_binning)

        self.inv_K=[]
        for sn in range(self.N_sn):
            if self.as_the_same_time:
                self.New_mean= Interpolate_mean(self.Time_mean,self.Mean_Y,self.new_binning[sn])
            self.inv_K.append(N.zeros((len(self.Time[sn]),len(self.Time[sn]))))
            Mean_Y=Interpolate_mean(self.Time_mean,self.Mean_Y,self.Time[sn])
            Y_ket=(self.y[sn]-Mean_Y).reshape(len(self.y[sn]),1)
            #SVD deconposition for K matrix
            U,s,V=N.linalg.svd(self.K[sn])
            # Pseudo-inverse 
            Filtre=(s>10**-15)
            #if N.sum(Filtre)!=len(Filtre):
            #     print 'ANDALOUSE :', len(Filtre)-N.sum(Filtre)
            inv_K=N.dot(V.T[:,Filtre],N.dot(N.diag(1./s[Filtre]),U.T[Filtre]))
            self.inv_K[sn]=inv_K
            
            self.Prediction[sn]+=(N.dot(self.HT[sn],N.dot(inv_K,Y_ket))).T[0]
            self.Prediction[sn]+=self.New_mean
            

        if COV:
            self.get_covariance_matrix()

        if self.SUBSTRACT_MEAN and self.CONTEUR_MEAN==0:
            for sn in range(self.N_sn):
                if self.as_the_same_time:
                    True_mean=Interpolate_mean(self.Time_mean,self.TRUE_mean,self.new_binning[sn])
                else:
                    True_mean=Interpolate_mean(self.Time_mean,self.TRUE_mean,self.new_binning)
                self.Prediction[sn]+=True_mean+self.diff[sn]
                self.y[sn]+=self.Mean_Y_in_BINNING_Y[sn]
            self.Mean_Y=copy.deepcopy(self.TRUE_mean)
            self.CONTEUR_MEAN+=1


    def get_covariance_matrix(self,LINEAR_TRANSFORM=False):
        
        self.covariance_matrix=[]
        self.Linear_covariance=[]

        for sn in range(self.N_sn):

            if LINEAR_TRANSFORM:
                A=self.HT[sn].dot(self.inv_K[sn])
                self.Linear_covariance.append(A.dot(N.dot(N.diag(self.y_err[sn]),A.T)))
                
            #else:
            self.covariance_matrix.append(-N.dot(self.HT[sn],N.dot(self.inv_K[sn],self.HT[sn].T)))
            if self.as_the_same_time:
                self.covariance_matrix[sn]+=RBF_kernel_1D(self.new_binning[sn],self.hyperparameters['sigma'],self.hyperparameters['l'],0)
            else:
                self.covariance_matrix[sn]+=RBF_kernel_1D(self.new_binning,self.hyperparameters['sigma'],self.hyperparameters['l'],0)

            
    def get_pull(self,PLOT=True):
        self.get_prediction(new_binning=None)
        self.pull=[]
        self.PULL=[]
        for sn in range(self.N_sn):
            pull=(self.Prediction[sn]-self.y[sn])/N.sqrt(self.y_err[sn]**2+N.diag(self.covariance_matrix[sn]))
            self.pull.append(pull)
            for t in range(len(pull)):
                self.PULL.append(pull[t])
        self.Moyenne_pull,self.ecart_type_pull=NORMAL_LAW.fit(self.PULL)
        if PLOT:
            P.hist(self.PULL,bins=60,normed=True)
            xmin, xmax = P.xlim() 
            X = N.linspace(xmin, xmax, 100) 
            PDF = NORMAL_LAW.pdf(X, Moyenne, ecart_type) 
            P.plot(X, PDF, 'r', linewidth=3) 
            title = "Fit results: $\mu$ = %.2f, $\sigma$ = %.2f" % (Moyenne, ecart_type) 
            P.title(title)
            P.show()

    def plot_prediction(self,sn,Error=False,TITLE=None):
        
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
            if len(self.Linear_covariance)!=0:
                P.fill_between(Time_predict,self.Prediction[sn]-CST_top-N.sqrt(N.diag(self.Linear_covariance[sn])),self.Prediction[sn]-CST_top+N.sqrt(N.diag(self.Linear_covariance[sn])),color='r',alpha=0.7 )
            P.fill_between(Time_predict,self.Prediction[sn]-CST_top-Y_err,self.Prediction[sn]-CST_top+Y_err,color='b',alpha=0.7 )

        P.ylabel('Mag AB + cst')
        if TITLE:
            P.title(TITLE)

        P.legend()
        P.xticks([-50,150],['toto','tata'])
        P.ylim(N.min(self.Prediction[sn]-CST_top)-1,N.max(self.Prediction[sn]-CST_top)+1)
        P.xlim(-15,45)
        P.gca().invert_yaxis()

        P.subplot(gs[1])
        
        Mean_Y=Interpolate_mean(self.Time_mean,self.Mean_Y,self.Time[sn])
        Mean_Y_new_binning=Interpolate_mean(self.Time_mean,self.Mean_Y,Time_predict)
        CST_bottom=self.diff[sn]#N.mean(self.Prediction[sn]-Mean_Y_new_binning)

        P.scatter(self.Time[sn],self.y[sn]-Mean_Y-CST_bottom,c='r')
        P.plot(Time_predict,self.Prediction[sn]-Mean_Y_new_binning-CST_bottom,'b')
        if Error:
            P.errorbar(self.Time[sn],self.y[sn]-Mean_Y-CST_bottom, linestyle='', yerr=self.y_err[sn],ecolor='red',alpha=0.9,marker='.',zorder=0)
            if len(self.Linear_covariance)!=0:
                P.fill_between(Time_predict,self.Prediction[sn]-CST_bottom-Mean_Y_new_binning-N.sqrt(N.diag(self.Linear_covariance[sn])),self.Prediction[sn]-Mean_Y_new_binning+N.sqrt(N.diag(self.Linear_covariance[sn]))-CST_bottom,color='r',alpha=0.7 )
            P.fill_between(Time_predict,self.Prediction[sn]-Mean_Y_new_binning-Y_err-CST_bottom,self.Prediction[sn]-Mean_Y_new_binning+Y_err-CST_bottom,color='b',alpha=0.7 )

        P.plot(Time_predict,N.zeros(len(self.Prediction[sn])),'k')
        
        P.xlim(-15,45)
        P.ylim(N.min(self.Prediction[sn]-CST_bottom-Mean_Y_new_binning)-0.5,N.max(self.Prediction[sn]-CST_bottom-Mean_Y_new_binning)+0.5)
        P.ylabel('Mag AB - $M_0(t)$')
        P.xlabel('Time (days)')


class build_pull:

    def __init__(self,y,y_err,Time,Time_mean,Mean_Y,sigma,L):

        self.y=y
        self.N_sn=len(y)
        self.y_err=y_err
        self.Time=Time
        self.Mean_Y=Mean_Y
        self.Time_mean=Time_mean
        self.sigma=sigma
        self.L=L

    def compute_pull(self,diFF=None):

        if diFF is None:
            diFF=N.zeros(self.N_sn)

        self.pull=[]
        self.PULL=[]

        for sn in range(self.N_sn):
            print '%i/%i'%((sn+1,self.N_sn))
            Pred=N.zeros(len(self.Time[sn]))
            Pred_var=N.zeros(len(self.Time[sn]))
            for t in range(len(self.Time[sn])):
                FILTRE=N.array([True]*len(self.Time[sn]))
                FILTRE[t]=False
                GPP=find_global_hyperparameters([self.y[sn][FILTRE]],[self.y_err[sn][FILTRE]],[self.Time[sn][FILTRE]],self.Time_mean,self.Mean_Y)
                GPP.substract_Mean(diff=[diFF[sn]])
                GPP.hyperparameters.update({'sigma':self.sigma,
                                           'l':self.L})

                GPP.get_prediction(new_binning=self.Time[sn])
                Pred[t]=GPP.Prediction[0][t]
                Pred_var[t]=GPP.covariance_matrix[0][t,t]

            pull=(Pred-self.y[sn])/N.sqrt(self.y_err[sn]**2+Pred_var)
            self.pull.append(pull)
            for t in range(len(self.Time[sn])):
                self.PULL.append(pull[t])


        self.Moyenne_pull,self.ecart_type_pull=NORMAL_LAW.fit(self.PULL)


    def plot_result(self,BIN=60,Lambda=None):
        
        P.hist(self.PULL,bins=BIN,normed=True)
        xmin, xmax = P.xlim()
        MAX=max([abs(xmin),abs(xmax)])
        P.xlim(-MAX,MAX)
        xmin, xmax = P.xlim()
        X = N.linspace(xmin, xmax, 100)
        PDF = NORMAL_LAW.pdf(X, self.Moyenne_pull, self.ecart_type_pull)
        P.plot(X, PDF, 'r', linewidth=3)
        if Lambda is None:
            title = "Fit results: $\mu$ = %.2f, $\sigma$ = %.2f" % (self.Moyenne_pull, self.ecart_type_pull)
        else:
            title = "Fit results ($\lambda = %i \AA$): $\mu$ = %.2f, $\sigma$ = %.2f" %((Lambda,self.Moyenne_pull, self.ecart_type_pull))
        P.title(title)
        P.ylabel('Number of points (normed)')
        P.xlabel('Pull')
        P.show()



class compare_spline_GP:

    def __init__(self,y,y_err,Time,Time_mean,Mean_Y,sigma,L):

        self.y=y
        self.N_sn=len(y)
        self.y_err=y_err
        self.Time=Time
        self.Mean_Y=Mean_Y
        self.Time_mean=Time_mean
        self.sigma=sigma
        self.L=L

    def compute_interpolation_prediction(self,SN,diFF=None,FILTRE=None,Time=N.linspace(-12,42,19)):

        if diFF is None:
            diFF=N.zeros(self.N_sn)

        self.pull=[]
        self.PULL=[]
        self.new_binning=Time
        self.sn=SN

        self.FILTRE=FILTRE

        for sn in range(self.N_sn):
            
            if sn == SN: 
                if FILTRE is None :
                    FILTRE=N.array([True]*len(self.Time[sn]))
                GPP=find_global_hyperparameters([self.y[sn][FILTRE]],[self.y_err[sn][FILTRE]],[self.Time[sn][FILTRE]],LCMC.Time_Mean,LCMC.Mean)
                GPP.substract_Mean(diff=[diFF[sn]])
                GPP.hyperparameters.update({'sigma':self.sigma,
                                            'l':self.L})

                GPP.get_prediction(new_binning=Time)
                self.Pred=GPP.Prediction[0]
                self.Pred_var=N.diag(GPP.covariance_matrix[0])
                func = inter.UnivariateSpline(self.Time[sn][FILTRE], self.y[sn][FILTRE],w=1./self.y_err[sn][FILTRE]**2,k=3)
                self.cubic_spline=func(Time)


    def plot_prediction(self,TITLE=None,ERROR=False,SPLINE=True):
        

        Time_predict=self.new_binning


        P.figure(figsize=(12,8))

        CST_top=N.mean(self.Pred)

        Y_err=N.sqrt(self.Pred_var)

        if N.sum(self.FILTRE)==len(self.Time[self.sn]):
            P.scatter(self.Time[self.sn][self.FILTRE],self.y[self.sn][self.FILTRE]-CST_top,c='r',s=75,label='Data')
            P.errorbar(self.Time[self.sn][self.FILTRE],self.y[self.sn][self.FILTRE]-CST_top, linestyle='', yerr=self.y_err[self.sn][self.FILTRE],ecolor='red',alpha=0.9,marker='.',zorder=0)
        else:
            P.scatter(self.Time[self.sn][self.FILTRE],self.y[self.sn][self.FILTRE]-CST_top,c='r',s=75,label='Data')
            P.scatter(self.Time[self.sn][~self.FILTRE],self.y[self.sn][~self.FILTRE]-CST_top,s=120,facecolors='none',marker='^',edgecolors='r',label='Data not used')
            P.errorbar(self.Time[self.sn][self.FILTRE],self.y[self.sn][self.FILTRE]-CST_top, linestyle='', yerr=self.y_err[self.sn][self.FILTRE],ecolor='red',alpha=0.9,marker='.',zorder=0)
            P.errorbar(self.Time[self.sn][~self.FILTRE],self.y[self.sn][~self.FILTRE]-CST_top, linestyle='', yerr=self.y_err[self.sn][~self.FILTRE],ecolor='red',alpha=0.9,marker='.',zorder=0)
        if SPLINE:
            P.plot(Time_predict,self.cubic_spline-CST_top,'k',label='Cubic spline interpolation',linewidth=2) 
        P.plot(Time_predict,self.Pred-CST_top,'b',label='Gaussian process prediction',linewidth=2)

        if ERROR:
            P.fill_between(Time_predict,self.Pred-CST_top-Y_err,self.Pred-CST_top+Y_err,color='b',alpha=0.4 )

        P.ylabel('Mag AB + cst')
        if TITLE:
            P.title(TITLE)
        P.ylim(-2.1,2.1)
        P.gca().invert_yaxis()
        P.legend()
        #P.xticks([-50,150],['toto','tata'])
        P.xlim(-15,45)
        P.xlabel('Time (days)')       



class generate_light_curves_SUGAR_MC:

    def __init__(self,SED):

        SED=N.loadtxt(SED) 
         
        #SED=cPickle.load(open(SED))      
        #SED_at_max=cPickle.load(open(SED_at_max))       
         
        self.alpha=SED[:,3:6]
        self.M0=SED[:,2]
        self.grey=N.ones(len(self.alpha[:,0]))
        self.Red=SED[:,6]

        self.Time=N.linspace(-12,42,19)
        self.X=SED[:,1]
        self.Time_scale=19
        self.wavelength_scale=170

    def select_bin(self,Bin):

        self.alpha1_LC=self.alpha[:,0][Bin*self.Time_scale:(Bin+1)*self.Time_scale]
        self.alpha2_LC=self.alpha[:,1][Bin*self.Time_scale:(Bin+1)*self.Time_scale]
        self.alpha3_LC=self.alpha[:,2][Bin*self.Time_scale:(Bin+1)*self.Time_scale]

        self.M0_LC=self.M0[Bin*self.Time_scale:(Bin+1)*self.Time_scale]
        
        self.Red_LC=self.Red[Bin*self.Time_scale:(Bin+1)*self.Time_scale]
        self.grey_LC=N.ones(self.Time_scale)

        
    def generate_LC_MC(self,BIN,N_sn):
        
        self.TIME=[]
        self.select_bin(BIN)

        self.Supernovae_MC=N.zeros((N_sn,self.Time_scale))

        N.random.seed(1)

        for sn in range(N_sn):

            self.TIME.append(self.Time)
            self.Supernovae_MC[sn]+=self.M0_LC
            
            self.Supernovae_MC[sn]+=N.random.normal(scale=3.5)*self.alpha1_LC
            self.Supernovae_MC[sn]+=N.random.normal(scale=3.)*self.alpha2_LC
            self.Supernovae_MC[sn]+=N.random.normal(scale=1.)*self.alpha3_LC

            self.Supernovae_MC[sn]+=N.random.normal(scale=0.25)*self.Red_LC

            self.Supernovae_MC[sn]+=N.random.normal(scale=0.10)*self.grey_LC

        self.TIME=N.array(self.TIME)

        self.N_sn=N_sn

    def mixe_time(self):

        Supernovae_MC=[]
        TIME=[]

        for i in range(self.N_sn):
            T=N.linspace(self.Time[0],self.Time[-1],10+i)
            
            Supernovae_MC.append(Interpolate_mean(self.TIME[i],self.Supernovae_MC[i],T))
            TIME.append(T)


        self.Supernovae_MC=Supernovae_MC
        self.TIME=TIME

    def add_noise(self,sigma=0.15):
        
        N.random.seed(1)
        
        self.Y_err=[]
        
        for sn in range(self.N_sn) :
            self.Y_err.append(N.zeros(len(self.Supernovae_MC[sn])))
            for i in range(len(self.Supernovae_MC[sn])):
                noise = N.random.normal(scale=sigma)
                #noise=0
                self.Supernovae_MC[sn][i] += noise
                self.Y_err[sn][i]=N.sqrt(noise**2)
                
        

class generate_light_curves_hyper_parameter(generate_light_curves_SUGAR_MC):

    def __init__(self,SED,sigma=2,l=6):

        generate_light_curves_SUGAR_MC.__init__(self,SED)
        self.sigma=sigma
        self.l=l
        self.K=RBF_kernel_1D(self.Time ,self.sigma,self.l,0.)
        

    def generate_supernovae(self,Bin,N_sn):
         self.generate_LC_MC(Bin,1)
         self.Supernovae_MC=N.zeros((N_sn,len(self.Time)))
         TIME=[]
         for sn in range(N_sn):
              self.Supernovae_MC[sn]=N.random.multivariate_normal(self.M0_LC,self.K)
              TIME.append(self.Time)
         
         self.TIME=TIME
         self.Y_err=N.zeros(N.shape(self.Supernovae_MC))
         self.N_sn=N_sn


class Build_light_curves_from_SNF_data:
    
    def __init__(self,Repertoire_in,Bin,sn_list,Number_bin_wavelength=190):

        self.sn_name=N.loadtxt(sn_list,dtype='string')
        #self.sn_name=self.sn_name[:,8]

        self.Repertoire_in=Repertoire_in
        self.Bin=Bin
        self.Number_bin_wavelength=Number_bin_wavelength
        self.Y=[]
        self.Y_err=[]
        self.TIME=[]
        self.wavelength=[]

        
    def build_data(self):

        for i,sn in enumerate(self.sn_name):
            data=N.loadtxt(self.Repertoire_in+sn)
            Number_points=len(data[:,0])/self.Number_bin_wavelength
            Y=N.zeros(Number_points)
            Y_err=N.zeros(Number_points)
            Time=N.zeros(Number_points)
            wavelength=N.zeros(Number_points)
            
            for j in range(Number_points):
                Y[j]=data[(self.Number_bin_wavelength*j)+self.Bin,2]
                Y_err[j]=data[(self.Number_bin_wavelength*j)+self.Bin,3]
                Time[j]=data[(self.Number_bin_wavelength*j)+self.Bin,0]
                wavelength[j]=data[(self.Number_bin_wavelength*j)+self.Bin,1]

            self.Y.append(Y)
            self.Y_err.append(Y_err)
            self.TIME.append(Time)
            self.wavelength.append(wavelength)

            
    def build_mean(self,Mean_file):

        data=N.loadtxt(Mean_file)
        Number_points=len(data[:,0])/self.Number_bin_wavelength
        Y=N.zeros(Number_points)
        Time=N.zeros(Number_points)
        wavelength=N.zeros(Number_points)
        self.Mean_file=Mean_file

        Y=data[Number_points*self.Bin:(Number_points*self.Bin)+Number_points,2]
        Time=data[Number_points*self.Bin:(Number_points*self.Bin)+Number_points,0]
        wavelength=data[Number_points*self.Bin:(Number_points*self.Bin)+Number_points,1]

        self.Time_Mean=Time
        self.Mean=Y
        self.W_mean=wavelength

    def build_difference_mean(self):

        self.difference=N.zeros(len(self.sn_name))
        data=N.loadtxt(self.Mean_file)
        
        delta_mean=0
        delta_lambda=0

        for i in range(len(data[:,0])):
            if data[:,0][0]==data[:,0][i]:
                delta_lambda+=1
            if data[:,1][0]==data[:,1][i]:
                delta_mean+=1

        for i,sn in enumerate(self.sn_name):
            print sn
            Phase=self.TIME[i]
            Time=self.Time_Mean
            DELTA=len(Phase)
            self.Mean_new_binning=N.zeros(delta_lambda*len(Phase))
            
            for Bin in range(delta_lambda):
                self.Mean_new_binning[Bin*DELTA:(Bin+1)*DELTA]=Interpolate_mean(self.Time_Mean,data[:,2][Bin*delta_mean:(Bin+1)*delta_mean],Phase)

            reorder = N.arange(delta_lambda*DELTA).reshape(delta_lambda, DELTA).T.reshape(-1)
            self.Mean_new_binning=self.Mean_new_binning[reorder]

            Data=N.loadtxt(self.Repertoire_in+sn)
            self.difference[i]=N.mean(Data[:,2]-self.Mean_new_binning)
            
     #       if i%50==0:  
     #           P.plot(self.Mean_new_binning,'k')
     #           P.plot(Data[:,2]-self.difference[i],'b')
     #           P.gca().invert_yaxis()
     #           P.show()
     


    def control_plot(self):

        P.figure()
        for i in range(len(self.sn_name)):
            P.scatter(self.TIME[i],self.Y[i],c='b')
            P.errorbar(self.TIME[i],self.Y[i], linestyle='', yerr=self.Y_err[i],ecolor='red',alpha=0.9,marker='.',zorder=0)
            P.plot(self.Time_Mean,self.Mean)
        P.gca().invert_yaxis()
        P.show()


if __name__=="__main__":

    
    
    option = read_option()

    if not option.MC:

        option.bin=int(option.bin)

        LCMC=Build_light_curves_from_SNF_data(option.directory_input,option.bin,option.sn_list,Number_bin_wavelength=190)
        LCMC.build_data()
        LCMC.build_mean(option.mean)
        LCMC.build_difference_mean()

        
        GP=Gaussian_process(LCMC.Y,LCMC.Y_err,LCMC.TIME,LCMC.Time_Mean,LCMC.Mean)
        #if option.bin==83 or option.bin==84:
        #GP.find_hyperparameters(sigma_guess=0.362771407636,l_guess=7.7891745456)
        GP.substract_Mean(diff=LCMC.difference)
        GP.find_hyperparameters(sigma_guess=0.5,l_guess=8.)
        #else:
        #    GP.find_hyperparameters(sigma_guess=None,l_guess=None)
        #GP.get_pull(PLOT=False)
        #GP.get_prediction(N.linspace(-12,42,19))
        GP.get_prediction(new_binning=N.linspace(-12,42,19))
        #GP.get_prediction(new_binning=N.linspace(-5,30,60))
        #GP.get_prediction(new_binning=None)
        #GP.plot_prediction(i,Error=True,TITLE=LCMC.sn_name[i][:-4]+' ( $ \lambda=%i \AA$, $\sigma= %.2f$, $l=%.2f$ days)'%((N.mean(LCMC.wavelength[0]),dic['Hyperparametre']['sigma'],dic['Hyperparametre']['l'])))
        BP= build_pull(LCMC.Y,LCMC.Y_err,LCMC.TIME,LCMC.Time_Mean,LCMC.Mean,GP.hyperparameters['sigma'],GP.hyperparameters['l'])
        BP.compute_pull(diFF=LCMC.difference)
        #BP.plot_result()

        #CSGP=compare_spline_GP(LCMC.Y,LCMC.Y_err,LCMC.TIME,LCMC.Time_Mean,LCMC.Mean,GP.hyperparameters['sigma'],GP.hyperparameters['l'])
        #FF=N.array([True]*len(LCMC.TIME[71]))
        #FF[2]=False
        #FF[3]=False
        #FF[5]=False
        #FF[6]=False
        #FF[9]=False
        ##FF[0:6]=False
        ##FF[17:22]=False
        #CSGP.compute_interpolation_prediction(71,diFF=LCMC.difference,FILTRE=FF,Time=N.linspace(-12,42,100))
        #CSGP.plot_prediction(TITLE=LCMC.sn_name[71][:-4]+' ( $ \lambda=%i \AA$)'%(N.mean(LCMC.wavelength[0])),ERROR=True)

        dic={'Prediction':GP.Prediction,
             'Covariance':GP.covariance_matrix,
             'Hyperparametre':GP.hyperparameters,
             'Hyperparametre_cov':GP.hyperparameters_Covariance,
             'Time':GP.new_binning,
             'wavelength':N.mean(LCMC.wavelength[0]),
             'pull_per_supernovae':BP.pull,
             'global_pull':BP.PULL,
             'Moyenne_pull':BP.Moyenne_pull,
             'ecart_tupe_pull':BP.ecart_type_pull,
             'sn_name':LCMC.sn_name}
  

        File=open(option.directory_output+'Prediction_Bin_%i.pkl'%(option.bin),'w')
        cPickle.dump(dic,File)
        File.close()


    else:
        sed='/sps/snovae/user/leget/BEDELL/SUGAR_model.asci'

        
        N_SN=int(option.SN)
        BIN=42

        SIGG=0.4
        LL=8.

        LCMC=generate_light_curves_hyper_parameter(sed,sigma=SIGG,l=LL)
        LCMC.generate_supernovae(BIN,N_SN)
        LCMC.mixe_time()
        LCMC.add_noise(sigma=0.05)

        GP=find_global_hyperparameters(LCMC.Supernovae_MC,LCMC.Y_err,LCMC.TIME,N.linspace(-12,42,19),LCMC.M0_LC)
        GP.substract_Mean(diff=None)
        GP.find_hyperparameters(sigma_guess=0.5,l_guess=8.)
        GP.get_prediction(new_binning=N.linspace(-12,42,19))
        dic={'Prediction':GP.Prediction,
             'Covariance':GP.covariance_matrix,
             'Hyperparametre':GP.hyperparameters,
             'Hyperparametre_cov':GP.hyperparameters_Covariance,
             'Time':GP.new_binning,
             }
  

        File=open(option.directory_output+'Prediction_N_SN_%i.pkl'%(option.SN),'w')
        cPickle.dump(dic,File)
        File.close()

    #for i in range(N_SN):
    #     P.figure()
    #     P.errorbar(LCMC.TIME[i],LCMC.Supernovae_MC[i], linestyle='', yerr=LCMC.Y_err[i],ecolor='red',alpha=0.9,marker='.',zorder=0)
    #     P.scatter(LCMC.TIME[i],LCMC.Supernovae_MC[i])
    #     P.gca().invert_yaxis()
   
    #LCMC=generate_light_curves_SUGAR_MC(sed)
    #LCMC.generate_LC_MC(BIN,N_SN)
    ##LCMC.mixe_time()
    #LCMC.add_noise(sigma=0.05)

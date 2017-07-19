""" Test of differents implementations of GP."""

import cosmogp
import george
import sklearn.gaussian_process.kernels as skl_kernels
from sklearn.gaussian_process.gpr import GaussianProcessRegressor as skl_gpr
from svd_tmv import computeSVDInverse
from scipy.stats import norm as normal
import numpy as np
import pylab as plt
import time



def generate_data_test(number_of_data,number_of_point,
                       kernel_amplitude=1.,correlation_length=1.,
                       white_noise=0,seed=1):
    """
    Generation of simulated data for GP.
    Generating random gaussian fields in 1 dimensions (y depend 
    only from the x axis). Used of Squared exponential kernel.
    """
    np.random.seed(seed)
    
    x_axis = np.linspace(0,5,number_of_point)

    x = np.zeros((number_of_data,number_of_point))
    y = np.zeros((number_of_data,number_of_point))

    kernel = cosmogp.rbf_kernel_1d(x_axis, [kernel_amplitude,correlation_length],
                                   white_noise, floor=0.0, y_err=None)

    #inv_K, det_kernel = computeSVDInverse(kernel,return_logdet=True)
    det_kernel = np.linalg.det(kernel)
    
    for i in range(number_of_data):
        y[i] = np.random.multivariate_normal(np.zeros(len(x_axis)), kernel)
        x[i] = x_axis

    return x,y,kernel,det_kernel


def cosmogp_gp(x,y,xpredict=None,kernel='rbf1d',
               hyperparameter_default=None,
               search_hyperparameter=False):

    assert kernel in ['rbf1d'], 'not implemented kernel'


    if kernel == 'rbf1d' :
        kernel ='RBF1D'
        if hyperparameter_default is None:
            hyp = [1.,1.]
        else:
            hyp = hyperparameter_default

    timeA = time.time()
    gp = cosmogp.gaussian_process(x,y,kernel=kernel)

    if search_hyperparameter:
        gp.find_hyperparameters()
    else:
        gp.hyperparameter = hyp

    if xpredict is None:
        xpredict = x
        
    gp.get_prediction(new_binning=xpredict)
    ynew = gp.Prediction[0]    
    timeB = time.time()
    
    time_excution = timeB-timeA

    resids = []
    stds = []
    for i in range(len(x)):
        x_ = np.concatenate([x[:i], x[i+1:]])
        y_ = np.concatenate([y[:i], y[i+1:]])
        
        gp = cosmogp.gaussian_process(x_,y_,kernel=kernel)
        gp.get_prediction(xpredict)
        ypredict = gp.Prediction[0][i]
        ystd = np.sqrt(np.diag(gp.covariance_matrix[0]))
        ystd = ystd[i]
        
        resids.append(ypredict-y[i])
        stds.append(ystd)
    
    resids = np.array(resids)
    stds  = np.array(stds)

    
    return ynew,resids,stds,time_excution 


#def george_gp(x,y,xnew,kernel='rbf1d',
#              search_hyperparameter=False):
#
#    if kernel is 'rbf1d':
#        from george.kernels import ExpSquaredKernel as kernel_george
#
#    kernel = kernel_george(1.0)
#
#    timeA = time.time()
#    
#    gp = george.GP(kernel)
#    gp.compute(x, yerr)
#    gp.lnlikelihood(y)
#
#    t = np.linspace(0, 10, 500)
#    mu, cov = gp.predict(y, t)
#
#    timeB = time.time()
#    
#    return ynew,pull,std,time


def scikitlearn_gp(x,y,xpredict=None,kernel='rbf1d',
                   hyperparameter_default=None,
                   search_hyperparameter=False):

    x = x.reshape(len(x),1)
    y = y.reshape(len(y),1)
    
    assert kernel in ['rbf1d'], 'not implemented kernel'

    if kernel == 'rbf1d' :
        
        if hyperparameter_default is None:
            hyp = [1.,1.]
        else:
            hyp = hyperparameter_default
            
        ker = skl_kernels.ConstantKernel(constant_value = hyp[0]**2) * skl_kernels.RBF(length_scale = hyp[1])
    
    timeA = time.time()
    
    if not search_hyperparameter:
        gpr = skl_gpr(ker, optimizer=None)
    else:
        gpr = skl_gpr(ker)
        
    gpr.fit(x, y)
    
    if xpredict is None:
        xpredict = x
        
    ynew, diag_err = gpr.predict(xpredict, return_std=True)
    
    timeB = time.time()

    time_excution = timeB-timeA

    resids = []
    stds = []
    for i in range(len(x)):
        x_ = np.concatenate([x[:i], x[i+1:]])
        y_ = np.concatenate([y[:i], y[i+1:]])
        
        gpr.fit(x_, y_)
        ypredict, ystd = gpr.predict(x[i].reshape(-1, 1), return_std=True)
        resids.append(ypredict[0,0]-y[i,0])
        stds.append(ystd)
    
    resids = np.array(resids)
    stds  = np.array(stds)
        
    return ynew.T[0],resids,stds.T[0],time_excution



class test_gaussian_process:

    def __init__(self,kernel='rbf1d',search_hyperparameter=False,Number_data=10):
        """
        Test to compare different Gaussian Process codes.
        Three codes are tested.
        - Scikit learn implementation of GP.
        - George GP code 
        - cosmogp code (done for SNIa intepolation of SED)

        kernel: string, type of kernel 

        search_hyperparameter: boolean, optimization or 
        not of hyperparameter. Per default, no optimization 
        is done. 
        """

        self.kernel = kernel
        self.search_hyperparameter = search_hyperparameter
        
        #self.N = np.logspace(1,2,10).astype(int)
        self.N = np.linspace(5,50,10).astype(int)
        self.Number_data = Number_data
        

        self.ynew_skl = []
        self.resids_skl = []
        self.stds_skl = []
        self.time_skl = np.zeros(len(self.N))

        self.ynew_cosmogp = []
        self.resids_cosmogp = []
        self.stds_cosmogp = []
        self.time_cosmogp = np.zeros(len(self.N))

        self.ynew_george = []
        self.resids_george = []
        self.stds_george = []
        self.time_george = np.zeros(len(self.N))

        
    def run_test(self):

        for i in range(len(self.N)):

            print i
            
            x, y, kernel, det_kernel = generate_data_test(self.Number_data,self.N[i],
                                                          kernel_amplitude=1.,correlation_length=1.,
                                                          white_noise=0,seed=1)

            
            ynew_skl = np.zeros(self.N[i]*self.Number_data)
            resids_skl = np.zeros(self.N[i]*self.Number_data)
            stds_skl = np.zeros(self.N[i]*self.Number_data)
            t_skl = np.zeros(self.Number_data)
            
            ynew_cosmogp = np.zeros(self.N[i]*self.Number_data)
            resids_cosmogp = np.zeros(self.N[i]*self.Number_data)
            stds_cosmogp = np.zeros(self.N[i]*self.Number_data)
            t_cosmogp = np.zeros(self.Number_data)
            
            ynew_george = np.zeros(self.N[i]*self.Number_data)
            resids_george = np.zeros(self.N[i]*self.Number_data)
            stds_george = np.zeros(self.N[i]*self.Number_data)
            t_george = np.zeros(self.Number_data)
            
            t = 0
            
            for j in range(self.Number_data):

                #scikit learn gaussian process 
            
                ynew,resids,stds,time_excution = scikitlearn_gp(x[j],y[j],xpredict=None,kernel=self.kernel,
                                                                hyperparameter_default=[1.,1.],
                                                                search_hyperparameter=self.search_hyperparameter)
                
                ynew_skl[t:t+len(ynew)] = ynew
                resids_skl[t:t+len(ynew)] = resids
                stds_skl[t:t+len(ynew)] = stds
                t_skl[j] = time_excution

                #cosmogp gaussian process

                ynew,resids,stds,time_excution = cosmogp_gp(x[j],y[j],xpredict=None,kernel=self.kernel,
                                                            hyperparameter_default=[1.,1.],
                                                            search_hyperparameter=self.search_hyperparameter)
                ynew_cosmogp[t:t+len(ynew)] = ynew
                resids_cosmogp[t:t+len(ynew)] = resids
                stds_cosmogp[t:t+len(ynew)] = stds
                t_cosmogp[j] = time_excution

                #george gaussian process

                #TO DO
                
                t+=len(ynew)
 
            self.ynew_skl.append(ynew_skl)
            self.resids_skl.append(resids_skl)
            self.stds_skl.append(stds_skl)            
            self.time_skl[i] = np.mean(t_skl)

            self.ynew_cosmogp.append(ynew_cosmogp)
            self.resids_cosmogp.append(resids_cosmogp)
            self.stds_cosmogp.append(stds_cosmogp)            
            self.time_cosmogp[i] = np.mean(t_cosmogp)
                

    def plot_test_result(self):

        self.pull_std_skl = np.zeros(len(self.N))
        self.pull_mean_skl = np.zeros(len(self.N))
        self.res_std_skl = np.zeros(len(self.N))
        self.res_mean_skl = np.zeros(len(self.N))
        self.WARNING_skl = np.array([False]*len(self.N))


        self.pull_std_cosmogp = np.zeros(len(self.N))
        self.pull_mean_cosmogp = np.zeros(len(self.N))
        self.res_std_cosmogp = np.zeros(len(self.N))
        self.res_mean_cosmogp = np.zeros(len(self.N))
        self.WARNING_cosmogp = np.array([False]*len(self.N))
        
        for i in range(len(self.N)):
            self.pull_mean_skl[i], self.pull_std_skl[i] = normal.fit(self.resids_skl[i]/self.stds_skl[i])
            self.WARNING_skl[i] = np.isnan(self.pull_mean_skl[i])
            if self.WARNING_skl[i]:
                pull = self.resids_skl[i]/self.stds_skl[i]
                self.pull_mean_skl[i], self.pull_std_skl[i] = normal.fit(pull[np.isfinite(pull)])
            self.res_mean_skl[i], self.res_std_skl[i] = normal.fit(self.resids_skl[i])

            self.pull_mean_cosmogp[i], self.pull_std_cosmogp[i] = normal.fit(self.resids_cosmogp[i]/self.stds_cosmogp[i])
            self.WARNING_cosmogp[i] = np.isnan(self.pull_mean_cosmogp[i])
            if self.WARNING_cosmogp[i]:
                pull = self.resids_cosmogp[i]/self.stds_cosmogp[i]
                self.pull_mean_cosmogp[i], self.pull_std_cosmogp[i] = normal.fit(pull[np.isfinite(pull)])
            self.res_mean_cosmogp[i], self.res_std_cosmogp[i] = normal.fit(self.resids_cosmogp[i])
            

        plt.figure(figsize=(14,8))
        plt.plot(self.N,self.time_skl,'r-s',linewidth=3,markersize=10,label='scikit learn (kernel = %s)'%self.kernel)
        plt.plot(self.N,self.time_cosmogp,'b-s',linewidth=3,markersize=10,label='cosmogp (kernel = %s)'%self.kernel)
        plt.ylabel('Time of excution (s)',fontsize=18)
        plt.xlabel('number of points',fontsize=18)
        plt.legend(loc=4)


        plt.figure(figsize=(14,8))
        plt.subplots_adjust(hspace = 0.01)
        
        plt.subplot(2,1,1)
        plt.plot(self.N,self.pull_std_skl,'r',linewidth=3,label='scikit learn (kernel = %s)'%self.kernel,zorder=0)
        plt.scatter(self.N[~self.WARNING_skl],self.pull_std_skl[~self.WARNING_skl],c='r',marker='s',s=75,zorder=10)
        plt.scatter(self.N[self.WARNING_skl],self.pull_std_skl[self.WARNING_skl],c='r',marker='*',s=200,zorder=10)
        
        plt.plot(self.N,self.pull_std_cosmogp,'b',linewidth=3,label='cosmogp (kernel = %s)'%self.kernel,zorder=0)
        plt.scatter(self.N[~self.WARNING_cosmogp],self.pull_std_cosmogp[~self.WARNING_cosmogp],c='b',marker='s',s=75,zorder=10)
        plt.scatter(self.N[self.WARNING_cosmogp],self.pull_std_cosmogp[self.WARNING_cosmogp],c='b',marker='*',s=200,zorder=10)
        plt.xticks([],[])
        plt.ylabel('pull STD',fontsize=18)
        plt.legend()

        plt.subplot(2,1,2)
        plt.plot(self.N,self.pull_mean_skl,'r',linewidth=3,label='scikit learn (kernel = %s)'%self.kernel,zorder=0)
        plt.scatter(self.N[~self.WARNING_skl],self.pull_mean_skl[~self.WARNING_skl],c='r',marker='s',s=75,zorder=10)
        plt.scatter(self.N[self.WARNING_skl],self.pull_mean_skl[self.WARNING_skl],c='r',marker='*',s=200,zorder=10)

        plt.plot(self.N,self.pull_mean_cosmogp,'b',linewidth=3,label='cosmogp (kernel = %s)'%self.kernel,zorder=0)
        plt.scatter(self.N[~self.WARNING_cosmogp],self.pull_mean_cosmogp[~self.WARNING_cosmogp],c='b',marker='s',s=75,zorder=10)
        plt.scatter(self.N[self.WARNING_cosmogp],self.pull_mean_cosmogp[self.WARNING_cosmogp],c='b',marker='*',s=200,zorder=10)
        plt.xlabel('number of points',fontsize=18)
        plt.ylabel('pull average',fontsize=18)

        
        plt.figure(figsize=(14,8))
        plt.subplots_adjust(hspace = 0.01)
        
        plt.subplot(2,1,1)
        plt.plot(self.N,self.res_std_skl,'r',linewidth=3,label='scikit learn (kernel = %s)'%self.kernel,zorder=0)
        plt.scatter(self.N[~self.WARNING_skl],self.res_std_skl[~self.WARNING_skl],c='r',marker='s',s=75,zorder=10)
        plt.scatter(self.N[self.WARNING_skl],self.res_std_skl[self.WARNING_skl],c='r',marker='*',s=200,zorder=10)

        plt.plot(self.N,self.res_std_cosmogp,'b',linewidth=3,label='cosmogp (kernel = %s)'%self.kernel,zorder=0)
        plt.scatter(self.N[~self.WARNING_cosmogp],self.res_std_cosmogp[~self.WARNING_cosmogp],c='b',marker='s',s=75,zorder=10)
        plt.scatter(self.N[self.WARNING_cosmogp],self.res_std_cosmogp[self.WARNING_cosmogp],c='b',marker='*',s=200,zorder=10)
        
        plt.xticks([],[])
        plt.ylabel('residual STD',fontsize=18)
        plt.legend()

        plt.subplot(2,1,2)
        plt.plot(self.N,self.res_mean_skl,'r',linewidth=3,label='scikit learn (kernel = %s)'%self.kernel,zorder=0)
        plt.scatter(self.N[~self.WARNING_skl],self.res_mean_skl[~self.WARNING_skl],c='r',marker='s',s=75,zorder=10)
        plt.scatter(self.N[self.WARNING_skl],self.res_mean_skl[self.WARNING_skl],c='r',marker='*',s=200,zorder=0)

        plt.plot(self.N,self.res_mean_cosmogp,'b',linewidth=3,label='cosmogp (kernel = %s)'%self.kernel,zorder=0)
        plt.scatter(self.N[~self.WARNING_cosmogp],self.res_mean_cosmogp[~self.WARNING_cosmogp],c='b',marker='s',s=75,zorder=10)
        plt.scatter(self.N[self.WARNING_cosmogp],self.res_mean_cosmogp[self.WARNING_cosmogp],c='b',marker='*',s=200,zorder=10)
        
        plt.xlabel('number of points',fontsize=18)
        plt.ylabel('residual average',fontsize=18)

        

        

            
if __name__=='__main__':
    
    test_gp = test_gaussian_process(kernel='rbf1d',search_hyperparameter=False)
    test_gp.run_test()
    test_gp.plot_test_result()

        
        
    
    
        

""" Test of differents implementations of GP."""

import cosmogp
import george
import george.kernels as kernel_george
import sklearn.gaussian_process.kernels as skl_kernels
from sklearn.gaussian_process.gpr import GaussianProcessRegressor as skl_gpr
from svd_tmv import computeSVDInverse
from scipy.stats import norm as normal
import numpy as np
import pylab as plt
import time


def generate_data_test(number_of_data,number_of_point,
                       kernel_amplitude=1.,correlation_length=1.,
                       white_noise=0,noise=0.2,seed=1):
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
                                   nugget=white_noise, floor=0.0, y_err=None)

    #inv_K, det_kernel = computeSVDInverse(kernel,return_logdet=True)
    det_kernel = np.linalg.det(kernel)

    for i in range(number_of_data):
        if noise != 0:
            y[i] = np.random.multivariate_normal(np.zeros(len(x_axis)), kernel + (np.eye(len(kernel)) * noise**2))
        else:
            y[i] = np.random.multivariate_normal(np.zeros(len(x_axis)), kernel)
        x[i] = x_axis

    return x,y,kernel,det_kernel


def cosmogp_gp(x, y, y_err=None, xpredict=None, kernel='rbf1d',
               hyperparameter_default=None,
               search_hyperparameter=False,
               substract_mean=False, svd=True):
    """
    cosmogp gaussian process.
    """

    print 'cosmogp'
    assert kernel in ['rbf1d'], 'not implemented kernel'


    if kernel == 'rbf1d' :
        kernel ='RBF1D'
        if hyperparameter_default is None:
            hyp = [1.,1.]
        else:
            hyp = hyperparameter_default

    timeA = time.time()
    gp = cosmogp.gaussian_process(y, x, y_err=y_err, kernel=kernel,
                                  substract_mean=substract_mean)

    if search_hyperparameter:
        gp.find_hyperparameters()
    else:
        gp.hyperparameters = hyp

    if xpredict is None:
        xpredict = x

    gp.get_prediction(new_binning=xpredict,svd_method=svd)
    ynew = gp.Prediction[0]
    cov = gp.covariance_matrix[0]
    timeB = time.time()
    
    if y_err is None:
        y_err = None
    else:
        y_err = [y_err]
    
    time_excution = timeB-timeA
        
    bp = cosmogp.build_pull([y], [x], gp.hyperparameters,
                            kernel=kernel, y_err=y_err)
    bp.compute_pull(diff=None, svd_method=svd,
                    substract_mean=substract_mean)
    resids = np.array(bp.residual)
    stds = np.array(bp.residual) / np.array(bp.pull)

    return ynew,resids,stds,time_excution,gp.hyperparameters


def george_gp(x, y, y_err=None, xpredict=None, kernel='rbf1d',
              hyperparameter_default=None,
              search_hyperparameter=False):
    """
    george gaussian process.
    """
    print 'george'
    assert kernel in ['rbf1d'], 'not implemented kernel'

    if kernel == 'rbf1d' :

        if hyperparameter_default is None:
            hyp = [1.,1.]
        else:
            hyp = hyperparameter_default

        kernel = hyp[0]**2 * kernel_george.ExpSquaredKernel(hyp[1]**2)


    import scipy.optimize as op

    def nll(p):
        gp.set_parameter_vector(p)
        ll = gp.log_likelihood(y, quiet=True)
        return -ll if np.isfinite(ll) else 1e25

    def grad_nll(p):
        gp.set_parameter_vector(p)
        return -gp.grad_log_likelihood(y, quiet=True)
                    
    timeA = time.time()

    if y_err is None:
        y_err = np.zeros(len(y))
        gp = george.GP(kernel)
    else :
        gp = george.GP(kernel)#, solver=george.HODLRSolver)
    gp.compute(x, y_err)

    if search_hyperparameter:
      p0 = gp.get_parameter_vector()
      results = op.minimize(nll, p0, jac=grad_nll, method="L-BFGS-B")
      
                        
    if xpredict is None:
        xpredict = x

    mu, cov = gp.predict(y, xpredict)

    timeB = time.time()

    time_excution = timeB - timeA

    resids = []
    stds = []
    
    for t in range(len(x)):

        filter_pull = np.array([True] * len(x))
        filter_pull[t] = False

        gp.compute(x[filter_pull], y_err[filter_pull])
        mu_, cov_ = gp.predict(y[filter_pull], x)

        resids.append(mu_[t] - y[t])
        stds.append(np.sqrt(abs(cov_[t, t]+y_err[t]**2)))

    resids = np.array(resids)
    stds = np.array(stds)

    return mu,resids,stds,time_excution,np.sqrt(results['x']**2)
    #return mu,cov,resids,stds,time_excution,gp


def scikitlearn_gp(x, y, y_err=None, xpredict=None, kernel='rbf1d',
                   hyperparameter_default=None,
                   search_hyperparameter=False):
    """
    scikit learn gaussian process.
    """
    print 'sklearn'
    x = x.reshape(len(x),1)
    y = y.reshape(len(y),1)

    assert kernel in ['rbf1d'], 'not implemented kernel'

    if kernel == 'rbf1d' :

        if hyperparameter_default is None:
            hyp = [1.,1.]
        else:
            hyp = hyperparameter_default

        
        ker = skl_kernels.ConstantKernel(constant_value = hyp[0]**2) * skl_kernels.RBF(length_scale = hyp[1])

    if y_err is None:
        alpha = 1e-10
        y_err = np.zeros_like(y)
    else:
        alpha = y_err**2

    timeA = time.time()

    if not search_hyperparameter:
        gpr = skl_gpr(ker, alpha=alpha, optimizer=None)
    else:
        gpr = skl_gpr(ker, alpha=alpha)

        
    gpr.fit(x, y)
    hyp = np.sqrt(gpr.kernel_.theta**2)

    if xpredict is None:
        xpredict = x

    ynew, cov = gpr.predict(xpredict, return_cov=True)

    timeB = time.time()

    time_excution = timeB-timeA

    resids = []
    stds = []

    for t in range(len(x)):

        filter_pull = np.array([True] * len(x))
        filter_pull[t] = False

        if y_err is not None:
            gpr = skl_gpr(ker, alpha=alpha[filter_pull], optimizer=None)
        else:
            gpr = skl_gpr(ker, alpha=alpha, optimizer=None)
            
        gpr.fit(x[filter_pull], y[filter_pull])
        ypredict, cov = gpr.predict(x, return_cov=True)
        resids.append(ypredict.T[0][t] - y[t,0])
        stds.append(np.sqrt(abs(cov[t, t])+alpha[t]))
    
    resids = np.array(resids)
    stds  = np.array(stds)

    return ynew.T[0],resids,stds,time_excution,hyp


class test_gaussian_process:

    def __init__(self,kernel='rbf1d', noise = 0.2, search_hyperparameter=False, Number_data=10):
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

        if noise !=0.:
            self.N = np.logspace(1,1.7,10).astype(int)
        else:
            self.N = np.linspace(5,20,10).astype(int)
        self.Number_data = Number_data

        self.noise = noise

        

        self.ynew_skl = []
        self.resids_skl = []
        self.stds_skl = []
        self.time_skl = np.zeros(len(self.N))
        self.hyp_skl = np.zeros((len(self.N),2))
        self.hyp_std_skl = np.zeros((len(self.N),2))

        self.ynew_cosmogp = []
        self.resids_cosmogp = []
        self.stds_cosmogp = []
        self.time_cosmogp = np.zeros(len(self.N))
        self.hyp_cosmogp = np.zeros((len(self.N),2))
        self.hyp_std_cosmogp = np.zeros((len(self.N),2))

        self.ynew_george = []
        self.resids_george = []
        self.stds_george = []
        self.time_george = np.zeros(len(self.N))
        self.hyp_george = np.zeros((len(self.N),2))
        self.hyp_std_george = np.zeros((len(self.N),2))


    def run_test(self):

        for i in range(len(self.N)):

            print i

            if self.noise != 0:
                self.y_err = np.ones(self.N[i]) * self.noise
            else:
                self.y_err = None
            
            x, y, kernel, det_kernel = generate_data_test(self.Number_data,self.N[i],
                                                          kernel_amplitude=1.,correlation_length=1.,
                                                          white_noise=0,noise=self.noise,seed=1)


            ynew_skl = np.zeros(self.N[i]*self.Number_data)
            resids_skl = np.zeros(self.N[i]*self.Number_data)
            stds_skl = np.zeros(self.N[i]*self.Number_data)
            t_skl = np.zeros(self.Number_data)
            hyp_skl = np.zeros((self.Number_data,2))

            ynew_cosmogp = np.zeros(self.N[i]*self.Number_data)
            resids_cosmogp = np.zeros(self.N[i]*self.Number_data)
            stds_cosmogp = np.zeros(self.N[i]*self.Number_data)
            t_cosmogp = np.zeros(self.Number_data)
            hyp_cosmogp = np.zeros((self.Number_data,2))

            ynew_george = np.zeros(self.N[i]*self.Number_data)
            resids_george = np.zeros(self.N[i]*self.Number_data)
            stds_george = np.zeros(self.N[i]*self.Number_data)
            t_george = np.zeros(self.Number_data)
            hyp_george = np.zeros((self.Number_data,2))

            t = 0

            for j in range(self.Number_data):

                #scikit learn gaussian process 

                ynew,resids,stds,time_excution,hyp = scikitlearn_gp(x[j],y[j],y_err=self.y_err,xpredict=None,kernel=self.kernel,
                                                                    hyperparameter_default=[1.,1.],
                                                                    search_hyperparameter=self.search_hyperparameter)

                ynew_skl[t:t+len(ynew)] = ynew
                resids_skl[t:t+len(ynew)] = resids
                stds_skl[t:t+len(ynew)] = stds
                t_skl[j] = time_excution
                hyp_skl[j] = hyp

                #cosmogp gaussian process

                ynew,resids,stds,time_excution,hyp = cosmogp_gp(x[j],y[j],y_err=self.y_err,xpredict=None,kernel=self.kernel,
                                                                hyperparameter_default=[1.,1.],
                                                                search_hyperparameter=self.search_hyperparameter)
                ynew_cosmogp[t:t+len(ynew)] = ynew
                resids_cosmogp[t:t+len(ynew)] = resids
                stds_cosmogp[t:t+len(ynew)] = stds
                t_cosmogp[j] = time_excution
                hyp_cosmogp[j] = hyp

                #george gaussian process

                ynew,resids,stds,time_excution,hyp = george_gp(x[j],y[j],y_err=self.y_err,xpredict=None,kernel=self.kernel,
                                                               hyperparameter_default=[1.,1.],
                                                               search_hyperparameter=self.search_hyperparameter)
                ynew_george[t:t+len(ynew)] = ynew
                resids_george[t:t+len(ynew)] = resids
                stds_george[t:t+len(ynew)] = stds
                t_george[j] = time_excution
                hyp_george[j] = hyp

                t+=len(ynew)
 
            self.ynew_skl.append(ynew_skl)
            self.resids_skl.append(resids_skl)
            self.stds_skl.append(stds_skl)            
            self.time_skl[i] = np.mean(t_skl)
            self.hyp_skl[i,0] = np.mean(hyp_skl[:,0])
            self.hyp_std_skl[i,0] = np.std(hyp_skl[:,0],ddof=1.5)
            self.hyp_skl[i,1] = np.mean(hyp_skl[:,1])
            self.hyp_std_skl[i,1] = np.std(hyp_skl[:,1],ddof=1.5)

            self.ynew_cosmogp.append(ynew_cosmogp)
            self.resids_cosmogp.append(resids_cosmogp)
            self.stds_cosmogp.append(stds_cosmogp)            
            self.time_cosmogp[i] = np.mean(t_cosmogp)
            self.hyp_cosmogp[i,0] = np.mean(hyp_cosmogp[:,0])
            self.hyp_std_cosmogp[i,0] = np.std(hyp_cosmogp[:,0],ddof=1.5)
            self.hyp_cosmogp[i,1] = np.mean(hyp_cosmogp[:,1])
            self.hyp_std_cosmogp[i,1] = np.std(hyp_cosmogp[:,1],ddof=1.5)

            self.ynew_george.append(ynew_george)
            self.resids_george.append(resids_george)
            self.stds_george.append(stds_george)            
            self.time_george[i] = np.mean(t_george)
            self.hyp_george[i,0] = np.mean(hyp_george[:,0])
            self.hyp_std_george[i,0] = np.std(hyp_george[:,0],ddof=1.5)
            self.hyp_george[i,1] = np.mean(hyp_george[:,1])
            self.hyp_std_george[i,1] = np.std(hyp_george[:,1],ddof=1.5)

    def plot_test_result(self,skl_invertor = 'Choleski (scipy)', cosmogp_invertor = 'LDL (TMV)', george_invertor = 'Choleski (scipy)'):
        """
        Plot performances of differents gp algorithm.
        """
        if self.noise != 0:
            george_invertor = 'HOLDR (c++)'

        self.pull_std_skl = np.zeros(len(self.N))
        self.pull_mean_skl = np.zeros(len(self.N))
        self.res_std_skl = np.zeros(len(self.N))
        self.res_mean_skl = np.zeros(len(self.N))

        self.pull_std_cosmogp = np.zeros(len(self.N))
        self.pull_mean_cosmogp = np.zeros(len(self.N))
        self.res_std_cosmogp = np.zeros(len(self.N))
        self.res_mean_cosmogp = np.zeros(len(self.N))

        self.pull_std_george = np.zeros(len(self.N))
        self.pull_mean_george = np.zeros(len(self.N))
        self.res_std_george = np.zeros(len(self.N))
        self.res_mean_george = np.zeros(len(self.N))

        for i in range(len(self.N)):
            self.pull_mean_skl[i], self.pull_std_skl[i] = normal.fit(self.resids_skl[i]/self.stds_skl[i])
            self.res_mean_skl[i], self.res_std_skl[i] = normal.fit(self.resids_skl[i])

            self.pull_mean_cosmogp[i], self.pull_std_cosmogp[i] = normal.fit(self.resids_cosmogp[i]/self.stds_cosmogp[i])
            self.res_mean_cosmogp[i], self.res_std_cosmogp[i] = normal.fit(self.resids_cosmogp[i])

            self.pull_mean_george[i], self.pull_std_george[i] = normal.fit(self.resids_george[i]/self.stds_george[i])
            self.res_mean_george[i], self.res_std_george[i] = normal.fit(self.resids_george[i])

        plt.figure(figsize=(14,8))
        plt.plot(self.N,self.time_skl,'r-s',linewidth=3,markersize=10,label='scikit learn ('+skl_invertor+', kernel = %s)'%self.kernel)
        plt.plot(self.N,self.time_cosmogp,'b-s',linewidth=3,markersize=10,label='cosmogp ('+cosmogp_invertor+', kernel = %s)'%self.kernel)
        plt.plot(self.N,self.time_george,'k-s',linewidth=3,markersize=10,label='george ('+george_invertor+', kernel = %s)'%self.kernel)
        plt.ylabel('Time of excution (s)',fontsize=18)
        plt.xlabel('number of points',fontsize=18)
        plt.legend(loc=2)


        plt.figure(figsize=(14,8))
        plt.subplots_adjust(hspace = 0.01)

        plt.subplot(2,1,1)

        plt.plot(self.N,np.ones_like(self.N),'k-.')
        
        plt.plot(self.N,self.pull_std_skl,'r',linewidth=6,label='scikit learn ('+skl_invertor+', kernel = %s)'%self.kernel,zorder=0)
        plt.scatter(self.N,self.pull_std_skl,c='r',marker='s',s=75,zorder=10)

        plt.plot(self.N,self.pull_std_cosmogp,'b',linewidth=4,label='cosmogp ('+cosmogp_invertor+', kernel = %s)'%self.kernel,zorder=0)
        plt.scatter(self.N,self.pull_std_cosmogp,c='b',marker='s',s=75,zorder=10)

        plt.plot(self.N,self.pull_std_george,'k',linewidth=2,label='george ('+george_invertor+', kernel = %s)'%self.kernel,zorder=0)
        plt.scatter(self.N,self.pull_std_george,c='k',marker='s',s=75,zorder=10)
        plt.xticks([],[])
        plt.ylabel('pull STD',fontsize=18)

        plt.ylim(0,2)
        
        plt.legend()

        plt.subplot(2,1,2)

        plt.plot(self.N,np.zeros_like(self.N),'k-.')
        
        plt.plot(self.N,self.pull_mean_skl,'r',linewidth=6,zorder=0)
        plt.scatter(self.N,self.pull_mean_skl,c='r',marker='s',s=75,zorder=10)

        plt.plot(self.N,self.pull_mean_cosmogp,'b',linewidth=4,zorder=0)
        plt.scatter(self.N,self.pull_mean_cosmogp,c='b',marker='s',s=75,zorder=10)

        plt.plot(self.N,self.pull_mean_george,'k',linewidth=2,zorder=0)
        plt.scatter(self.N,self.pull_mean_george,c='k',marker='s',s=75,zorder=10)

        plt.xlabel('number of points',fontsize=18)
        plt.ylabel('pull average',fontsize=18)

        plt.ylim(-0.2,0.2)
        
        plt.figure(figsize=(14,8))
        plt.subplots_adjust(hspace = 0.01)

        plt.subplot(2,1,1)
        plt.plot(self.N,self.res_std_skl,'r',linewidth=6,label='scikit learn ('+skl_invertor+', kernel = %s)'%self.kernel,zorder=0)
        plt.scatter(self.N,self.res_std_skl,c='r',marker='s',s=75,zorder=10)

        plt.plot(self.N,self.res_std_cosmogp,'b',linewidth=4,label='cosmogp ('+cosmogp_invertor+', kernel = %s)'%self.kernel,zorder=0)
        plt.scatter(self.N,self.res_std_cosmogp,c='b',marker='s',s=75,zorder=10)

        plt.plot(self.N,self.res_std_george,'k',linewidth=2,label='george ('+george_invertor+', kernel = %s)'%self.kernel,zorder=0)
        plt.scatter(self.N,self.res_std_george,c='k',marker='s',s=75,zorder=10)

        plt.xticks([],[])
        plt.ylabel('residual STD',fontsize=18)
        plt.legend()

        plt.subplot(2,1,2)

        plt.plot(self.N,np.zeros_like(self.N),'k-.')

        plt.plot(self.N,self.res_mean_skl,'r',linewidth=6,zorder=0)
        plt.scatter(self.N,self.res_mean_skl,c='r',marker='s',s=75,zorder=10)

        plt.plot(self.N,self.res_mean_cosmogp,'b',linewidth=4,zorder=0)
        plt.scatter(self.N,self.res_mean_cosmogp,c='b',marker='s',s=75,zorder=10)

        plt.plot(self.N,self.res_mean_george,'k',linewidth=2,zorder=0)
        plt.scatter(self.N,self.res_mean_george,c='k',marker='s',s=75,zorder=10)

        plt.ylim(-0.2,0.2)
        plt.xlabel('number of point',fontsize=18)
        plt.ylabel('residual average',fontsize=18)


        plt.figure(figsize=(14,8))
        plt.subplots_adjust(hspace = 0.01)

        plt.subplot(2,1,1)

        plt.plot(self.N,np.ones_like(self.N),'k-.')

        plt.plot(self.N,self.hyp_skl[:,0],'r',linewidth=6,zorder=0,label='scikit learn ('+cosmogp_invertor+', kernel = %s)'%self.kernel)
        plt.scatter(self.N,self.hyp_skl[:,0],c='r',marker='s',s=75,zorder=10)
        plt.errorbar(self.N,self.hyp_skl[:,0], linestyle='', yerr=self.hyp_std_skl[:,0], elinewidth=6, ecolor='r',marker='.',zorder=0)
        
        plt.plot(self.N,self.hyp_cosmogp[:,0],'b',linewidth=4,zorder=0,label='cosmogp ('+cosmogp_invertor+', kernel = %s)'%self.kernel)
        plt.scatter(self.N,self.hyp_cosmogp[:,0],c='b',marker='s',s=75,zorder=10)
        plt.errorbar(self.N,self.hyp_cosmogp[:,0], linestyle='', yerr=self.hyp_std_cosmogp[:,0], elinewidth=4, ecolor='b',marker='.',zorder=0)
        
        plt.plot(self.N,self.hyp_george[:,0],'k',linewidth=2,zorder=0,label='george ('+george_invertor+', kernel = %s)'%self.kernel)
        plt.scatter(self.N,self.hyp_george[:,0],c='k',marker='s',s=75,zorder=10)
        plt.errorbar(self.N,self.hyp_george[:,0], linestyle='', yerr=self.hyp_std_george[:,0], elinewidth=2, ecolor='k',marker='.',zorder=0)
        
        plt.ylabel(r'$\sigma$',fontsize=18)
        plt.xticks([],[])
        plt.legend()
        
        plt.subplot(2,1,2)

        plt.plot(self.N,np.ones_like(self.N),'k-.')

        plt.plot(self.N,self.hyp_skl[:,1],'r',linewidth=6,zorder=0,label='scikit learn ('+cosmogp_invertor+', kernel = %s)'%self.kernel)
        plt.scatter(self.N,self.hyp_skl[:,1],c='r',marker='s',s=75,zorder=10)
        plt.errorbar(self.N,self.hyp_skl[:,1], linestyle='', yerr=self.hyp_std_skl[:,1], elinewidth=6, ecolor='r',marker='.',zorder=0)

        plt.plot(self.N,self.hyp_cosmogp[:,1],'b',linewidth=4,zorder=0)
        plt.scatter(self.N,self.hyp_cosmogp[:,1],c='b',marker='s',s=75,zorder=10)
        plt.errorbar(self.N,self.hyp_cosmogp[:,1], linestyle='', yerr=self.hyp_std_cosmogp[:,1], elinewidth=4, ecolor='b',marker='.',zorder=0)
        
        plt.plot(self.N,self.hyp_george[:,1],'k',linewidth=2,zorder=0)
        plt.scatter(self.N,self.hyp_george[:,1],c='k',marker='s',s=75,zorder=10)
        plt.errorbar(self.N,self.hyp_george[:,1], linestyle='', yerr=self.hyp_std_george[:,1], elinewidth=2, ecolor='k',marker='.',zorder=0)
        plt.xlabel('number of point',fontsize=18)
        plt.ylabel(r'$l$',fontsize=18)

        
if __name__=='__main__':

    test_gp = test_gaussian_process(kernel='rbf1d',noise=0.2,search_hyperparameter=True)
    test_gp.run_test()
    test_gp.plot_test_result()

""" Test of differents implementations of GP."""

import cosmogp
import george
import george.kernels as kernel_george
import sklearn.gaussian_process.kernels as skl_kernels
from sklearn.gaussian_process.gpr import GaussianProcessRegressor as skl_gpr
from svd_tmv import computeSVDInverse
from scipy.stats import norm as normal
from scipy.optimize import fmin_l_bfgs_b
import numpy as np
import pylab as plt
import time
import copy
import piff

def generate_data_test(number_of_data,number_of_point,
                       kernel_amplitude=1.,correlation_length=1.,
                       white_noise=0,noise=0.2,seed=1,dim=1):
    """
    Generation of simulated data for GP.
    Generating random gaussian fields in 1 dimensions (y depend
    only from the x axis). Used of Squared exponential kernel.
    """
    np.random.seed(seed)

    if dim == 1:
        

        x_axis = np.linspace(0,5,number_of_point)
        
        x = np.zeros((number_of_data,number_of_point))
        y = np.zeros((number_of_data,number_of_point))
    
        kernel = cosmogp.rbf_kernel_1d(x_axis, [kernel_amplitude,correlation_length],
                                       nugget=white_noise, floor=0.0, y_err=None)

        for i in range(number_of_data):
            if noise != 0:
                y[i] = np.random.multivariate_normal(np.zeros(len(x_axis)), kernel + (np.eye(len(kernel)) * noise**2))
            else:
                y[i] = np.random.multivariate_normal(np.zeros(len(x_axis)), kernel)
            x[i] = x_axis

    if dim == 2:

        x_axis = np.linspace(0,5,number_of_point)
        x_axis, y_axis = np.meshgrid(x_axis,x_axis)

        X = np.array([x_axis.reshape(number_of_point**2) ,
                      y_axis.reshape(number_of_point**2)]).T
        
        x = np.zeros((number_of_data,number_of_point**2,2))
        y = np.zeros((number_of_data,number_of_point**2))
    
        kernel = cosmogp.rbf_kernel_2d(X, [kernel_amplitude,correlation_length,correlation_length,0.],
                                       nugget=white_noise, floor=0.0, y_err=None)

        for i in range(number_of_data):
            if noise != 0:
                y[i] = np.random.multivariate_normal(np.zeros(len(y[i])), kernel + (np.eye(len(y[i])) * noise**2))
            else:
                y[i] = np.random.multivariate_normal(np.zeros(len(y[i])), kernel)
            x[i] = X

        
    return x,y,kernel


def cosmogp_gp(x, y, y_err=None, xpredict=None, kernel='rbf1d',
               hyperparameter_default=None, number_point_pull = 10,
               search_hyperparameter=False,
               substract_mean=False, svd=False):
    """
    cosmogp gaussian process.
    """

    print 'cosmogp'
    assert kernel in ['rbf1d','rbf2d'], 'not implemented kernel'


    if kernel == 'rbf1d':
        kernel = 'RBF1D'
        if hyperparameter_default is None:
            hyp = [1.,1.]
        else:
            hyp = hyperparameter_default

    if kernel == 'rbf2d':
        kernel = 'RBF2D'
        if hyperparameter_default is None:
            hyp = [1.,1.,1.,0.]
        else:
            hyp = hyperparameter_default

    timeA = time.time()
    gp = cosmogp.gaussian_process(y, x, y_err=y_err, kernel=kernel,
                                  substract_mean=substract_mean)

    if search_hyperparameter:
        C=time.time()
        gp.find_hyperparameters(hyperparameter_guess=hyp)
        D=time.time()
    else:
        gp.hyperparameters = hyp

    if xpredict is None:
        xpredict = x

    print 'I will do interpolation'
    gp.get_prediction(new_binning=xpredict,svd_method=svd)
    ynew = gp.Prediction[0]
    cov = gp.covariance_matrix[0]
    timeB = time.time()
    print 'finish interpolation'
    
    time_excution = timeB-timeA

    if search_hyperparameter:
        time_hyp = D-C
        time_excution -= time_hyp
    else:
        time_hyp = 0
                

    hyp = copy.deepcopy(gp.hyperparameters)
    
    resids = []
    stds = []

    conteur = 0


    
    for t in range(len(x)):

        if t > 0:
    
            filter_pull = np.array([True] * len(x))
            filter_pull[t] = False

            if y_err is None:
                yerr = None
                yerr_add = 0
            else:
                yerr = y_err[filter_pull]
                yerr_add = y_err[t]
            gp = cosmogp.gaussian_process(y[filter_pull], x[filter_pull], y_err=yerr, kernel=kernel,
                                          substract_mean=substract_mean)

            gp.hyperparameters = hyp

            gp.get_prediction(new_binning=x,svd_method=svd)

            resids.append(gp.Prediction[0][t] - y[t])
            stds.append(np.sqrt(abs(gp.covariance_matrix[0][t,t]+yerr_add**2)))

            conteur += 1
            
        if conteur == number_point_pull:
            break
    
    resids = np.array(resids)
    stds = np.array(stds) 

    return ynew,resids,stds,time_excution,gp.hyperparameters,time_hyp


def george_gp(x, y, y_err=None, xpredict=None, kernel='rbf1d',
              hyperparameter_default=None, number_point_pull = 10,
              search_hyperparameter=False,HODLR=False):
    """
    george gaussian process.
    """
    print 'george'
    assert kernel in ['rbf1d','rbf2d'], 'not implemented kernel'

    if kernel == 'rbf1d' :

        if hyperparameter_default is None:
            hyp = [1.,1.]
        else:
            hyp = hyperparameter_default

        kernel = hyp[0]**2 * kernel_george.ExpSquaredKernel(hyp[1]**2)

    if kernel == 'rbf2d':
        if hyperparameter_default is None:
            hyp = [1.,1.,1.,0.]
        else:
            hyp = hyperparameter_default
    
        kernel = hyp[0] * kernel_george.ExpSquaredKernel(metric=[[hyp[1]**2, hyp[3]], [hyp[3], hyp[2]**2]], ndim=2)

    import scipy.optimize as op

    def nll(p):
        #p = np.log(p)
        gp.set_parameter_vector(p)
        ll = gp.log_likelihood(y, quiet=True)
        return -ll if np.isfinite(ll) else 1e25

    def grad_nll(p):
        gp.set_parameter_vector(p)
        return -gp.grad_log_likelihood(y, quiet=True)


    def obj_func(theta, eval_gradient=True):
        if eval_gradient:
            lml, grad = nll(theta), grad_nll(theta)
            return lml, grad
        else:
            return nll(theta)

    # First optimize starting from theta specified in kernel
    #optima = [(self._constrained_optimization(obj_func,
    #                                          self.kernel_.theta,
    #                                          self.kernel_.bounds))]
                                                               
                    
    if y_err is None:
        y_err = np.zeros(len(y))

    if HODLR:
        gp = george.GP(kernel, solver=george.HODLRSolver)
    else:
        gp = george.GP(kernel)
                       
    gp.compute(x, y_err)

    timeA = time.time()

    if search_hyperparameter:
        C = time.time()
        p0 = gp.get_parameter_vector()
        #results = op.minimize(nll, p0, jac=grad_nll, method="L-BFGS-B")
        bounds = np.array([[-11.51292546,  11.51292546],
                           [ -5.        ,   5.        ],
                           [ -5.        ,   5.        ],
                           [ -5.        ,   5.        ]])
        theta_opt, func_min, convergence_dict = fmin_l_bfgs_b(obj_func, p0, bounds=bounds)
        print convergence_dict
        results = {'x':theta_opt}
        #results = op.minimize(nll, p0, method="L-BFGS-B")
        if len(p0) == 2:
            hyper_output = np.sqrt(np.exp(results['x']))
        else:
            L = np.zeros((2,2))
            sigma = np.sqrt(2.*np.exp(results['x'][0]))
            L[0,0] = np.exp(results['x'][1])
            L[1,1] = np.exp(results['x'][3])
            L[1,0] = results['x'][2]
            Chol = L.dot(L.T)
            hyper_output = np.array([sigma,Chol[0,0],Chol[1,1],Chol[1,0]])
    else:
        hyper_output = hyp
    
    if xpredict is None:
        xpredict = x

    D = time.time() 
    mu, cov = gp.predict(y, xpredict)

    timeB = time.time()

    time_excution = timeB - timeA

    if search_hyperparameter:
        time_hyp = D-C
        time_excution -= time_hyp
    else:
        time_hyp = 0
        
    resids = []
    stds = []

    conteur = 0

    if search_hyperparameter:
        gp.set_parameter_vector(results['x'])
    
    for t in range(len(x)):

        if t > 0:

            filter_pull = np.array([True] * len(x))
            filter_pull[t] = False

            gp.compute(x[filter_pull], y_err[filter_pull])
            mu_, cov_ = gp.predict(y[filter_pull], x)

            resids.append(mu_[t] - y[t])
            stds.append(np.sqrt(abs(cov_[t, t]+y_err[t]**2)))

            conteur += 1
            
        if conteur == number_point_pull:
            break
            
    resids = np.array(resids)
    stds = np.array(stds)

    return mu,resids,stds,time_excution,hyper_output,time_hyp
    #return mu,cov,resids,stds,time_excution,gp


def scikitlearn_gp(x, y, y_err=None, xpredict=None, kernel='rbf1d',
                   hyperparameter_default=None, number_point_pull = 10,
                   search_hyperparameter=False):
    """
    scikit learn gaussian process.
    """
    print 'sklearn'
    if kernel == 'rbf1d':
        x = x.reshape(len(x),1)
    y = y.reshape(len(y),1)

    assert kernel in ['rbf1d','rbf2d'], 'not implemented kernel'

    if kernel == 'rbf1d' :

        if hyperparameter_default is None:
            hyp = [1.,1.]
        else:
            hyp = hyperparameter_default

        
        ker = skl_kernels.ConstantKernel(constant_value = hyp[0]**2) * skl_kernels.RBF(length_scale = hyp[1])

    if kernel == 'rbf2d':

        if hyperparameter_default is None:
            hyp = [1.,1.,1.,0.]
        else:
            hyp = hyperparameter_default
        
        inv_l = np.array(([hyp[2]**2,-hyp[3]],
                          [-hyp[3],hyp[1]**2]))

        inv_l *= 1./(((hyp[1]**2)*(hyp[2]**2))-hyp[3]**2)
                
        ker = hyp[0]**2 * piff.AnisotropicRBF(invLam=inv_l)

    if y_err is None:
        alpha = 1e-10
    else:
        alpha = y_err**2

    if not search_hyperparameter:
        gpr = skl_gpr(ker, alpha=alpha, optimizer=None)
    else:
        gpr = skl_gpr(ker, alpha=alpha)

    timeA = time.time()
    C = time.time()    
    gpr.fit(x, y)
    D = time.time()
    if len(gpr.kernel_.theta)==2:
        hyp = np.sqrt(np.exp(gpr.kernel_.theta))
    else:
        sigma = np.sqrt(np.exp(gpr.kernel_.theta[0]))
        TT = np.exp(gpr.kernel_.theta[1])
        TT = np.exp(gpr.kernel_.theta[2])
        TTT = gpr.kernel_.theta[3]
        Lchol = np.array(([TT,TTT],[0,TT]))
        LLT = np.linalg.inv(Lchol.dot(Lchol.T))
        hyp = np.array([sigma,LLT[0,0],LLT[1,1],LLT[0,1]])

    if xpredict is None:
        xpredict = x

    ynew, cov = gpr.predict(xpredict, return_cov=True)

    timeB = time.time()

    time_excution = timeB-timeA

    if search_hyperparameter:
        time_hyp = D-C
        time_excution -= time_hyp
    else:
        time_hyp = 0

    resids = []
    stds = []

    conteur = 0
    
    for t in range(len(x)):

        if t > 0 :

            filter_pull = np.array([True] * len(x))
            filter_pull[t] = False

            if y_err is not None:
                gpr = skl_gpr(ker, alpha=alpha[filter_pull], optimizer=None)
            else:
                gpr = skl_gpr(ker, alpha=alpha, optimizer=None)
            
            gpr.fit(x[filter_pull], y[filter_pull])
            ypredict, cov = gpr.predict(x, return_cov=True)
            resids.append(ypredict.T[0][t] - y[t,0])

            if y_err is not None:
                stds.append(np.sqrt(abs(cov[t, t])+alpha[t]))
            else:
                stds.append(np.sqrt(abs(cov[t, t])+alpha))

            conteur +=1
            
        if conteur == number_point_pull:
            break
            
    resids = np.array(resids)
    stds  = np.array(stds)

    return ynew.T[0],resids,stds,time_excution,hyp,time_hyp


class test_gaussian_process:

    def __init__(self,kernel='rbf1d', noise = 0.2, search_hyperparameter=False, HODLR=False,number_point_pull = 10, Number_data=20):
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
        self.HODLR = HODLR

        if noise !=0.:
            if self.search_hyperparameter:
                self.N = np.logspace(1.1,1.7,10).astype(int)
            else:
                self.N = np.logspace(1.1,3,10).astype(int)
        else:
            self.N = np.linspace(5,20,30).astype(int)
        self.N1 = np.ones_like(self.N)
            
        if kernel == 'rbf1d':
            self.ndim = 1
            self.nhyp = 2
            self.hyp_default = [1.,1.]
        if kernel == 'rbf2d':
            self.ndim = 2
            self.hyp_default = [1.,1.,1.,0.]
            self.nhyp = 4
            self.N=np.linspace(5,30,6).astype(int)
            self.N1 = self.N
            
        self.Number_data = Number_data
        self.number_point_pull = number_point_pull

        self.noise = noise

        self.ynew_skl = []
        self.resids_skl = []
        self.stds_skl = []
        self.time_skl = np.zeros(len(self.N))
        self.time_hyp_skl = np.zeros(len(self.N))
        self.hyp_skl = np.zeros((len(self.N),self.nhyp))
        self.hyp_std_skl = np.zeros((len(self.N),self.nhyp))

        self.ynew_cosmogp = []
        self.resids_cosmogp = []
        self.stds_cosmogp = []
        self.time_cosmogp = np.zeros(len(self.N))
        self.time_hyp_cosmogp = np.zeros(len(self.N))
        self.hyp_cosmogp = np.zeros((len(self.N),self.nhyp))
        self.hyp_std_cosmogp = np.zeros((len(self.N),self.nhyp))

        self.ynew_george = []
        self.resids_george = []
        self.stds_george = []
        self.time_george = np.zeros(len(self.N))
        self.time_hyp_george = np.zeros(len(self.N))
        self.hyp_george = np.zeros((len(self.N),self.nhyp))
        self.hyp_std_george = np.zeros((len(self.N),self.nhyp))


    def run_test(self):

        for i in range(len(self.N)):

            print i

            x, y, kernel = generate_data_test(self.Number_data,self.N[i],
                                              kernel_amplitude=1.,correlation_length=1.,
                                              white_noise=0,noise=self.noise,seed=1, dim=self.ndim)

            if self.noise != 0:
                self.y_err = np.ones(len(y[0])) * self.noise
            else:
                self.y_err = None

            ynew_skl = np.zeros(self.N[i]*self.N1[i]*self.Number_data)
            resids_skl = np.zeros(self.number_point_pull*self.Number_data)
            stds_skl = np.zeros(self.number_point_pull*self.Number_data)
            t_skl = np.zeros(self.Number_data)
            t_hyp_skl = np.zeros(self.Number_data)
            hyp_skl = np.zeros((self.Number_data,self.nhyp))

            ynew_cosmogp = np.zeros(self.N[i]*self.N1[i]*self.Number_data)
            resids_cosmogp = np.zeros(self.number_point_pull*self.Number_data)
            stds_cosmogp = np.zeros(self.number_point_pull*self.Number_data)
            t_cosmogp = np.zeros(self.Number_data)
            t_hyp_cosmogp = np.zeros(self.Number_data)
            hyp_cosmogp = np.zeros((self.Number_data,self.nhyp))

            ynew_george = np.zeros(self.N[i]*self.N1[i]*self.Number_data)
            resids_george = np.zeros(self.number_point_pull*self.Number_data)
            stds_george = np.zeros(self.number_point_pull*self.Number_data)
            t_george = np.zeros(self.Number_data)
            t_hyp_george = np.zeros(self.Number_data)
            hyp_george = np.zeros((self.Number_data,self.nhyp))

            t = 0

            for j in range(self.Number_data):

                #scikit learn gaussian process 

                ynew,resids,stds,time_excution,hyp,thyp = scikitlearn_gp(x[j],y[j],y_err=self.y_err,xpredict=None,kernel=self.kernel,
                                                                         hyperparameter_default=self.hyp_default,number_point_pull=self.number_point_pull,
                                                                         search_hyperparameter=self.search_hyperparameter)

                ynew_skl[t:t+len(ynew)] = ynew
                resids_skl[t:t+self.number_point_pull] = resids
                stds_skl[t:t+self.number_point_pull] = stds
                t_skl[j] = time_excution
                t_hyp_skl[j] = thyp
                hyp_skl[j] = hyp

                #cosmogp gaussian process

                #george gaussian process

                ynew,resids,stds,time_excution,hyp,thyp = george_gp(x[j],y[j],y_err=self.y_err,xpredict=None,kernel=self.kernel,
                                                                    hyperparameter_default=self.hyp_default,number_point_pull=self.number_point_pull,
                                                                    search_hyperparameter=self.search_hyperparameter,HODLR=self.HODLR)
                ynew_george[t:t+len(ynew)] = ynew
                resids_george[t:t+self.number_point_pull] = resids
                stds_george[t:t+self.number_point_pull] = stds
                t_george[j] = time_excution
                t_hyp_george[j] = thyp
                hyp_george[j] = hyp

                #ynew,resids,stds,time_excution,hyp,thyp = cosmogp_gp(x[j],y[j],y_err=self.y_err,xpredict=None,kernel=self.kernel,
                #                                                     hyperparameter_default=self.hyp_default,number_point_pull=self.number_point_pull,
                #                                                     search_hyperparameter=self.search_hyperparameter)
                #ynew_cosmogp[t:t+len(ynew)] = ynew
                #resids_cosmogp[t:t+self.number_point_pull] = resids
                #stds_cosmogp[t:t+self.number_point_pull] = stds
                #t_cosmogp[j] = time_excution
                #t_hyp_cosmogp[j] = thyp
                #hyp_cosmogp[j] = hyp

                t += self.number_point_pull
                
            self.ynew_skl.append(ynew_skl)
            self.resids_skl.append(resids_skl)
            self.stds_skl.append(stds_skl)            
            self.time_skl[i] = np.mean(t_skl)
            self.time_hyp_skl[i] = np.mean(t_hyp_skl)
            self.hyp_skl[i] = np.mean(hyp_skl,axis=0)
            self.hyp_std_skl[i] = np.std(hyp_skl,axis=0,ddof=1.5)


            self.ynew_cosmogp.append(ynew_cosmogp)
            self.resids_cosmogp.append(resids_cosmogp)
            self.stds_cosmogp.append(stds_cosmogp)            
            self.time_cosmogp[i] = np.mean(t_cosmogp)
            self.time_hyp_cosmogp[i] = np.mean(t_hyp_cosmogp)            
            self.hyp_cosmogp[i,0] = np.mean(hyp_cosmogp[:,0])
            self.hyp_std_cosmogp[i,0] = np.std(hyp_cosmogp[:,0],ddof=1.5)
            self.hyp_cosmogp[i,1] = np.mean(hyp_cosmogp[:,1])
            self.hyp_std_cosmogp[i,1] = np.std(hyp_cosmogp[:,1],ddof=1.5)

            self.ynew_george.append(ynew_george)
            self.resids_george.append(resids_george)
            self.stds_george.append(stds_george)            
            self.time_george[i] = np.mean(t_george)
            self.time_hyp_george[i] = np.mean(t_hyp_george)
            self.hyp_george[i] = np.mean(hyp_george,axis=0)
            self.hyp_std_george[i] = np.std(hyp_george,axis=0,ddof=1.5)


    def plot_test_result(self,skl_invertor = 'Choleski (scipy)', cosmogp_invertor = 'LDL (TMV)', george_invertor = 'Choleski (scipy)'):
        """
        Plot performances of differents gp algorithm.
        """
        if self.HODLR:
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

        self.number_points = np.zeros(len(self.N))

        for i in range(len(self.N)):
            self.pull_mean_skl[i], self.pull_std_skl[i] = normal.fit(self.resids_skl[i]/self.stds_skl[i])
            self.res_mean_skl[i], self.res_std_skl[i] = normal.fit(self.resids_skl[i])

            self.pull_mean_cosmogp[i], self.pull_std_cosmogp[i] = normal.fit(self.resids_cosmogp[i]/self.stds_cosmogp[i])
            self.res_mean_cosmogp[i], self.res_std_cosmogp[i] = normal.fit(self.resids_cosmogp[i])

            self.pull_mean_george[i], self.pull_std_george[i] = normal.fit(self.resids_george[i]/self.stds_george[i])
            self.res_mean_george[i], self.res_std_george[i] = normal.fit(self.resids_george[i])

            self.number_points[i] = len(self.resids_george[i])

        plt.figure(figsize=(14,8))
        plt.plot(self.N*self.N1,self.time_skl,'r-s',linewidth=3,markersize=10,label='scikit learn ('+skl_invertor+', kernel = %s)'%self.kernel)
        plt.plot(self.N*self.N1,self.time_cosmogp,'b-s',linewidth=3,markersize=10,label='cosmogp ('+cosmogp_invertor+', kernel = %s)'%self.kernel)
        plt.plot(self.N*self.N1,self.time_george,'k-s',linewidth=3,markersize=10,label='george ('+george_invertor+', kernel = %s)'%self.kernel)
        plt.ylabel('Time of excution (s): interpolation',fontsize=18)
        plt.xlabel('number of points',fontsize=18)
        plt.legend(loc=2)

        if self.search_hyperparameter:
            plt.figure(figsize=(14,8))
            plt.plot(self.N*self.N1,self.time_hyp_skl,'r-s',linewidth=3,markersize=10,label='scikit learn ('+skl_invertor+', kernel = %s)'%self.kernel)
            plt.plot(self.N*self.N1,self.time_hyp_cosmogp,'b-s',linewidth=3,markersize=10,label='cosmogp ('+cosmogp_invertor+', kernel = %s)'%self.kernel)
            plt.plot(self.N*self.N1,self.time_hyp_george,'k-s',linewidth=3,markersize=10,label='george ('+george_invertor+', kernel = %s)'%self.kernel)
            plt.ylabel('Time of excution (s): hyperparameter search',fontsize=18)
            plt.xlabel('number of points',fontsize=18)
            plt.legend(loc=2)

        plt.figure(figsize=(14,8))
        plt.subplots_adjust(hspace = 0.01)

        plt.subplot(2,1,1)

        plt.plot(self.N*self.N1,np.ones_like(self.N),'k-.')
        
        plt.plot(self.N*self.N1,self.pull_std_skl,'r',linewidth=6,label='scikit learn ('+skl_invertor+', kernel = %s)'%self.kernel,zorder=5)
        plt.scatter(self.N*self.N1,self.pull_std_skl,c='r',marker='s',s=75,zorder=10)
        plt.errorbar(self.N*self.N1,self.pull_std_skl, linestyle='', yerr=self.pull_std_skl/(np.sqrt(2.*(self.number_points-1.))), elinewidth=6, ecolor='r',marker='.',zorder=1)

        plt.plot(self.N*self.N1,self.pull_std_cosmogp,'b',linewidth=4,label='cosmogp ('+cosmogp_invertor+', kernel = %s)'%self.kernel,zorder=5)
        plt.scatter(self.N*self.N1,self.pull_std_cosmogp,c='b',marker='s',s=75,zorder=10)
        plt.errorbar(self.N*self.N1,self.pull_std_cosmogp, linestyle='', yerr=self.pull_std_cosmogp/(np.sqrt(2.*(self.number_points-1.))), elinewidth=4, ecolor='b',marker='.',zorder=2)

        plt.plot(self.N*self.N1,self.pull_std_george,'k',linewidth=2,label='george ('+george_invertor+', kernel = %s)'%self.kernel,zorder=5)
        plt.scatter(self.N*self.N1,self.pull_std_george,c='k',marker='s',s=75,zorder=10)
        plt.errorbar(self.N*self.N1,self.pull_std_george, linestyle='', yerr=self.pull_std_george/(np.sqrt(2.*(self.number_points-1.))), elinewidth=2, ecolor='k',marker='.',zorder=3)
        plt.xticks([],[])
        plt.ylabel('pull STD',fontsize=18)

        plt.ylim(0,2)
        
        plt.legend()

        plt.subplot(2,1,2)

        plt.plot(self.N*self.N1,np.zeros_like(self.N),'k-.')
        
        plt.plot(self.N*self.N1,self.pull_mean_skl,'r',linewidth=6,zorder=5)
        plt.scatter(self.N*self.N1,self.pull_mean_skl,c='r',marker='s',s=75,zorder=10)
        plt.errorbar(self.N*self.N1,self.pull_mean_skl, linestyle='', yerr=self.pull_std_skl/(np.sqrt(self.number_points)), elinewidth=6, ecolor='r',marker='.',zorder=1)

        plt.plot(self.N*self.N1,self.pull_mean_cosmogp,'b',linewidth=4,zorder=5)
        plt.scatter(self.N*self.N1,self.pull_mean_cosmogp,c='b',marker='s',s=75,zorder=10)
        plt.errorbar(self.N*self.N1,self.pull_mean_cosmogp, linestyle='', yerr=self.pull_std_cosmogp/(np.sqrt(self.number_points)), elinewidth=4, ecolor='b',marker='.',zorder=2)

        plt.plot(self.N*self.N1,self.pull_mean_george,'k',linewidth=2,zorder=5)
        plt.scatter(self.N*self.N1,self.pull_mean_george,c='k',marker='s',s=75,zorder=10)
        plt.errorbar(self.N*self.N1,self.pull_mean_george, linestyle='', yerr=self.pull_std_george/(np.sqrt(self.number_points)), elinewidth=2, ecolor='k',marker='.',zorder=3)

        plt.xlabel('number of points',fontsize=18)
        plt.ylabel('pull average',fontsize=18)

        
        plt.figure(figsize=(14,8))
        plt.subplots_adjust(hspace = 0.01)

        plt.subplot(2,1,1)
        plt.plot(self.N*self.N1,self.res_std_skl,'r',linewidth=6,label='scikit learn ('+skl_invertor+', kernel = %s)'%self.kernel,zorder=0)
        plt.scatter(self.N*self.N1,self.res_std_skl,c='r',marker='s',s=75,zorder=10)
        plt.errorbar(self.N*self.N1,self.res_std_skl, linestyle='', yerr=self.res_std_skl/(np.sqrt(2.*(self.number_points-1.))), elinewidth=6, ecolor='r',marker='.',zorder=1)

        plt.plot(self.N*self.N1,self.res_std_cosmogp,'b',linewidth=4,label='cosmogp ('+cosmogp_invertor+', kernel = %s)'%self.kernel,zorder=5)
        plt.scatter(self.N*self.N1,self.res_std_cosmogp,c='b',marker='s',s=75,zorder=10)
        plt.errorbar(self.N*self.N1,self.res_std_cosmogp, linestyle='', yerr=self.res_std_cosmogp/(np.sqrt(2.*(self.number_points-1.))), elinewidth=4, ecolor='b',marker='.',zorder=2)

        plt.plot(self.N*self.N1,self.res_std_george,'k',linewidth=2,label='george ('+george_invertor+', kernel = %s)'%self.kernel,zorder=5)
        plt.scatter(self.N*self.N1,self.res_std_george,c='k',marker='s',s=75,zorder=10)
        plt.errorbar(self.N*self.N1,self.res_std_george, linestyle='', yerr=self.res_std_george/(np.sqrt(2.*(self.number_points-1.))), elinewidth=2, ecolor='k',marker='.',zorder=3)

        plt.xticks([],[])
        plt.ylabel('residual STD',fontsize=18)
        plt.legend()

        plt.subplot(2,1,2)

        plt.plot(self.N*self.N1,np.zeros_like(self.N),'k-.')

        plt.plot(self.N*self.N1,self.res_mean_skl,'r',linewidth=6,zorder=5)
        plt.scatter(self.N*self.N1,self.res_mean_skl,c='r',marker='s',s=75,zorder=10)
        plt.errorbar(self.N*self.N1,self.res_mean_skl, linestyle='', yerr=self.res_std_skl/(np.sqrt(self.number_points)), elinewidth=6, ecolor='r',marker='.',zorder=1)

        plt.plot(self.N*self.N1,self.res_mean_cosmogp,'b',linewidth=4,zorder=5)
        plt.scatter(self.N*self.N1,self.res_mean_cosmogp,c='b',marker='s',s=75,zorder=10)
        plt.errorbar(self.N*self.N1,self.res_mean_cosmogp, linestyle='', yerr=self.res_std_cosmogp/(np.sqrt(self.number_points)), elinewidth=4, ecolor='b',marker='.',zorder=2)

        plt.plot(self.N*self.N1,self.res_mean_george,'k',linewidth=2,zorder=5)
        plt.scatter(self.N*self.N1,self.res_mean_george,c='k',marker='s',s=75,zorder=10)
        plt.errorbar(self.N*self.N1,self.res_mean_george, linestyle='', yerr=self.res_std_george/(np.sqrt(self.number_points)), elinewidth=2, ecolor='k',marker='.',zorder=3)

        plt.ylim(-0.2,0.2)
        plt.xlabel('number of point',fontsize=18)
        plt.ylabel('residual average',fontsize=18)


        if self.search_hyperparameter:
        
            plt.figure(figsize=(14,8))
            plt.subplots_adjust(hspace = 0.01)

            plt.subplot(2,1,1)
            
            plt.plot(self.N*self.N1,np.ones_like(self.N),'k-.')

            plt.plot(self.N*self.N1,self.hyp_skl[:,0],'r',linewidth=6,zorder=0,label='scikit learn ('+cosmogp_invertor+', kernel = %s)'%self.kernel)
            plt.scatter(self.N*self.N1,self.hyp_skl[:,0],c='r',marker='s',s=75,zorder=10)
            plt.errorbar(self.N*self.N1,self.hyp_skl[:,0], linestyle='', yerr=self.hyp_std_skl[:,0], elinewidth=6, ecolor='r',marker='.',zorder=0)
        
            plt.plot(self.N*self.N1,self.hyp_cosmogp[:,0],'b',linewidth=4,zorder=0,label='cosmogp ('+cosmogp_invertor+', kernel = %s)'%self.kernel)
            plt.scatter(self.N*self.N1,self.hyp_cosmogp[:,0],c='b',marker='s',s=75,zorder=10)
            plt.errorbar(self.N*self.N1,self.hyp_cosmogp[:,0], linestyle='', yerr=self.hyp_std_cosmogp[:,0], elinewidth=4, ecolor='b',marker='.',zorder=0)
        
            plt.plot(self.N*self.N1,self.hyp_george[:,0],'k',linewidth=2,zorder=0,label='george ('+george_invertor+', kernel = %s)'%self.kernel)
            plt.scatter(self.N*self.N1,self.hyp_george[:,0],c='k',marker='s',s=75,zorder=10)
            plt.errorbar(self.N*self.N1,self.hyp_george[:,0], linestyle='', yerr=self.hyp_std_george[:,0], elinewidth=2, ecolor='k',marker='.',zorder=0)
        
            plt.ylabel(r'$\sigma$',fontsize=18)
            plt.xticks([],[])
            plt.legend()
            plt.ylim(0,2)
            
            plt.subplot(2,1,2)
        
            plt.plot(self.N,np.ones_like(self.N),'k-.')
            
            plt.plot(self.N*self.N1,self.hyp_skl[:,1],'r',linewidth=6,zorder=0,label='scikit learn ('+cosmogp_invertor+', kernel = %s)'%self.kernel)
            plt.scatter(self.N*self.N1,self.hyp_skl[:,1],c='r',marker='s',s=75,zorder=10)
            plt.errorbar(self.N*self.N1,self.hyp_skl[:,1], linestyle='', yerr=self.hyp_std_skl[:,1], elinewidth=6, ecolor='r',marker='.',zorder=0)

            plt.plot(self.N*self.N1,self.hyp_cosmogp[:,1],'b',linewidth=4,zorder=0)
            plt.scatter(self.N*self.N1,self.hyp_cosmogp[:,1],c='b',marker='s',s=75,zorder=10)
            plt.errorbar(self.N*self.N1,self.hyp_cosmogp[:,1], linestyle='', yerr=self.hyp_std_cosmogp[:,1], elinewidth=4, ecolor='b',marker='.',zorder=0)
        
            plt.plot(self.N*self.N1,self.hyp_george[:,1],'k',linewidth=2,zorder=0)
            plt.scatter(self.N*self.N1,self.hyp_george[:,1],c='k',marker='s',s=75,zorder=10)
            plt.errorbar(self.N*self.N1,self.hyp_george[:,1], linestyle='', yerr=self.hyp_std_george[:,1], elinewidth=2, ecolor='k',marker='.',zorder=0)
            plt.xlabel('number of point',fontsize=18)
            plt.ylabel(r'$l$',fontsize=18)
            plt.ylim(0,2)

            if len(self.hyp_skl[0])!=2:

                plt.figure(figsize=(14,8))
                plt.subplots_adjust(hspace = 0.01)

                plt.subplot(2,1,1)
                
                plt.plot(self.N,np.ones_like(self.N),'k-.')
            
                plt.plot(self.N*self.N1,self.hyp_skl[:,2],'r',linewidth=6,zorder=0,label='scikit learn ('+cosmogp_invertor+', kernel = %s)'%self.kernel)
                plt.scatter(self.N*self.N1,self.hyp_skl[:,2],c='r',marker='s',s=75,zorder=10)
                plt.errorbar(self.N*self.N1,self.hyp_skl[:,2], linestyle='', yerr=self.hyp_std_skl[:,2], elinewidth=6, ecolor='r',marker='.',zorder=0)

                plt.plot(self.N*self.N1,self.hyp_george[:,2],'k',linewidth=2,zorder=0)
                plt.scatter(self.N*self.N1,self.hyp_george[:,2],c='k',marker='s',s=75,zorder=10)
                plt.errorbar(self.N*self.N1,self.hyp_george[:,2], linestyle='', yerr=self.hyp_std_george[:,2], elinewidth=2, ecolor='k',marker='.',zorder=0)
                plt.xlabel('number of point',fontsize=18)
                plt.ylabel(r'$l_y$',fontsize=18)
                plt.ylim(0,2)


                plt.subplot(2,1,2)
                
                plt.plot(self.N,np.zeros_like(self.N),'k-.')
            
                plt.plot(self.N*self.N1,self.hyp_skl[:,3],'r',linewidth=6,zorder=0,label='scikit learn ('+cosmogp_invertor+', kernel = %s)'%self.kernel)
                plt.scatter(self.N*self.N1,self.hyp_skl[:,3],c='r',marker='s',s=75,zorder=10)
                plt.errorbar(self.N*self.N1,self.hyp_skl[:,3], linestyle='', yerr=self.hyp_std_skl[:,3], elinewidth=6, ecolor='r',marker='.',zorder=0)

                plt.plot(self.N*self.N1,self.hyp_george[:,3],'k',linewidth=2,zorder=0)
                plt.scatter(self.N*self.N1,self.hyp_george[:,3],c='k',marker='s',s=75,zorder=10)
                plt.errorbar(self.N*self.N1,self.hyp_george[:,3], linestyle='', yerr=self.hyp_std_george[:,3], elinewidth=2, ecolor='k',marker='.',zorder=0)
                plt.xlabel('number of point',fontsize=18)
                plt.ylabel(r'$l_{xy}$',fontsize=18)
                plt.ylim(-0.5,0.5)
        
if __name__=='__main__':

    ##1d (squared exponential kernel) without noise, with fixed hyperparameter 

    #test_gp = test_gaussian_process(kernel='rbf1d',noise=0.0,
    #                                search_hyperparameter=False,
    #                                number_point_pull=4,
    #                                Number_data=50)
    #test_gp.run_test()
    #test_gp.plot_test_result()

    ##1d (squared exponential kernel) with noise, with fixed hyperparameter 

    #test_gp = test_gaussian_process(kernel='rbf1d',noise=0.2,
    #                                search_hyperparameter=False,
    #                                number_point_pull=10,
    #                                Number_data=20,HODLR=False)
    #test_gp.run_test()
    #test_gp.plot_test_result()


    ##1d (squared exponential kernel) with noise, with free hyperparameter 

    #test_gp = test_gaussian_process(kernel='rbf1d',noise=0.2,
    #                                search_hyperparameter=True,
    #                                number_point_pull=5,
    #                                Number_data=50,HODLR=False)
    #test_gp.run_test()
    #test_gp.plot_test_result()

    ##1d (squared exponential kernel) with noise, with fixed hyperparameter 

    test_gp = test_gaussian_process(kernel='rbf2d',noise=0.2,
                                    search_hyperparameter=True,
                                    number_point_pull=2,
                                    Number_data=3,HODLR=False)
    test_gp.run_test()
    test_gp.plot_test_result()

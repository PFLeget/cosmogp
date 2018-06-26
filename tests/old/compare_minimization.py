import numpy as np
import pylab as plt
import cosmogp
import george
from test_gp_algorithm import generate_data_test


x,y,kernel,det_kernel = generate_data_test(20,30,
                                           kernel_amplitude=2.,correlation_length=2.5,
                                           white_noise=0,noise=0.2,seed=1)

#########
#cosmogp#
#########

gpc = cosmogp.gaussian_process(y[0], x[0], y_err=np.ones(len(x[0]))*0.2, kernel='RBF1D',
                               substract_mean=False)

gpc.find_hyperparameters()

gpc.get_prediction(new_binning=xpredict,svd_method=svd)
ynew = gpc.Prediction[0]
cov = gpc.covariance_matrix[0]
            
########
#george#
########

import scipy.optimize as op

def nll(p):
    #p = np.log(p)
    gp.set_parameter_vector(p)
    ll = gp.log_likelihood(y, quiet=True)
    return -ll if np.isfinite(ll) else 1e25

def grad_nll(p):
    gp.set_parameter_vector(p)
    return -gp.grad_log_likelihood(y, quiet=True)


kernel = 1**2 * kernel_george.ExpSquaredKernel(1**2)


gp = george.GP(kernel)
gp.compute(x, y_err)

p0 = gp.get_parameter_vector()
hyper_output = op.fmin(nll, p0, disp=False)
hyper_output = np.sqrt(np.exp(hyper_output))
#hyper_output = np.sqrt(np.exp(results['x']))

xpredict = x

mu, cov = gp.predict(y, xpredict)

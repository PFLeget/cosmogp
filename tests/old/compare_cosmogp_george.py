"""Compare differences between both. """

from test_gp_algorithm import generate_data_test
import numpy as np
import pylab as plt
import cosmogp
import george


####### compute data #########

np.random.seed(1)

n_object = 15
total_point = 0

x = []
y = []

for i in range(n_object): # generate object not on the same grid of observation
    n_point = 20 #int(np.random.uniform(25,30)) # random number of points for one object
    total_point += n_point
    grid = np.linspace(0,5,n_point) # fixed grid where gp will be generated
    k = cosmogp.rbf_kernel_1d(grid,np.array([1.,1.])) # build the kernel with fixed hyperparameter
    yy = np.random.multivariate_normal(np.zeros_like(grid), k) # generation of gaussian process
    x.append(grid)
    y.append(yy)
    

######### george ############

kernel = 1.**2 * george.kernels.ExpSquaredKernel(1.**2)
gpg = george.GP(kernel, fit_kernel=False)
gpg.compute(x[0], np.zeros(len(x[0])))
ygeorge, cov_george = gpg.predict(y[0], x[0])


from scipy.linalg import cholesky, cho_solve

r_george = np.ascontiguousarray(gpg._check_dimensions(y[0]) -
                                gpg._call_mean(x[0]),
                                dtype=np.float64)

kgeorge = gpg.kernel.get_value(x[0].reshape(len(x[0]),1))
_factor_george = (cholesky(kgeorge, overwrite_a=True, lower=False), False)
alpha_george = cho_solve(_factor_george, r_george, overwrite_b=True)
                                    
xs_george = gpg.parse_samples(x[0])
Kxs_george = gpg.kernel.get_value(xs_george, gpg._x)
pred_george = np.dot(Kxs_george, alpha_george) + gpg._call_mean(xs_george)


######### cosmogp ############

gpc = cosmogp.gaussian_process(y[0], x[0], kernel='RBF1D',
                               substract_mean=False)
gpc.hyperparameters = [1.,1.]
gpc.get_prediction(new_binning=x[0], svd_method=False)
print gpc.y0
ycosmogp = gpc.Prediction[0]
cov_cosmogp = gpc.covariance_matrix[0]

from svd_tmv import computeLDLInverse as chol

Hcosmogp = gpc.kernel(x[0], [1.,1.], new_x=x[0])
Kcosmogp = gpc.kernel(x[0], [1.,1.])
Kcosmogp_inv = chol(Kcosmogp,return_logdet=False)
y0_cosmogp = 0.
ycosmogp_ket = (y[0]-y0_cosmogp).reshape(len(y[0]),1)
Bcosmogp = np.dot(Kcosmogp_inv,ycosmogp_ket)
pred_cosmogp = np.dot(Hcosmogp,Bcosmogp).T[0]
pred_cosmogp += y0_cosmogp


######### sklearn ############

import sklearn.gaussian_process.kernels as skl_kernels
from sklearn.gaussian_process.gpr import GaussianProcessRegressor as skl_gpr

ker = skl_kernels.ConstantKernel(constant_value = 1.**2) * skl_kernels.RBF(length_scale = 1.**2)

xx = x[0].reshape(len(x[0]),1)
yy = y[0].reshape(len(y[0]),1)
        
gpr = skl_gpr(ker, optimizer=None)
gpr.fit(xx, yy)
yskl, cov_skl = gpr.predict(xx, return_cov=True)
yskl = yskl.T[0]

Kskl = gpr.kernel_(gpr.X_train_)
L_skl = cholesky(Kskl, lower=True)
alpha_skl = cho_solve((L_skl, True), gpr.y_train_)

K_trans = gpr.kernel_(xx, gpr.X_train_)
y_mean = K_trans.dot(alpha_skl)  # Line 4 (y_mean = f_star)
y_mean = gpr.y_train_mean + y_mean  # undo normal.

v = cho_solve((L_skl, True), K_trans.T)  # Line 5
y_cov = gpr.kernel_(xx) - K_trans.dot(v)

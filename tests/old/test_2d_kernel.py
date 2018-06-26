"""Compare the 2d kernel in each situation."""

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
import copy
import piff

sigma = 2.
lx = 5.
ly = 4.
lxy = 0.1


A = np.linspace(0,5,5)
A, B = np.meshgrid(A,A)
C = np.array([A.reshape(25),B.reshape(25)])

AA = np.linspace(0,5,7)
AA, BB = np.meshgrid(AA,AA)
CC = np.array([AA.reshape(7*7),BB.reshape(7*7)])

K_cosmogp = cosmogp.rbf_kernel_2d(C.T, [sigma,np.sqrt(lx),np.sqrt(ly),lxy],
                                  new_x=None, nugget=0.0,
                                  floor=0.0, y_err=None)
H_cosmogp = cosmogp.rbf_kernel_2d(C.T, [sigma,np.sqrt(lx),np.sqrt(ly),lxy],
                                  new_x=CC.T, nugget=0.0,
                                  floor=0.0, y_err=None)

#kernel_georgee = sigma**2 * kernel_george.ExpSquaredKernel(metric=[[lx, lxy], [lxy, ly]], ndim=2)
kernel_georgee = sigma**2 * kernel_george.ExpSquaredKernel(metric=0.3**2, ndim=2)
gp_george = george.GP(kernel_georgee,white_noise=5,fit_white_noise=True)
p0 = gp_george.get_parameter_vector()
K_george = gp_george.get_matrix(C.T)
H_george = gp_george.get_matrix(C.T,x2=CC.T)


inv_l = np.array(([ly,-lxy],
                  [-lxy,lx]))
inv_l *= 1./(((lx)*(ly))-lxy**2)
        

#kernel_skl = sigma**2 * piff.AnisotropicRBF(invLam=inv_l)
kernel_skl = sigma**2 * skl_kernels.RBF(0.3)
gp_skl = skl_gpr(kernel_skl, alpha=0,optimizer=None,normalize_y=True)
gp_skl.fit(C.T,np.ones(25))
K_skl = gp_skl.kernel_(C.T)
H_skl = gp_skl.kernel_(C.T,Y=CC.T)



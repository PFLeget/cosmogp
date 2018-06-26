from scipy import linalg
import numpy as np
from svd_tmv import computeSVDInverse as svd


def svd_scipy(matrix):

    U,s,V = linalg.svd(matrix)
    Filter = (s>10**-15)
    if np.sum(Filter) != len(Filter):
        print 'Pseudo-inverse decomposition :', len(Filter)-np.sum(Filter)
        
    inv_S = np.diag(1./s[Filter])
    inv_matrix = np.dot(V.T[:,Filter],np.dot(inv_S,U.T[Filter]))

    return inv_matrix

x_axis = np.linspace(0,5,200)

A = x_axis - x_axis[:,None]

kernel = (1.**2) * np.exp(-0.5 * ((A * A) / (1.**2)))

kernel += (np.eye(len(x_axis)) * (0.2**2))

#kernel_inv = svd(kernel)
kernel_inv = svd_scipy(kernel)

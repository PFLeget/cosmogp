from scipy import linalg
import numpy as np

def svd_inverse(matrix,return_logdet=False):

    U,s,V = linalg.svd(matrix)
    Filtre = (s>10**-15)
    if np.sum(Filtre)!=len(Filtre):
        print 'Pseudo-inverse decomposition :', len(Filtre)-np.sum(Filtre)

    inv_S = np.diag(1./s[Filtre])
    inv_matrix = np.dot(V.T[:,Filtre],np.dot(inv_S,U.T[Filtre]))

    if return_logdet:
        log_det=np.sum(np.log(s[Filtre]))
        return inv_matrix,log_det
    else:
        return inv_matrix

    
def cholesky_inverse(matrix,return_logdet=False):

    L = linalg.cholesky(matrix, lower=True)
    inv_L = linalg.inv(L)
    inv_matrix = np.dot(inv_L.T, inv_L)

    if return_logdet:
        log_det=np.sum(2.*np.log(np.diag(L)))
        return inv_matrix,log_det
    else:
        return inv_matrix


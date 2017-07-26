"""
Some description.
"""

#import os
#import glob

# Automatically import all modules (python files)
#__all__ = [os.path.basename(m).replace('.py', '') for m in glob.glob("cosmogp/*.py")
#           if '__init__' not in m]

from .inv_matrix import svd_inverse
from .inv_matrix import cholesky_inverse

from .mean import return_mean

from .Gaussian_process import Gaussian_process
from .Gaussian_process import gaussian_process
from .Gaussian_process import gaussian_process_nobject

from .kernel import init_rbf
from .kernel import rbf_kernel_1d
from .kernel import rbf_kernel_2d

from .pull import build_pull

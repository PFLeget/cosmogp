#!/usr/bin/env python

"""
Some description
"""

#import os
#import glob

# Automatically import all modules (python files)
#__all__ = [os.path.basename(m).replace('.py', '') for m in glob.glob("cosmogp/*.py")
#           if '__init__' not in m]


from .Gaussian_process import Gaussian_process
from .Gaussian_process import gp_1D_1object
from .Gaussian_process import gp_1D_Nobject
from .Gaussian_process import RBF_kernel_1D
from .Gaussian_process import interpolate_mean

from .pull import build_pull

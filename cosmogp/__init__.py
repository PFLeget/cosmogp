#!/usr/bin/env python

"""
Some description
"""

import os
import glob

# Automatically import all modules (python files)
__all__ = [os.path.basename(m).replace('.py', '') for m in glob.glob("cosmogp/*.py")
           if '__init__' not in m]

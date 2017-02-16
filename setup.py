#!/usr/bin/env python

"""Setup script."""

import os
import glob
import numpy
import yaml
from setuptools import setup, find_packages, Extension

# Package name
name = 'cosmogp'

# Packages (subdirectories in clusters/)
packages = find_packages()

# Scripts (in scripts/)
scripts = glob.glob("scripts/*.py")

package_data = {}

setup(name=name,
      description=("cosmogp"),
      classifiers=["Topic :: Scientific :: Astronomy",
                   "Intended Audience :: Science/Research"],
      author="PF",
      packages=packages,
      scripts=scripts)

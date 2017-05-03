.. image:: https://readthedocs.org/projects/cosmogp/badge/?version=latest
   :target: http://cosmogp.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status
.. image:: https://landscape.io/github/PFLeget/cosmogp/master/landscape.svg?style=flat
   :target: https://landscape.io/github/PFLeget/cosmogp/master
   :alt: Code Health
____

**WARNING**: Package under development

____

.. inclusion-marker-do-not-remove                                                                                            

cosmogp
--------

cosmogp is a gaussian process code interpolator develloped in python.

cosmogp was mainly developed to :

*   interpolate in one dimension (like supernova ligth-curves)
*   interpolate in two dimensions (like PSF within the full focal plane)

Mathematical part and fews numerical implematentation are decribe in french 
`here <https://tel.archives-ouvertes.fr/tel-01467899>`_ (chapter 8, will come in english soon). 

    
	
Installation
------------

To install::

  git clone https://github.com/PFLeget/cosmogp.git
  pip install cosmogp/

To install in a local directory ``mypath``, use::

  pip install --prefix='mypath' cosmogp/

and do not forget to add it to your PYTHONPATH.

To upgrade to a new version (after a ``git pull`` or a local modification), use::
  
  pip install --upgrade (--prefix='mypath') cosmogp/
  
Package developers will want to run::

  python setup.py develop

Dependencies
------------

``Clusters`` has for now the following dependencies (see the quick
installs below):

- Python 2.7 and libraries listed in the `requirements <requirements.txt>`_ file
   

Python
``````

To install the python dependencies, simply do::

  pip install -r requirements.txt
	      
			  

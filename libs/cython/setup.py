'''
* Created by Zhenia Syryanyy (Yevgen Syryanyy)
* e-mail: yuginboy@gmail.com
* License: this code is in GPL license
* Last modified: 2017-06-20
'''
from distutils.core import setup, Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=cythonize([Extension('cython_equations', ['cython_equations.pyx'], include_dirs=[numpy.get_include()])])
)
# from the local directory run the command:
# python3.6 setup.py build_ext --inplace
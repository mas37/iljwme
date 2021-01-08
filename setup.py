# Compilers
cc = 'mpiicc'
cxx = 'mpiicpc'

# Compile and link flags
comp_args = '-O3 -std=c++11 -Wno-sign-compare -DMLIP_MPI -DMLIP_DEV -lgfortran'
link_flags = '-std=c++11 -lgfortran'
link_libs = '_mlip ifcore mkl_rt'
link_libdirs = '/opt/intel/2017/u8/compilers_and_libraries_2017.8.262/linux/mkl/lib/intel64 ./lib'

blas_dir = '/opt/intel/2017/u8/compilers_and_libraries_2017.8.262/linux/mkl/include'
comp_args = comp_args.split()
link_args = link_flags.split()
libs = link_libs.split()
lib_dirs = link_libdirs.split()

import os
import numpy
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
try:
    from Cython.Build import cythonize
except ImportError:
    sys.exit("Cython not found. Cython is needed to build the extension modules.")

src_dir = "./"
src_py_dir = "./src/external/python"

os.environ["CC"] = cc
os.environ["CXX"] = cxx

ext_name = "lib.mlippy"
src_files = [ './lib/mlippy.pyx', src_py_dir + '/mlip_handler.cpp']


mlp_mod = Extension('lib.mlippy',
           sources=src_files,
           language="c++",
           include_dirs=[src_py_dir,src_dir,blas_dir,numpy.get_include()],
           extra_compile_args=comp_args,
           extra_link_args=link_args,
           libraries=libs,
           library_dirs=lib_dirs,
           )

setup(  name='mlippy',
#         packages=['mlippy','mlippy.mtp'],
        cmdclass={'build_ext': build_ext},
#         options={'build_ext':{'inplace':True,'force':True}},
        ext_modules = cythonize([ mlp_mod ],
            build_dir="./obj/mlippy")
      )


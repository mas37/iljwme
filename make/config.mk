# Directories
PREFIX = .
BIN_DIR = $(PREFIX)/bin
LIB_DIR = $(PREFIX)/lib
FORTRAN_DIR = /usr/local/gfortran/lib

# Compilers for the executable
CC_EXE  = mpiicc
CXX_EXE = mpiicpc
FC_EXE  = mpiifort

# Compilers for the library
CC_LIB  = icc
CXX_LIB = icpc
FC_LIB  = ifort

# Compile and link flags
CPPFLAGS += -O3 
FFLAGS += -O3 
CXXFLAGS += -DMLIP_MPI
LDFLAGS += -L/opt/intel/2020/u4/compilers_and_libraries_2020.4.304/linux/mkl/lib/intel64 -lmkl_rt -lifcore
CPPFLAGS += -I/opt/intel/2020/u4/compilers_and_libraries_2020.4.304/linux/mkl/include
CXXFLAGS += -DMLIP_INTEL_MKL 

# Extra variables

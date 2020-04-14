SHELL := /bin/bash
################################################################################
#
#   This Makefile will compile the potential energy surface module to expose it
#   to Python using f2py.
#
################################################################################

F2PY=~/anaconda2/bin/f2py

.PHONY: all build clean

all: build

build:
	$(F2PY) -c lib_mlip_rpmd.a mlip_interface.f mlip_interface.pyf --build-dir build
	# dirty hack: we are using that the last command of f2py is linking with gfortran
	# we need to substitute gfortran with g++
	rm PES.so
	$(F2PY) -c lib_mlip_rpmd.a mlip_interface.f mlip_interface.pyf --build-dir build > build/_comm.txt
	rm PES.so
	sed -i 's/gfortran/g++/1' build/_comm.txt
	`tail -n 1 build/_comm.txt`

# pyf:
# 	$(F2PY) -m PES -h mlip_interface.pyf mlip_interface.f --overwrite-signature

clean:
	rm -f *.o *.so

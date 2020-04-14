#!/bin/bash

# $1 = path to LAMMPS+MLIP binaries

LMP_EXE_par=../../../bin/lmp_mpi
LMP_EXE_ser=../../../bin/lmp_serial

TMP_DIR=./out
mkdir -p $TMP_DIR

# Body:
# This tests parallel and serial binaries
if [[ -e $LMP_EXE_par ]]; then $LMP_EXE_par < lmp.inp -log $TMP_DIR/lammps_parallel.log || exit 1; fi

if [[ -e $LMP_EXE_ser ]]; then $LMP_EXE_ser < lmp.inp -log $TMP_DIR/lammps_serial.log || exit 1; fi


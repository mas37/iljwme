#!/bin/bash

# Preamble, common for all examples
LMP_EXE=../../../bin/lmp_serial
TMP_DIR=./out
mkdir -p $TMP_DIR
rm -rf $TMP_DIR/* > /dev/null

# Body:
# This example demonstrates the work of LAMMPS (serial version) with linked MLIP. 
# MLIP is linked for EFS calculation by entire configuration.
if [[ -e $LMP_EXE ]]; then $LMP_EXE < lmp.inp -log $TMP_DIR/lammps.log; fi

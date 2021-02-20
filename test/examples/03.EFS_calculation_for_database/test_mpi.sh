#/bin/bash

# Preamble, common for all examples
MLP_EXE=../../../bin/mlp
TMP_DIR=./out
mkdir -p $TMP_DIR
rm $TMP_DIR/* > /dev/null

# Body:
# This calculates energy, forces and stresses (EFS) with fitted.mtp
# for all configurations from the database trainset.cfg, 
# and saves configurations with this data to out/Li_EFSbyMTP.cfg
mpirun -n 3 $MLP_EXE run mlip.ini noEFS.cfg 

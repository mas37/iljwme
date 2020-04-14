#/bin/bash

# Preamble, common for all examples
MLP_EXE=../../../bin/mlp
TMP_DIR=./out
mkdir -p $TMP_DIR

# Body:
# This calculates energy, forces and stresses (EFS) with fitted.mtp
# for all configurations from the database trainset.cfg, 
# and saves configurations with this data to out/Li_EFSbyMTP.cfg
$MLP_EXE run mlip.ini --filename=trainset.cfg --log=stdout

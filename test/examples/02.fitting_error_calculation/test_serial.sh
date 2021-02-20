#/bin/bash

# Preamble, common for all examples
MLP_EXE=../../../bin/mlp
TMP_DIR=./out
mkdir -p $TMP_DIR
rm $TMP_DIR/* > /dev/null

# Body:
# This calculates errors of energy, forces, and stresses (EFS)
# computed by fitted.mtp as compared to DFT (in file trainset.cfg)
$MLP_EXE run mlip.ini trainset.cfg

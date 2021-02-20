#/bin/bash

# Preamble, common for all examples
MLP_EXE=../../../bin/mlp
TMP_DIR=./out
mkdir -p $TMP_DIR
rm $TMP_DIR/* > /dev/null

# Body:
# This trains the potential *.mtp on configurations from the database trainset.cfg.
# The trained potential is saved to out
mpirun -n 3 $MLP_EXE run mlip.ini trainset.cfg 

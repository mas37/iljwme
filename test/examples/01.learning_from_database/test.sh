#/bin/bash

# Preamble, common for all examples
MLP_EXE=../../../bin/mlp
TMP_DIR=./out
mkdir -p $TMP_DIR

# Body:
# This trains the potential *.mtp on configurations from the database trainset.cfg.
# The trained potential is saved to out
$MLP_EXE run mlip.ini --filename=trainset.cfg --log=stdout

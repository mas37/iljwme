#/bin/bash

# Preamble, common for all examples
MLP_EXE=../../../bin/mlp
TMP_DIR=./out
mkdir -p $TMP_DIR
rm $TMP_DIR/* > /dev/null 

# Body:
# This reads sparse.cfg, selects the configurations with active learning (by energy equation),
# and saves the selected configurations to out/selected.cfg
$MLP_EXE run mlip.ini pool.cfg 

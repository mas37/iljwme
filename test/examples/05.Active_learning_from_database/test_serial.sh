#/bin/bash

# Preamble, common for all examples
MLP_EXE=../../../bin/mlp
TMP_DIR=./out
mkdir -p $TMP_DIR
rm $TMP_DIR/* > /dev/null

# Body:
# This reads sparce.cfg, selects the configurations with active learning ("by neighborhoods"),
# saves the selected configurations to out/TS100.cfg,
# and fits MTP on them (the trained MTP is saved to out/fitted.mtp)
$MLP_EXE run mlip.ini MDtrajectory.cfg 

#/bin/bash

# Preamble, common for all examples
MLP_EXE=../../../bin/mlp
TMP_DIR=./out
mkdir -p $TMP_DIR
rm $TMP_DIR/* > /dev/null

#echo $(pwd TMP_DIR)

# Body:
# This converts OUTCAR to the internal format out/relax.cfg.
mpirun -n 1 $MLP_EXE convert_cfg --input_format=vasp_outcar OUTCAR out/relax.cfg

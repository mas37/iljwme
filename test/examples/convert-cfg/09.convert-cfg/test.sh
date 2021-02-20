#!/bin/bash

# Preamble, common for all examples
MLP_EXE=../../../../../bin/mlp
TMP_DIR=./out
mkdir -p $TMP_DIR

$MLP_EXE convert_cfg OUTCAR out/POSCAR --input_format=vasp_outcar --output_format=vasp_poscar > /dev/null
diff correctPOSCAR out/POSCAR 1>&2


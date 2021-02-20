#!/bin/bash

# Preamble, common for all examples
MLP_EXE=../../../../../bin/mlp
TMP_DIR=./out
mkdir -p $TMP_DIR

$MLP_EXE convert_cfg OUTCAR out/output.bin.cfg --input_format=vasp_outcar --output_format=bin  --absolute_elements > /dev/null
diff correct_output.bin.cfg out/output.bin.cfg 1>&2

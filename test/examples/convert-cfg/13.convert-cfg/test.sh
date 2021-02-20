#!/bin/bash

# Preamble, common for all examples
MLP_EXE=../../../../../bin/mlp
TMP_DIR=./out
mkdir -p $TMP_DIR

cp output.cfg out/output.cfg
$MLP_EXE convert_cfg OUTCAR out/output.cfg --input_format=vasp_outcar --append --absolute_elements > /dev/null
diff correct_output.cfg out/output.cfg 1>&2

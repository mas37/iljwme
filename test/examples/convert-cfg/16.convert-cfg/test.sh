#!/bin/bash

# Preamble, common for all examples
MLP_EXE=../../../../bin/mlp
TMP_DIR=./out
mkdir -p $TMP_DIR

$MLP_EXE convert-cfg molpro.out out/output.cfg --input-format=molpro-out --output-format=txt > /dev/null
diff correct_molpro.cfg out/output.cfg 1>&2


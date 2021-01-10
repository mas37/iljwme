#!/bin/bash

MLP_EXE=../../../bin/mlp

$MLP_EXE relax relax.ini --cfg-filename=to-relax.cfg --save-unrelaxed=out/unrelaxed.cfg --save-relaxed=out/relaxed.cfg --max-step=0.05

cat out/relaxed.cfg* > out/relaxed_bin.cfg
rm out/relaxed.cfg*


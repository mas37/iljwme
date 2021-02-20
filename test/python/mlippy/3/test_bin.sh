#!/bin/bash

MLP_EXE=../../../bin/mlp

$MLP_EXE relax relax.ini to-relax.cfg out/relaxed_bin.cfg --save_unrelaxed=out/unr.cfg --max_step=0.05

rm Trained.mtp


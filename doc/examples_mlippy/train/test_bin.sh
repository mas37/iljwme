#!/bin/bash

mkdir out

MLP_EXE=../../../bin/mlp
$MLP_EXE train pot_light.mtp train.cfg --energy-weight=1 --force-weight=1e-3 --stress-weight=1e-4 --init-params=same  --no-mindist-update=TRUE --max-iter=20 --trained-pot-name=out/Trained_bin.mtp --skip-preinit


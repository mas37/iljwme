#!/bin/bash

MLP_EXE=../../../bin/mlp
$MLP_EXE train pot_light.mtp train.cfg --energy_weight=1 --force_weight=1e-3 --stress_weight=1e-4  --no_mindist_update=TRUE --max_iter=20 --save_to=Trained.mtp --bfgs_conv_tol=1e-8 --skip_preinit

cp Trained.mtp out/Trained_bin.mtp

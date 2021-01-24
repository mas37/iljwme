#!/bin/bash

mkdir out

MLP_EXE=../../../bin/mlp
$MLP_EXE select-add Trained_3sp.mtp train_init.cfg train_vasp.cfg out/diff_bin.cfg --selection-limit=20 --als-filename=out/state_bin.als --selected-filename='out/selected.cfg'

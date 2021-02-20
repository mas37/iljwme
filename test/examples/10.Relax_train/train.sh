#!/bin/bash

rm -f $3/Trained.mtp_
mpirun -n $1 $2 train $3/pot.mtp $3/train.cfg --energy_weight=1 --iteration_limit=30 --save_to=$3/Trained.mtp_ --bfgs_conv_tol=1e-8 --skip_preinit=TRUE --init_random=FALSE

mv $3/Trained.mtp_ $3/pot.mtp   #trained MTP from prev. step 

#generate the state.mvs
mpirun -n $1 $2 select_add $3/pot.mtp $3/train.cfg $3/train.cfg $3/temp.cfg --save_to=$3/state.mvs
rm $3/temp.cfg* > /dev/null 
cp $3/state.mvs $3/pot.mtp

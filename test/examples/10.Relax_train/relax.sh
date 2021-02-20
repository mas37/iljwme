#!/bin/bash

rm -f $3/unrelaxed.cfg  > /dev/null 
rm -f $3/relaxed.cfg	> /dev/null 

mpirun -n $1 $2 relax relax.ini to-relax.cfg $3/relaxed.cfg --save_unrelaxed=$3/unrelaxed.cfg

cat $3/unrelaxed.cfg_* >> $3/unrelaxed.cfg 
rm -f $3/selected.cfg* > /dev/null 
cat $3/relaxed.cfg_* >> $3/relaxed.cfg
rm -f $3/unrelaxed.cfg_* > /dev/null 
rm -f $3/relaxed.cfg_* > /dev/null 


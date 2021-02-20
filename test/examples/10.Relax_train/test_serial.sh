#!/bin/bash

MLP_EXE=../../../bin/mlp
TMP_DIR=./out
mkdir -p ${TMP_DIR}
rm -rf ${TMP_DIR}/* > /dev/null 

NCORES=1 #number of cores to use 

cp pot_clean.mtp ${TMP_DIR}/pot.mtp
cp train_init.cfg ${TMP_DIR}/train.cfg

#training of the potential
#./train.sh $NCORES $MLP_EXE ${TMP_DIR} >> ${TMP_DIR}/train.log

#relaxation
#./relax.sh $NCORES $MLP_EXE ${TMP_DIR} >> ${TMP_DIR}/relax.log

#let n=$(grep -o -i BEGIN_CFG ${TMP_DIR}/relaxed.cfg | wc -l)
#let unrel=$(grep -o -i BEGIN_CFG ${TMP_DIR}/unrelaxed.cfg | wc -l)
#let tot_rel=$(grep -o -i BEGIN_CFG to-relax.cfg | wc -l)

#echo "$tot_rel to relax, $n relaxed and $unrel unrelaxed"
#if [[ "$n+$unrel" -ne "$tot_rel" ]]; then exit 1; fi

#extend the training set
#cat vasp.cfg >> ${TMP_DIR}/train.cfg

#retraining of the potential
#./train.sh $NCORES $MLP_EXE ${TMP_DIR} >> ${TMP_DIR}/train.log

#relaxation
#./relax.sh $NCORES $MLP_EXE ${TMP_DIR} >> ${TMP_DIR}/relax.log

#let m=$(grep -o -i BEGIN_CFG ${TMP_DIR}/relaxed.cfg | wc -l)

#echo "was $n now $m"
#if (( "$m" <= "$n" )); then exit 1; fi

echo !Ok

exit 0






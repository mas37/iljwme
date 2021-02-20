#/bin/bash

#export VASP_EXE=$(readlink -f ../../../bin/vaspMPI)
#export VASP_EXE=$(readlink -f ../../../../../../vaspMPI)

# Preamble, common for all examples
LMP_EXE=../../../bin/lmp_mpi
#export MLP_EXE=$(readlink -f ../../../bin/mlp)
TMP_DIR=./out
mkdir -p ${TMP_DIR}
rm -rf ${TMP_DIR}/* > /dev/null

mkdir -p ./vasp
cp ./*CAR ./vasp
cp ./KPOINTS ./vasp

if [ -x "${VASP_EXE}" ]; then
    if [[ -x $LMP_EXE ]]; then $LMP_EXE < lmp.inp -log $TMP_DIR/lammps.log; fi
fi

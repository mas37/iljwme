#/bin/bash

export VASP_EXE=$(readlink -f ../../../bin/vaspMPI)
#export VASP_EXE=$(readlink -f ../../../../../../vaspMPI)

# Preamble, common for all examples
export MLP_EXE=$(readlink -f ../../../../bin/mlp)
TMP_DIR=./out
mkdir -p ${TMP_DIR}

mkdir -p ./vasp
cp ./*CAR ./vasp
cp ./KPOINTS ./vasp
#$MLP_EXE convert_cfg in.cfg POSCAR --output_format=vasp_poscar

echo $MLP_EXE

if [ -x "${VASP_EXE}" ]; then
$MLP_EXE run TS.cfg --settings_file=mlip.ini
fi

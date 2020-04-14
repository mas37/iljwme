#!/bin/bash
#
# TESTING
# Integration testing of pymatnest 
#
# https://github.com/libAtoms/pymatnest
#
# Note: LAMMPS_DIR variable shoud be exported in outside
#       PYMATNEST_DIR variable may be expported in outside

Err() { echo "$@" 1>&2; }

LIB_DIR=$(realpath ../../../lib/)

if [ -d "${PYMATNEST_DIR}" ]; then
  rm -f ${PYMATNEST_DIR}/MLIP_test_out.*
else
  Err "PYMATNEST is not installed on specified directory."
  exit 99
fi
if [ ! -d "${LAMMPS_DIR}" ]; then
  Err "LAMMPS is not installed on specified directory."
  exit 99
fi


cp ${LAMMPS_DIR}/python/lammps.py ${PYMATNEST_DIR}
cp ${LAMMPS_DIR}/src/liblammps_*mpi.so ${PYMATNEST_DIR}/liblammps_mpi.so

cp ./inp ${PYMATNEST_DIR}
cp ./mlip.ini ${PYMATNEST_DIR}
cp ./pot.mtp ${PYMATNEST_DIR}

cd ${PYMATNEST_DIR}
export PYTHONPATH="${PYTHONPATH}:${LIB_DIR}" 

./ns_run < ./inp

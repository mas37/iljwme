This example demonstrates ab-initio molecular dynamics with writing a configuration database. 
DFT by VASP is ab-initio model, LAMMPS (serial version) is MD driver.
Each tenth configuration is recorded.

NOTE: to run this example VASP is required. Path to VASP should be specified in run_vasp.sh script

Execute:
$ mkdir -p ./out
$ $(Path_to_LAMMPS_with_MLIP)/lmp_serial < lmp.inp

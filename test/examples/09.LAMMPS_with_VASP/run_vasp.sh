$MLP_EXE convert_cfg in.cfg POSCAR --output_format=vasp_poscar
cp POSCAR ./vasp
cd vasp
${VASP_EXE} 
cd ..
$MLP_EXE convert_cfg vasp/OUTCAR out.cfg --input_format=vasp_outcar

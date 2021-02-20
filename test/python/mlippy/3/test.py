import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../lib')))
import numpy as np
import ase
import mlippy
from ase.optimize import BFGS

import mpi4py

comm = mpi4py.MPI.COMM_WORLD
rank = comm.Get_rank()

mlippy.initialize(comm)

mlip = mlippy.mtp('Trained_3sp.mtp')

to_relax = mlippy.ase_loadcfgs('to-relax.cfg')

opts = {"select":"TRUE",
"load_state_from":"state.mvs",
"save_selected_to":"",
"threshold":"1.1",
"threshold_break":"1.5",
"relaxation:log":"relax.txt",
"relaxation:iteration_limit":"990",
"relaxation:mindist":"1.5",
"relaxation:init_mindist":"1.5"
}

relaxed_py = mlippy.ase_relax(mlip,to_relax,opts)

en_py = []
en_bin  = []

if (rank==0):
	os.system('cat ./out/relaxed_bin.cfg_* >> ./out/relaxed_bin.cfg')
	relaxed_bin = mlippy.ase_loadcfgs('out/relaxed_bin.cfg')

	for x in relaxed_py:
		if (x.energy != None):
			en_py.append(x.energy)

	for x in relaxed_bin:
		en_bin.append(x.energy)

	en_py.sort()
	en_bin.sort()

en_py = np.array(en_py).astype(np.float)
en_bin = np.array(en_bin).astype(np.float)

comm.Barrier
status=os.EX_OK

for i in range(len(en_bin)):
	if (round(en_py[i],4)!=round(en_bin[i],4)):
		status=1

sys.exit(status)

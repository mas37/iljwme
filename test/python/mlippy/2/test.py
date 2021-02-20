import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../lib')))

import ase
import mlippy
from ase.optimize import BFGS
import filecmp
import mpi4py
import numpy as np

comm = mpi4py.MPI.COMM_WORLD
rank = comm.Get_rank()

mlippy.initialize(comm)

mlip = mlippy.mtp()

mlip.load_potential('Trained_3sp.mtp')

train_init = mlippy.ase_loadcfgs('train_init.cfg')
train_vasp = mlippy.ase_loadcfgs('train_vasp.cfg')

diff_py = mlippy.ase_select(mlip, train_init, train_vasp)

status=os.EX_OK

en_py = []
en_bin  = []

if (rank==0):
	os.system('cat ./out/diff.cfg_* >> ./out/diff.cfg')
	diff_bin = mlippy.ase_loadcfgs('out/diff.cfg')

	for x in diff_py:
		en_py.append(x.energy)

	for x in diff_bin:
		en_bin.append(x.energy)

	en_py.sort()
	en_bin.sort()

en_py = np.array(en_py).astype(np.float)
en_bin = np.array(en_bin).astype(np.float)

comm.Barrier
status=os.EX_OK

for i in range(len(en_py)):
	if (round(en_py[i],4)!=round(en_bin[i],4)):
		status=1

sys.exit(status)


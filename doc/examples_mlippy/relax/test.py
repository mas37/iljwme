import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../lib')))
import numpy as np
import ase
import mlippy
from ase.optimize import BFGS

import mpi4py

comm = mpi4py.MPI.COMM_WORLD
rank = comm.Get_rank()

mlippy.initialize(comm)

mlip = mlippy.mtp('state.als')

to_relax = mlippy.ase_loadcfgs('to-relax.cfg')

opts = {"select":"TRUE", 
"mtp-filename":"state.als",
"load-state":"state.als",
"save-selected":"out/selected_py.cfg",
"threshold":"2",
"threshold-break":"5",
"abinitio":"null"
}

os.system('mkdir out')

relax_opts = {"iteration-limit":"990",
"min-dist":"1.5"}

result_py = mlippy.ase_relax(mlip,to_relax,opts,relax_opts)
relaxed_py = []
unrelaxed_py = []

for cfg in result_py:
	if (cfg.energy!=None):
		relaxed_py.append(cfg)
	else:
		unrelaxed_py.append(cfg)

mlippy.ase_savecfgs('out/relaxed_py.cfg',relaxed_py)
mlippy.ase_savecfgs('out/unrelaxed_py.cfg',unrelaxed_py)

comm.Barrier

status=os.EX_OK
sys.exit(status)

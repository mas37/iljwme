import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../lib')))
import mlippy
from ase.optimize import BFGS
import filecmp
import mpi4py

comm = mpi4py.MPI.COMM_WORLD
rank = comm.Get_rank()

mlippy.initialize(comm)

mlip = mlippy.mtp()
mlip.load_potential('pot_light.mtp')

c = mlippy.ase_loadcfgs('train.cfg')

opts = {"max_iter":"20",
"energy_weight":"1",
"force_weight":"1e-3",
"stress_weight":"1e-4",
"skip_preinit":"FALSE",
"no_mindist_update":"TRUE"
}

mlippy.ase_train(mlip,c,opts)  

mlip.save_potential('out/Trained_py.mtp')

status=os.EX_OK

if (filecmp.cmp('out/Trained_py.mtp', 'out/Trained_bin.mtp')==False):
	status=1

#if (rank==0):
#	os.system('rm -rf out')

sys.exit(status)
	

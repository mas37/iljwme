import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../lib')))
import mlippy
from ase.optimize import BFGS
import filecmp
import mpi4py

comm = mpi4py.MPI.COMM_WORLD
rank = comm.Get_rank()

mlippy.initialize(comm)

mlip = mlippy.mtp()
mlip.load_potential('pot_light.mtp')

ts = mlippy.ase_loadcfgs('train.cfg')

opts = {"max-iter":"20",
"energy-weight":"1",
"force-weight":"1e-3",
"stress-weight":"1e-4",
"skip-preinit":"TRUE",
"no-mindist-update":"TRUE",
"init-params":"same",

}
mlippy.ase_train(mlip,ts,opts)  
errors = mlippy.ase_errors(mlip,ts)
import pprint as pp

if (rank==0):
	pp.pprint(errors)
	os.system('mkdir out')
	mlip.save_potential('out/Trained_py.mtp')

status=os.EX_OK

print ('success')
sys.exit(status)
	

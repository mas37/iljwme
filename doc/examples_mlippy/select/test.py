import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../lib')))

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

opts = {"selection-limit":"20",
"als-filename":"out/state_py.als"
}

os.system('mkdir out')

diff_py = mlippy.ase_select(mlip, train_init, train_vasp,opts)


comm.Barrier

mlippy.ase_savecfgs('out/diff_py.cfg',diff_py)


status=os.EX_OK
sys.exit(status)


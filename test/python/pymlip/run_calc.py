import os,sys,string,inspect

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../bin')))

from ase import *
import numpy as np
import numpy.linalg as la
from numpy import identity

from ase.optimize import BFGS
from mpi4py import MPI

from ase.calculators import eam
import pymlip

from MLIP_Calculator import MLIP_Calculator

a = Atoms('Al3', [(0, 0, 0), (3, 0, 0), (0, 0, 3)])
a.set_cell([(8,0,0),(0,8,0),(0,0,8)])

calc = MLIP_Calculator('Trained.mtp')
calc.mlp.add_type(13)	
calc.mlp.load_settings('mlip.ini')		

#try:
#	from mpi4py import MPI
#	comm = MPI.COMM_WORLD
#	rank = comm.Get_rank()
#	print('Parallel mode is on, I am process ' + str(rank))
#except ImportError:
#	print('No MPI, serial mode is on')


	 
a.set_calculator(calc)
dyn = BFGS(a)

print a.get_potential_energy()
dyn.run(fmax=1e-3)


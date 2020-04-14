import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../bin')))

from pymlip import *
try:
    from mpi4py import MPI
except ImportError:
    sys.exit("mpi4py not found. mpi4py is needed to run test.")


comm = MPI.COMM_WORLD
rank = comm.Get_rank()

mlp = pymlip('test2')

mlp.load_potential('pot.mtp')
mlp.load_training_set('train.cfg')
#mlp.train(1e-2,1e-3,2,1e-4)
#mlp.get_train_errors()

if not os.path.exists('./out'):
	os.makedirs('./out')
else:
	if os.path.isfile("./out/saved0.cfg"):
		os.remove('./out/saved0.cfg')

status=os.EX_OK

if (rank==0):
	print('configurations reading/writing test'	)

	mlp.read_db('train.cfg') #need 2 configurations

	e1 = mlp.get_energy(0)
	e2 = mlp.get_energy(1)

	f1 = mlp.get_forces(0)
	f2 = mlp.get_forces(1)

	s1 = mlp.get_stresses(0)
	s2 = mlp.get_stresses(1)

	p1 = mlp.get_positions(0)
	l1 = mlp.get_lattice(0)
	t1 = mlp.get_types(0)

	mlp.write_cfg('./out/saved0.cfg',0)

	mlp.calc_cfg(0)
	mlp.calc_cfg(1)

	if (e1==mlp.get_energy(0)):
		status=1
	if (e2==mlp.get_energy(1)):
		status=1

	N0 = mlp.get_size(0)
	N1 = mlp.get_size(1)

	f_new1 = mlp.get_forces(0)
	s_new1 = mlp.get_stresses(0)

	for i in range(N0):
		for j in range (3):
			if (f1[i,j]==f_new1[i,j]):
				status=1

	for i in range(6):
		if (s1[i]==s_new1[i]):
			status=1


	mlp.reset()
	mlp.read_db('./out/saved0.cfg')

	if (e1!=mlp.get_energy(0)):
		status=1

	for i in range(N0):
		for j in range (3):
			if (f1[i,j]!=mlp.get_forces(0)[i,j]):
				status=1

	for i in range(6):
		if (s1[i]!=mlp.get_stresses(0)[i]):
			status=1

	for i in range(N0):
		for j in range(3):
			if (p1[i,j]!=mlp.get_positions(0)[i,j]):
				status=1

	for i in range(3):
		for j in range(3):
			if (l1[i,j]!=mlp.get_lattice(0)[i,j]):
				status=1

	for i in range(N0):
		if (t1[i]!=mlp.get_types(0)[i]):
			status=1

sys.exit(status)

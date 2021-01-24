import os,sys,string,inspect
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../lib')))
import numpy as np
import mlippy
import pprint  #for pretty printing of dictionaries

mlippy.initialize()

fname = sys.argv[1] #filename to load cfgs from
stress_trsh = (float)(sys.argv[2]) #threshold for stress error

cfgs = mlippy.ase_loadcfgs(fname)

mlp = mlippy.mtp()
mlp.load_potential('pot.mtp')

screened = []
rejected = []
for i in range(len(cfgs)):
	report = mlippy.ase_errors(mlp,[cfgs[i]])
	if ((float)(report['Stresses: Average absolute difference']) < stress_trsh):
		screened.append(cfgs[i])
	else:
		rejected.append(cfgs[i])

os.system('mkdir out')
mlippy.ase_savecfgs("out/" + fname + "_screened",screened)
mlippy.ase_savecfgs("out/" + fname + "_rejected",rejected)


report = mlippy.ase_errors(mlp,cfgs)

print ("MLIP errors on the whole training set:")
pprint.pprint(report)

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../lib')))
import mlippy
import ase
from ase.optimize import BFGS

mlp = mlippy.mtp('state.als')

opts = {"select":"TRUE",
"load-state":"state.als",
"save-selected":"out/selected.cfg",
"threshold":"10",
"threshold-break":"100",
"write-cfgs":"out/record.cfgs",
"write-cfgs:skip":"50"
}


mlp.add_atomic_type(26)
mlp.add_atomic_type(27)
print(mlp.get_types_mapping())

calc = mlippy.MLIP_Calculator(mlp, opts)	

b = 5.01
a = ase.Atoms('FeCo',positions=[(0,0, 0), (b/2, b/2, b/2)],cell=[(0, b, b), (b, 0, b), (b, b, 0)],pbc=True)

a=a.repeat(2)
a.rattle(stdev=0.5)

a.set_calculator(calc)

os.system('mkdir out')

en_initial = a.get_potential_energy()
mlippy.ase_savecfgs('out/initial_ase.cfg',[a])


dyn = BFGS(a)
dyn.run(fmax=1e-2)

mlippy.ase_savecfgs('out/relaxed_ase.cfg',[a])
en_final = a.get_potential_energy()

print ("Potential energy of initial configuration: "+str(en_initial))
print ("Potential energy of final configuration: "+str(en_final))
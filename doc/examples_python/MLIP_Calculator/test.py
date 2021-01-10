import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../lib')))
import mlippy
import ase
from ase.optimize import BFGS

mlp = mlippy.mtp('state.mvs')

opts = {"select":"FALSE",
"load_state":"state.mvs",
"save_selected":"selected.cfg",
"threshold":"2",
"threshold_break":"10",
"write_cfgs":"record.cfgs",
"write_cfgs:skip":"3"
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
mlippy.ase_savecfgs('initial_ase.cfg',[a])

print(a.get_potential_energy())

dyn = BFGS(a)
dyn.run(fmax=1e-2)

mlippy.ase_savecfgs('relaxed_ase.cfg',[a])


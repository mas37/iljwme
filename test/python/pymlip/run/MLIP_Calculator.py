import os,sys,string,inspect

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../bin')))

from ase import *
import numpy as np
import numpy.linalg as la
from numpy import identity

from ase.optimize import BFGS
from mpi4py import MPI

import pymlip


def equal(a, b, tol=None):
    """ndarray-enabled comparison function."""
    if isinstance(a, np.ndarray):
        b = np.array(b)
        if a.shape != b.shape:
            return False
        if tol is None:
            return (a == b).all()
        else:
            return np.allclose(a, b, rtol=tol, atol=tol)
    if isinstance(b, np.ndarray):
        return equal(b, a, tol)
    if tol is None:
        return a == b
    return abs(a - b) < tol * abs(b) + tol


class MLIP_Calculator:	
    mlp = pymlip.pymlip('for_calc')         # MLIP FUNCTIONS!!!

    implemented_properties = ['energy','forces','stresses']
    default_parameters = {}

    all_properties = ['energy', 'forces', 'stress']
    all_changes = ['positions', 'numbers', 'cell', 'pbc']

    def __init__(self,potname,restart=None, ignore_bad_restart_file=False, label=None,
                 atoms=None, **kwargs):
       
       
        self.mlp.load_potential(potname)    # MLIP FUNCTIONS!!!


        self.atoms = None  # copy of atoms object from last calculation
        self.results = {}  # calculated properties (energy, forces, ...)
        self.parameters = None  # calculational parameters

        if atoms is not None:
            atoms.calc = self
            if self.atoms is not None:
                # Atoms were read from file.  Update atoms:
                if not (equal(atoms.numbers, self.atoms.numbers) and
                        (atoms.pbc == self.atoms.pbc).all()):
                    raise RuntimeError('Atoms not compatible with file')
                atoms.positions = self.atoms.positions
                atoms.cell = self.atoms.cell
                
        self.set(**kwargs)

        if not hasattr(self, 'name'):
            self.name = self.__class__.__name__.lower()

    def get_default_parameters(self):
        return Parameters(copy.deepcopy(self.default_parameters))

    def todict(self):
        default = self.get_default_parameters()
        return dict((key, value)
                    for key, value in self.parameters.items()
                    if key not in default or value != default[key])

    def reset(self):
        """Clear all information from old calculation."""

        self.atoms = None
        self.results = {}
        
    def get_atoms(self):
        if self.atoms is None:
            raise ValueError('Calculator has no atoms')
        atoms = self.atoms.copy()
        atoms.calc = self
        return atoms

    @classmethod
    def read_atoms(cls, restart, **kwargs):
        return cls(restart=restart, label=restart, **kwargs).get_atoms()

    def set(self, **kwargs):
        """Set parameters like set(key1=value1, key2=value2, ...).
        
        A dictionary containing the parameters that have been changed
        is returned.

        Subclasses must implement a set() method that will look at the
        chaneged parameters and decide if a call to reset() is needed.
        If the changed parameters are harmless, like a change in
        verbosity, then there is no need to call reset().

        The special keyword 'parameters' can be used to read
        parameters from a file."""

        if 'parameters' in kwargs:
            filename = kwargs.pop('parameters')
            parameters = Parameters.read(filename)
            parameters.update(kwargs)
            kwargs = parameters

        changed_parameters = {}

        for key, value in kwargs.items():
            oldvalue = self.parameters.get(key)
            if key not in self.parameters or not equal(value, oldvalue):
                if isinstance(oldvalue, dict):
                    # Special treatment for dictionary parameters:
                    for name in value:
                        if name not in oldvalue:
                            raise KeyError(
                                'Unknown subparameter "%s" in '
                                'dictionary parameter "%s"' % (name, key))
                    oldvalue.update(value)
                    value = oldvalue
                changed_parameters[key] = value
                self.parameters[key] = value

        return changed_parameters

    def check_state(self, atoms, tol=1e-15):
        """Check for system changes since last calculation."""
        if self.atoms is None:
            system_changes = self.all_changes
        else:
            system_changes = []
            if not equal(self.atoms.positions, atoms.positions, tol):
                system_changes.append('positions')
            if not equal(self.atoms.numbers, atoms.numbers):
                system_changes.append('numbers')
            if not equal(self.atoms.cell, atoms.cell, tol):
                system_changes.append('cell')
            if not equal(self.atoms.pbc, atoms.pbc):
                system_changes.append('pbc')
          
        return system_changes

    def get_potential_energy(self, atoms=None, force_consistent=False):
        energy = self.get_property('energy', atoms)
        if force_consistent:
            return self.results.get('free_energy', energy)
        else:
            return energy

    def get_forces(self, atoms=None):
        return self.get_property('forces', atoms)

    def get_stress(self, atoms=None):
        return self.get_property('stress', atoms)

    def get_property(self, name, atoms=None, allow_calculation=True):
        if name not in self.implemented_properties:
            raise NotImplementedError

        if atoms is None:
            atoms = self.atoms
            system_changes = []
        else:
            system_changes = self.check_state(atoms)
            if system_changes:
                self.reset()

        if name not in self.results:
            if not allow_calculation:
                return None
            try:
                self.calculate(atoms, [name], system_changes)
            except Exception:
                self.reset()
                raise


        result = self.results[name]
        if isinstance(result, np.ndarray):
            result = result.copy()
        return result

    def calculation_required(self, atoms, properties):
        system_changes = self.check_state(atoms)
        if system_changes:
            return True
        for name in properties:
            if name not in self.results:
                return True
        return False
        
    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=all_changes):
      
        if atoms is not None:
            self.atoms = atoms.copy()
   
            self.mlp.set_cfg(self.atoms,0)   

			#mlp_res =  self.mlp.calc_cfg(0)
            mlp_res =  self.mlp.calc_cfg_active(0)
           
            if (mlp_res==0):
                self.results = {'energy': self.mlp.get_energy(0),
                           'forces': self.mlp.get_forces(0),
                          'stress':self.mlp.get_stresses(0)}
            else:
                #print ('mlip EFS calculation failed, code ' + str(mlp_res))
                exit()

        else:
			print 'atomic system is empty'
			exit()






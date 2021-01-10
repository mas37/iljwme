from __future__ import print_function
from ctypes import *
import os, sys
import ase
import ast

import numpy as np
cimport numpy as np

from libc.stdlib cimport malloc, free

def initialize(MPI.Comm comm=None):
	if comm != None:
		init_mpi(comm.ob_mpi)
	else:
		init()

cdef cfg_set(atoms, cfg_data& cfg):
	cdef np.ndarray[long, ndim=1, mode = 'c'] types = atoms.numbers
	cdef np.ndarray[double, ndim=2, mode = 'c'] pos = atoms.positions
	cdef np.ndarray[double, ndim=2, mode = 'c'] lat = atoms.cell
	cdef np.ndarray[double, ndim=2, mode = 'c'] forces 
	cdef np.ndarray[double, ndim=1, mode = 'c'] stresses

	cfg.size = len(atoms.numbers)
	cfg.types = <long*>types.data
	cfg.pos = <double*>pos.data
	cfg.lat = <double*>lat.data
	cfg.forces = NULL
	cfg.stresses = NULL
	cfg.has_energy=0
	if hasattr(atoms, 'forces'):
		forces = atoms.forces
		cfg.forces = <double*>forces.data
	if hasattr(atoms, 'energy'):
		cfg.energy = <double>atoms.energy
		cfg.has_energy=1
	if hasattr(atoms, 'stresses'):
		stresses = atoms.stresses
		cfg.stresses = <double*>stresses.data

def ase_relax(pot,atom_cfgs,options={},relax_options={}):
	N = len(atom_cfgs)

	cdef cfg_data* atom_pcfgs = <cfg_data*> malloc(N*sizeof(cfg_data))
	for i in range(N):
		if not hasattr(atom_cfgs[i], 'energy'):
			atom_cfgs[i].energy = float(0.0)
		if not hasattr(atom_cfgs[i], 'forces') or atom_cfgs[i].forces is None:
			size = len(atom_cfgs[i].numbers)
			atom_cfgs[i].forces = np.zeros((size,3))	
		if not hasattr(atom_cfgs[i], 'stresses') or atom_cfgs[i].stresses is not None:
			atom_cfgs[i].stresses = np.zeros(6)
		cfg_set(atom_cfgs[i], atom_pcfgs[i])

	cdef int n_rlxed = N
	cdef int* atom_rlxed = <int*> malloc(N*sizeof(int))

	opts = {}
	for key, value in options.items():
		opts[key.encode('utf-8')] = value.encode('utf-8')
		opts[('select:'+key).encode('utf-8')] = value.encode('utf-8')

	relax_opts = {}
	for key, value in relax_options.items():
		relax_opts[key.encode('utf-8')] = value.encode('utf-8')
		relax_opts[('relax:'+key).encode('utf-8')] = value.encode('utf-8')		

	_relax(pot_addr(pot),N, atom_pcfgs, n_rlxed, atom_rlxed,opts,relax_opts)

	#free(atom_pcfgs)
	
	if (n_rlxed!=0):
		relaxed =  list(map(lambda x: atom_rlxed[x], range(n_rlxed)))   
		relaxed_ens = list(map(lambda x: atom_pcfgs[x].energy,relaxed))			

		#for i in range (n_rlxed):
		#	relaxed_ase[i].energy = atom_pcfgs[relaxed[i]].energy		
	else:
		relaxed_ase = []
		
	for i in range(N):
		atom_cfgs[i].energy = None
	for i in range (n_rlxed):
		atom_cfgs[relaxed[i]].energy=relaxed_ens[i]
	for i in range(N):
		if (atom_cfgs[i].energy == None):
			atom_cfgs[i].forces = None
			atom_cfgs[i].stresses = None
		tmp_atoms = atom_cfgs[i].repeat(2)
		t = (tmp_atoms.get_all_distances()).flatten()
		t2 = [x for x in t if x != 0]
		atom_cfgs[i].features = {"mindist":str(round(min(t2),3))}
		
	free(atom_rlxed)
	
	return atom_cfgs
	#return relaxed_ase

def ase_select(pot,train_cfg,new_cfg,options={}):

	new_n = len(new_cfg)
	cdef cfg_data* new_pcfgs = <cfg_data*> malloc(new_n*sizeof(cfg_data))
	for i in range(new_n):
		cfg_set(new_cfg[i],new_pcfgs[i])

	train_n = len(train_cfg)
	cdef cfg_data* train_pcfgs = <cfg_data*> malloc(train_n*sizeof(cfg_data))
	for i in range(train_n):
		cfg_set(train_cfg[i],train_pcfgs[i])

	cdef int n_diff = new_n
	cdef int* pdiff = <int*> malloc(new_n*sizeof(int))

	opts = {}
	for key, value in options.items():
		opts[key.encode('utf-8')] = value.encode('utf-8')

	_select_add(pot_addr(pot),train_n,train_pcfgs,new_n,new_pcfgs,n_diff,pdiff,opts)

	free(new_pcfgs)

	diff =  list(map(lambda x: pdiff[x], range(n_diff)))
	selected_ase = list(map(lambda x: new_cfg[x],diff))

	free(pdiff)
	return selected_ase

def ase_train(pot,train_cfg,options={}):
	train_n = len(train_cfg)
	cdef cfg_data* train_pcfgs = <cfg_data*> malloc(train_n*sizeof(cfg_data))
	for i in range(train_n):
		cfg_set(train_cfg[i],train_pcfgs[i])

	opts = {}
	for key, value in options.items():
		opts[key.encode('utf-8')] = value.encode('utf-8')

	_train(pot_addr(pot),train_n,train_pcfgs,opts)
	
	
def ase_errors(pot, check_cfgs, on_screen=False):
	conf_n = len(check_cfgs)
	cdef cfg_data* pcfgs = <cfg_data*> malloc(conf_n*sizeof(cfg_data))
	for i in range(conf_n):
		cfg_set(check_cfgs[i],pcfgs[i])

	map = _calcerrors(pot_addr(pot), conf_n, pcfgs, on_screen)
	
	dict = {}
	dict['Energy: Maximal absolute difference'] = map['Energy: Maximal absolute difference']#.decode('utf-8')
	dict['Energy: Average absolute difference'] = map['Energy: Average absolute difference']#.decode('utf-8')
	dict['Energy: RMS     absolute difference'] = map['Energy: RMS     absolute difference']#.decode('utf-8')
	dict['Energy per atom: Maximal absolute difference'] = map['Energy per atom: Maximal absolute difference']#.decode('utf-8')
	dict['Energy per atom: Average absolute difference'] = map['Energy per atom: Average absolute difference']#.decode('utf-8')
	dict['Energy per atom: RMS absolute difference'] = map['Energy per atom: RMS absolute difference']#.decode('utf-8')
	dict['Forces: Maximal absolute difference'] = map['Forces: Maximal absolute difference']#.decode('utf-8')
	dict['Forces: Average absolute difference'] = map['Forces: Average absolute difference']#.decode('utf-8')
	dict['Forces: RMS absolute difference'] = map['Forces: RMS absolute difference']#.decode('utf-8')
	dict['Forces: Max(ForceDiff) / Max(Force)'] = map['Forces: Max(ForceDiff) / Max(Force)']#.decode('utf-8')
	dict['Forces: RMS(ForceDiff) / RMS(Force)'] = map['Forces: RMS(ForceDiff) / RMS(Force)']#.decode('utf-8')
	dict['Stresses: Maximal absolute difference'] = map['Stresses: Maximal absolute difference']#.decode('utf-8')
	dict['Stresses: Average absolute difference'] = map['Stresses: Average absolute difference']#.decode('utf-8')
	dict['Stresses: RMS absolute difference'] = map['Stresses: RMS absolute difference']#.decode('utf-8')
	dict['Stresses: Max(StressDiff) / Max(Stress)'] = map['Stresses: Max(StressDiff) / Max(Stress)']#.decode('utf-8')
	dict['Stresses: RMS(StressDiff) / RMS(Stress)'] = map['Stresses: RMS(StressDiff) / RMS(Stress)']#.decode('utf-8')
	
	return dict

def convert_ase2vasp(atoms,outfn, order):
	ase_savecfgs('temp.cfg_pos',atoms)
	elems = ''
	for i in range(len(order)):
		elems+=str(order[i])
		if (i!=len(order)-1):
			elems+=','

	x = _run_command('temp.cfg_pos',outfn.encode('utf-8'),1,elems.encode('utf-8'))
	os.system('rm -f temp.cfg_pos 2> /dev/null')
	return x

def convert_vasp2ase(indir):
	_run_command(indir.encode('utf-8'),'temp.cfg_outc',0,"")
	t = ase_loadcfgs('temp.cfg_outc')
	os.system('rm -f temp.cfg_outc 2> /dev/null')
	return t

cdef class mtp: 
	def __init__(self,name=''):
		self.pot = new pot_handler()
		if (name!=''):
			self.load_potential(name)
	def __dealloc__(self):
		del self.pot
	def load_potential(self,name):
		self.pot.load_potential(name.encode('utf-8'))
	def save_potential(self,name):
		self.pot.save_potential(name.encode('utf-8'))
	def add_atomic_type(self,ase_type):
		return self.pot.add_atomic_type(ase_type)
	def get_types_mapping(self):
		return self.pot.get_types_mapping()
	def init_wrapper(self, options):					#"select":"TRUE" in options means considering grades
		opts = {}
		for key, value in options.items():
			opts[key.encode('utf-8')] = value.encode('utf-8')
			opts[('select:'+key).encode('utf-8')] = value.encode('utf-8')
		self.pot.init_wrapper(opts)
	def calc_cfg(self, atom_cfg):	 
		
		system_types = atom_cfg.get_atomic_numbers()
		unique_types = np.unique(atom_cfg.get_atomic_numbers())

		#changing of atomic types to relative numeration: 0,1,2....
				
		types_present = np.array(self.pot.get_types_mapping())
		
		for i in range(len(unique_types)):
			if (unique_types[i] in types_present):
				continue	
			else:
				raise Exception("ERROR: atomic number " + str(unique_types[i]) + " is not present in the MTP!")
				return -1
	
		for i in range(len(system_types)):
			system_types[i] = (np.where(types_present == system_types[i]))[0]

		atom_cfg.set_atomic_numbers(system_types)
		
										
		cdef cfg_data* atom_pcfg = <cfg_data*> malloc(sizeof(cfg_data))
		if not hasattr(atom_cfg, 'energy'):
			atom_cfg.energy = float(0.0)
		if not hasattr(atom_cfg, 'forces') or atom_cfg.forces is None:				#todo: delete?
			size = len(atom_cfg.numbers)
		atom_cfg.forces = np.zeros((size,3))	
		if not hasattr(atom_cfg, 'stresses') or atom_cfg.stresses is not None:
			atom_cfg.stresses = np.zeros(6)
		cfg_set(atom_cfg, atom_pcfg[0])

		size = len(atom_cfg)
		res = self.pot.calc_cfg_efs(atom_pcfg)    #if res < 0 then failure
		result = []
		result.append(res)	

		if (res>=0):
			result.append(_cfg_ene(atom_pcfg[0]))
		
			forces = np.zeros((size,3))
			for i in range(size):
				for j in range(3):
					forces[i,j]=_cfg_frc(atom_pcfg[0],i,j)
			
			stress = np.zeros(6)
			for i in range(6):
				stress[i]=_cfg_str(atom_pcfg[0],i)

			result.append(forces)
			result.append(stress)

			
		free(atom_pcfg)	
		return result

################
#### Atoms  ####
################ 

class Cfg:
	pos = None
	lat = None
	types = None
	energy = None
	forces = None
	stresses = None
	features = {}

def readcfg(f):
	cfg = Cfg()
	cfg.lat = np.zeros((3,3))
	cfg.features = {}
	size = -1
	mode = -1
	line = f.readline()
	while line:
		line_orig = line.strip()
		line = line.upper()
		line = line.strip()
		if mode == 0:
			if line.startswith('SIZE'):
				line = f.readline()
				size = int(line.strip())
				cfg.types = np.zeros(size)
				cfg.pos = np.zeros((size,3))
			elif line.startswith('SUPERCELL'):
				line = f.readline()
				vals = line.strip().split()
				cfg.lat[0,:] = vals[0:3]
				line = f.readline()
				vals = line.strip().split()
				cfg.lat[1,:] = vals[0:3]
				line = f.readline()
				vals = line.strip().split()
				cfg.lat[2,:] = vals[0:3]
			elif line.startswith('ATOMDATA'):
				if line.endswith('FZ'):
					cfg.forces = np.zeros((size,3))	
				for i in range(size):
					line = f.readline()
					vals = line.strip().split()
					cfg.types[i] = vals[1]
					cfg.pos[i,:] = vals[2:5]
					if cfg.forces is not None:
						cfg.forces[i,:] = vals[5:8]
			elif line.startswith('ENERGY'):
				line = f.readline()
				cfg.energy = float(line.strip())
			elif line.startswith('PLUSSTRESS'):
				line = f.readline()
				vals = line.strip().split()
				cfg.stresses = np.zeros(6)
				cfg.stresses[:] = vals[0:6]
			elif line.startswith('FEATURE'):
				vals = line_orig.split()
				cfg.features[vals[1]]=vals[2]			
		if line.startswith('BEGIN_CFG'):
			mode = 0
		elif line.startswith('END_CFG'):	 
			break
		line = f.readline()
	return cfg

def savecfg(f,cfg, desc = None):
	atstr1 = 'AtomData:  id type       cartes_x      cartes_y      cartes_z           fx          fy          fz'
	atstr2 = 'AtomData:  id type       cartes_x      cartes_y      cartes_z'
	size = len(cfg.types)
	print ('BEGIN_CFG', file=f )
	print ('Size', file=f)
	print ('   %-d' % size, file=f)
	if cfg.lat is not None:
		print ('SuperCell',file=f)
		for i in range(3):
			print ('  %14f%14f%14f' \
				   % (cfg.lat[i,0],cfg.lat[i,1],cfg.lat[i,2]) ,file=f)
	if cfg.forces is not None:
		print (atstr1,file=f)
	else:
		print (atstr2,file=f)
	for i in range(size):
		if cfg.forces is not None:
			print ('		 %4d %4d %14f%14f%14f  %11.6f %11.6f %11.6f' % \
			   ( i+1,cfg.types[i],cfg.pos[i,0],cfg.pos[i,1],cfg.pos[i,2], 
				 cfg.forces[i,0],cfg.forces[i,1],cfg.forces[i,2] ),file=f)
		else:
			print ('		 %4d %4d %14f%14f%14f' % \
			   ( i+1,cfg.types[i],cfg.pos[i,0],cfg.pos[i,1],cfg.pos[i,2]), 
				file=f)
	if cfg.energy is not None:
		print('Energy',file=f)
		print('\t   '+str(cfg.energy),file=f)
	if cfg.stresses is not None:
		print ('PlusStress:  xx          yy          zz          yz          xz          xy',file=f)
		print ('   %12.5f%12.5f%12.5f%12.5f%12.5f%12.5f' \
				% (cfg.stresses[0],cfg.stresses[1],cfg.stresses[2],
				   cfg.stresses[3],cfg.stresses[4],cfg.stresses[5]),file=f)
	if desc is not None:
		print ('Feature   from %s' % desc,file=f)
	if (cfg.features != {}):
		keys = list(cfg.features.keys())
		for i in range(len(keys)):
			print ('Feature   ' + keys[i] + ' ' +cfg.features[keys[i]],file=f)	
	print ('END_CFG',file=f)

class cfgparser:
	def __init__(self, file, max_cfgs = None):
		self.cfgs = []
		self.file = file
		self.max_cfgs = max_cfgs
	def __enter__(self):
		while True:
			if self.max_cfgs is not None and len(self.cfgs) == self.max_cfgs:
				break
			cfg = readcfg(self.file)
			if cfg.types is not None: 
				self.cfgs.append(cfg)
			else:
				break
		return self.cfgs
	def __exit__(self, *args):
		self.cfgs = []	

def printcfg(cfg):
	savecfg(None,cfg)

def loadcfgs(filename,max_cfgs = None):
	with open(filename, 'r') as file:
		with cfgparser(file,max_cfgs) as cfgs:
			return cfgs

def savecfgs(filename,cfgs,desc = None):
	with open(filename, 'w') as file:
		for cfg in cfgs:
			savecfg(file,cfg,desc)
			print ("",file=file)

def ase_atom(cfg):
	atom = ase.Atoms(positions=cfg.pos,cell=cfg.lat,numbers=cfg.types,
					pbc=[True,True,True])
	if cfg.energy is not None:
		atom.energy = cfg.energy
	if cfg.forces is not None:
		atom.forces = cfg.forces
	if cfg.stresses is not None:
		atom.stresses = cfg.stresses
	if (cfg.features != {}):
		atom.features = cfg.features
			
	return atom

def ase_cfg(atom):
	cfg = Cfg()
	cfg.pos = atom.positions
	cfg.lat = atom.cell
	cfg.types = atom.numbers
	if hasattr(atom, 'energy'):
		cfg.energy = atom.energy
	if hasattr(atom, 'forces'):
		cfg.forces = atom.forces
	if hasattr(atom, 'stresses'):	
		cfg.stresses = atom.stresses
	if hasattr(atom, 'features'):
		cfg.features = atom.features
	return cfg
	
class ase_atomparser:
	def __init__(self, file, max_cfgs = None):
		self.atoms = []
		self.file = file
		self.max_cfgs = max_cfgs
	def __enter__(self):
		while True:
			if self.max_cfgs is not None and len(self.atoms) == self.max_cfgs:
				break
			cfg = readcfg(self.file)
			if cfg.types is not None :
				atom = ase_atom(cfg)
				self.atoms.append(atom)
			else:
				break
		return self.atoms
	def __exit__(self, *args):
		self.cfgs = []


def ase_loadcfgs(filename,max_cfgs = None):
	with open(filename, 'r') as file:
		with ase_atomparser(file,max_cfgs) as atoms:
			return atoms
		
def ase_savecfgs(filename,atoms,desc = None):
	with open(filename, 'w') as file:
		for atom in atoms:
			cfg = ase_cfg(atom)
			savecfg(file,cfg,desc)
			print ("",file=file)


class MLIP_Calculator:
	mlip = None
	options = None
	implemented_properties = ['energy','forces','stress']
	default_parameters = {}

	all_properties = ['energy', 'forces', 'stress']
	all_changes = ['positions', 'numbers', 'cell', 'pbc']

	def __init__(self,mtp_pot,mlip_opts,restart=None, ignore_bad_restart_file=False, label=None,
				 atoms=None, **kwargs):
	
		self.mlip=mtp_pot
		options = mlip_opts
		options["mtp-filename"]="state.mvs"
		options["calculate_efs"]="TRUE"
		options["ab-initio"]="";
		self.mlip.init_wrapper(options)			

		self.atoms = None  # copy of atoms object from last calculation
		self.results = {}  # calculated properties (energy, forces, ...)
		self.parameters = None  # calculational parameters

		if atoms is not None:
			atoms.calc = self
			if self.atoms is not None:
				# Atoms were read from file.  Update atoms:
				if not (self.equal(atoms.numbers, self.atoms.numbers) and
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
			if key not in self.parameters or not self.equal(value, oldvalue):
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
			if not self.equal(self.atoms.positions, atoms.positions, tol):
				system_changes.append('positions')
			if not self.equal(self.atoms.numbers, atoms.numbers):
				system_changes.append('numbers')
			if not self.equal(self.atoms.cell, atoms.cell, tol):
				system_changes.append('cell')
			if not self.equal(self.atoms.pbc, atoms.pbc):
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

			mlp_res = self.mlip.calc_cfg(self.atoms)		  
			
			if (mlp_res[0]==0):
				self.results = {'energy': mlp_res[1],
						   'forces': mlp_res[2],
						  'stress': -mlp_res[3]}
			else:
				print ('mlip EFS calculation failed')
				sys.exit()
		else:
			print ('atomic system is empty')
			sys.exit()

	@staticmethod
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
			return self.equal(b, a, tol)
		if tol is None:
			return a == b
		return abs(a - b) < tol * abs(b) + tol

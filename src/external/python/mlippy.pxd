cimport mpi4py.MPI as MPI
from mpi4py.libmpi cimport *
from libcpp.vector cimport vector
from libcpp.string  cimport string
from libcpp.map cimport map as cmap
from libcpp cimport bool

cdef extern from "mlip_handler.h":
    ctypedef struct cfg_data:
      int size
      long* types
      double* pos
      double* lat
      double* forces
      double energy
      double* stresses
      int has_energy

    void init()
    void init_mpi(MPI_Comm) # pass an MPI communicator from Python to our C function
    
    void _train(void* pot,const int train_size, const cfg_data *train_cfgs,cmap[string,string] options);
    void _relax(void* pot, const int relax_size, cfg_data *relax_cfgs, int& relaxed_size, int *relaxed,cmap[string,string] options,cmap[string,string] relax_options);
    void _select_add(void* pot, const int train_size, const cfg_data *train_cfgs, const int new_size, const cfg_data *new_cfgs, int& diff_size, int *diff,cmap[string,string] options);
    int _run_command(const string& infname, const string& outfname, int cfg_pos, const string& elements); 		#if cfg_pos = 1, then it writes a poscar file. if cfg_pos=0, then reads an outcar file
    cmap[string,string] _calcerrors(void* pot_addr, const int size, const cfg_data *cfgs, bool on_screen);
	
	
    double _cfg_ene(cfg_data &atom_cfg);
    double _cfg_frc(cfg_data &atom_cfg, int n, int a);
    double _cfg_str(cfg_data &atom_cfg, int n);

    cdef cppclass pot_handler:
        pot_handler()
        void load_potential(const string& fname)
        void save_potential(const string& fname)
        void* get_address();	
        int calc_cfg_efs(cfg_data *atom_cfg);  
        void init_wrapper(cmap[string,string] options);
        vector[int] species_avail();

cdef class mtp: 
    cdef pot_handler *pot
    cdef inline void* get_address(self):
        return <void*>self.pot.get_address()    

cdef inline void* pot_addr(mtp pot):
    return <void*>pot.get_address()



#ifdef MLIP_MPI
#include <mpi.h>
#endif

#include <algorithm>
#include <iostream>

#include "../../../dev_src/mtpr.h"
#include "../../../src/configuration.h"
#include "../../../dev_src/non_linear_regression.h"
#include "../../../src/mlip_wrapper.h"

using namespace std;

extern "C" {

extern void init();
#ifdef MLIP_MPI
extern void init_mpi(MPI_Comm);
#endif
typedef struct _cfg_data {
    int size;
    long* types;
    double* pos;
    double* lat;
    double* forces;
    double energy;
    double* stresses;
	int has_energy;
} cfg_data;

extern void _train( void* pot_addr,
    const int train_size, const cfg_data *train_cfgs, 
    map<string,string> options);
extern void _relax( void* pot_addr, 
    const int relax_size, cfg_data *relax_cfgs, 
    int& relaxed_size, int *relaxed,map<string,string> options,map<string,string> relax_options);
extern void _select_add( void* pot_addr,
	const int train_size, const cfg_data *train_cfgs,
    const int new_size, const cfg_data *new_cfgs, int& diff_size, int *diff,map<string,string> options);
extern int _run_command(const string& infname, const string& outfname, int cfg_pos, const string& elements);					//if cfg_pos = 1, then it writes a poscar file. if cfg_pos=0, then reads an outcar file

extern map<string,string> _calcerrors(void* pot_addr, const int size, const cfg_data *cfgs, bool on_screen);

extern double _cfg_ene(cfg_data &atom_cfg);
extern double _cfg_frc(cfg_data &atom_cfg, int n, int a);
extern double _cfg_str(cfg_data &atom_cfg, int n);

}

class pot_handler {

public:
    MLMTPR *pMtpr;
    MLIP_Wrapper* potWrapper; //for using inside the MLIP_calculator
    int n_types=0;          //number of species in the potential
    int n_coeffs=0;         //number of coeffs in the potential   
public:
    pot_handler();
    ~pot_handler();

    void load_potential(const string& filename);
    void save_potential(const string& filename);
    void init_wrapper(map<string,string> options);
    int calc_cfg_efs(cfg_data *atom_cfg); 

    void* get_address()
        {  return static_cast<void*>(pMtpr); }
};




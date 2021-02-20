///////////////////////////////////////////////////////////////////////////////

#include "../../../src/configuration.h"

using namespace std;

class conf_handler {

public:
    char* my_name;
    vector<Configuration> cfg_array;
    int N_configs;
    int N_types;
    vector<int> types_mapping;

    conf_handler(char* name);
    ~conf_handler();
    
    void  reset();
    void  read_db(char* filename);
    void  write_cfg(char* out_name,int k);
    void  read_bin(char* out_name,int k);
    void  write_bin(char* out_name,int k);
    int     get_size(int k);
    double  get_energy(int k);
    double get_position(int k,int n_atom,int a);
    double get_force(int k,int n_atom,int a);
    double get_lattice(int k,int a,int b);
    double get_stress(int k,int a,int b);
    double get_min_dist(int k);
    int get_type(int k,int n_atom);
	map<string, string> get_features(int k);
	void add_feature(int k, const string& feat_name, const string& feat_val);
	void correct_supercell(int k);
    Configuration get_c_cfg(int k);
    void set_cfg_manual(int k, int size,vector<double> pos,vector<double> lat,vector<int> types,double energy,vector<double> forces,vector<double> stresses);
    void add_type(int atomic_num);                                              //maps an atomic number to the type (0,1,....)
    int atomic_number_from_type(int n);                                         //gets an atomic number corresponding to the type (0,1,...)
    int type_from_atomic_number(int atomic_num);                                //gets a type corresponding to the atomic number
    // convert cfg file to OUTCAR
    int convert_cfg2vasp(const string& input_fn, const string& output_fn);
    // convert POSTCAR to cfg
    int convert_vasp2cfg(const string& input_fn, const string& output_fn);
};

///////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <fstream>   
#include "conf_handler.h"

//namespace mlip {

///////////////////////////////////////////////////////////////////////////////

conf_handler::conf_handler(char* name) {
  my_name=name;
  N_configs=0;
  N_types=0;
}

conf_handler::~conf_handler() { 
  reset();
}

void conf_handler::reset(){
  N_configs=0;
  N_types=0;
  types_mapping.clear();
  cfg_array.clear();
  my_name=nullptr;
}

void conf_handler::read_db(char* file_name) {

    try{

    std::cout<<"Reading database from " << file_name << "...\n";

    std::ifstream ifs(file_name);

    int cntr;
    Configuration cfg; 

    for (cntr = 0; cfg.Load(ifs); cntr++)
        cfg_array.push_back(cfg);

    ifs.close();

    N_configs=cfg_array.size();

    std::cout << cntr << " configurations are found in database" << std::endl;

    }
    catch (MlipException& e) {
                    Warning("Database loading failed: " + e.message);
    }

}

void conf_handler::read_bin(char* out_name,int k) {

    try{

    std::ifstream ifs(out_name,ios::binary);
    cfg_array[k].Load(ifs); 
    ifs.close();

    }
    catch (MlipException& e) {
                    Warning("Reading of cfg from calculation failed: " + e.message);
    }

    remove(out_name);
}

void conf_handler::write_cfg(char* out_name,int k) {

    try{

    std::ofstream ofs(out_name,std::fstream::out | std::fstream::app);
    cfg_array[k].Save(ofs);
    ofs.close();

    }
    catch (MlipException& e) {
                    Warning("Configuration writing failed: " + e.message);
                }

}

void conf_handler::write_bin(char* out_name,int k) {
   
    try{

    std::ofstream ofs(out_name,ios::binary);
    cfg_array[k].Save(ofs,Configuration::SAVE_BINARY);
    ofs.close();

    }
    catch (MlipException& e) {
                    Warning("Writing of cfg for calculation failed: " + e.message);
    }


}

int conf_handler::get_size(int k){

    return cfg_array[k].size();
}

double conf_handler::get_energy(int k){

    return cfg_array[k].energy;
}

double conf_handler::get_position(int k,int n_atom,int a){

    return cfg_array[k].pos(n_atom)[a];
}

double conf_handler::get_force(int k,int n_atom,int a){

    return cfg_array[k].force(n_atom)[a];
}

double conf_handler::get_lattice(int k,int a,int b){

    return cfg_array[k].lattice[a][b];
}

double conf_handler::get_stress(int k,int a,int b){

    return cfg_array[k].stresses[a][b];
}

int conf_handler::get_type(int k,int n_atom){

    return cfg_array[k].type(n_atom);
}
double conf_handler::get_min_dist(int k){

    return cfg_array[k].MinDist();

}
void conf_handler::correct_supercell(int k){

    cfg_array[k].CorrectSupercell();

}
Configuration conf_handler::get_c_cfg(int k){

     return cfg_array[k];
}
map<string, string> conf_handler::get_features(int k){

     return cfg_array[k].features;
}
void conf_handler::add_feature(int k, const string& feat_name, const string& feat_val){

	cfg_array[k].features[feat_name]=feat_val;
}
void conf_handler::set_cfg_manual(int k, int size,vector<double> pos,vector<double> lat,vector<int> types, double energy, vector<double> forces, vector<double> stresses){

    if (k>=cfg_array.size())
        cfg_array.resize(k+1);

    //cfg_array[k].destroy();
    cfg_array[k].resize(size);
	
 	cfg_array[k].has_energy(true);
	cfg_array[k].has_forces(true);
	cfg_array[k].has_stresses(true);

    for (int i=0;i<size;i++)
    {
        for (int j=0;j<3;j++)
			{
            cfg_array[k].pos(i,j) = pos[j*size + i];
			cfg_array[k].force(i,j) = forces[j*size + i];
			}

        cfg_array[k].type(i) = types[i];
    }

	cfg_array[k].energy = energy;

    for (int i=0;i<3;i++)
        for (int j=0;j<3;j++)
            cfg_array[k].lattice[i][j] = lat[i*3+j];

	cfg_array[k].stresses[0][0] = stresses[0];
	cfg_array[k].stresses[1][1] = stresses[1];
	cfg_array[k].stresses[2][2] = stresses[2];
	cfg_array[k].stresses[1][2] = stresses[3];
	cfg_array[k].stresses[0][2] = stresses[4];
	cfg_array[k].stresses[1][0] = stresses[5];

	cfg_array[k].stresses[2][1] = cfg_array[k].stresses[1][2];
	cfg_array[k].stresses[2][0] = cfg_array[k].stresses[0][2];
	cfg_array[k].stresses[0][1] = cfg_array[k].stresses[1][0];

		
}
void conf_handler::add_type(int atomic_num){

    bool present=false;

    for (int i=0;i<N_types;i++)
        if (types_mapping[i]==atomic_num)
            present=true; 

    if (present){
        //std::cout << "You've already added this atomic number! Nothing happens." << std::endl;
    }
    else{
        types_mapping.push_back(atomic_num);
        N_types++;
        std::cout << "Atomic number " << atomic_num << " is now type " << N_types-1 << std::endl;
    }
}
int conf_handler::atomic_number_from_type(int n){

    if (n<types_mapping.size())
        return types_mapping[n];
    else
        return -1;  
}
int conf_handler::type_from_atomic_number(int atomic_num){

    int n=0;

    while (n<N_types){

        if (types_mapping[n]==atomic_num)
            return n;

        n++;
    }   

    add_type(atomic_num);
    
    return -1;  
}

int conf_handler::convert_cfg2vasp(const string& input_fn, const string& output_fn)
{
    int rc = 0;
    try {
        ifstream ifs(input_fn, ios::binary);
        if (ifs.fail())
            rc = 1;
        else
        {
			Warning("Converting cfg to POSCAR in Python should be used with relative species numeration only!");
			vector<int> types_mapping;
			for (int i=0;i<200;i++)
				types_mapping.push_back(i);
	
            int count=0;
            for(Configuration cfg; cfg.Load(ifs);)
            {
                cfg.WriteVaspPOSCAR(output_fn + std::to_string(count),types_mapping);
                count++;
            }
        }
    }
    catch (MlipException& e) {
        Warning(e.message);
        rc = 2;
    }
    return rc;
}

int conf_handler::convert_vasp2cfg(const string& input_fn, const string& output_fn)
{
    int rc = 0;
    try {
        vector<Configuration> db;

        ofstream ofs(output_fn, ios::binary | ios::app);

        if ( ofs.fail() ) return 1;

        if (Configuration::LoadDynamicsFromOUTCAR(db, input_fn))
        {
            for(Configuration& cfg: db)
            {
                cfg.Save(ofs);
            }
        }
        else
            rc = -1;
        ofs.close();
    }
    catch (MlipException& e) {
        Warning(e.message);
        rc = 2;
    }
    return rc;
}

///////////////////////////////////////////////////////////////////////////////

//}

// End of the file



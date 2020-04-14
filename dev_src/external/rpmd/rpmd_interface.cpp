/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 */

#include "../../../src/mlip_wrapper.h"
#include <fstream>
#include <iostream>
#include <string>

MLIP_Wrapper *MLIP_wrp = nullptr;
std::vector<Configuration> comm_cfg;
Configuration cfg_curr;
// int temp_count = 0;
// std::ofstream temp_cfg_stream;

//CN+CH4
//const int MLIP4PRMD_types[] = {0,1,0,0,0,1,2};
//OH+H2
//const int MLIP4PRMD_types[] = {0,1,1,1};

extern "C" 
void mlip_rpmd_init_()
{
	/*std::ifstream ifs("train.cfg");
	std::ofstream ofs("train2.cfg");

	Configuration cfg;
	for (int i = 0; cfg.Load(ifs); i++) {
		if (i % 1000 == 0) cfg.Save(ofs);
	}
	
	ofs.close();
	ifs.close();*/

	std::ios_base::sync_with_stdio();
	//temp_cfg_stream.open("temp.cfg");
	if (MLIP_wrp != nullptr)
		delete MLIP_wrp;
	try {
		std::string fnm = "mlip.ini";
		MLIP_wrp = new MLIP_Wrapper(LoadSettings(fnm));
		std::cout << "MLIP potential initialized successfully!" << std::endl;
	}
	catch (MlipException& exception) {
		Message(exception.What());
		exit(9991);
	}
}

extern "C" 
void mlip_rpmd_finalize_()
{
	try
	{
		delete MLIP_wrp;
	}
	catch (MlipException& exception)
	{
		Message(exception.What());
		exit(9994);
	}
	MLIP_wrp = nullptr;
	Message("RPMD-to-MLIP link has been terminated\n");
}

extern "C"
void mlip_rpmd_calc_new_(double *_coords, int * types_, double *_xi, int *_Natoms, int *_bead, double *_V, double *_dVdq, int *_info)
{
        int bead = *_bead;
        const int Natoms = *_Natoms;
	bead = bead-1;

        //if(Nbeads<=0 || Natoms <=0) ERROR("Nbeads<=0 || Natoms <=0");

	Configuration cfg;
 
	{       
        	cfg.resize(Natoms);
        	cfg.has_energy(true);	
        	cfg.has_forces(true);
        	cfg.lattice = Matrix3(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0);
	}

        //if(sizeof(MLIP4PRMD_types)/sizeof(int) != Natoms)
        //	ERROR("Natoms should be " + to_string(sizeof(MLIP4PRMD_types)/sizeof(int)));

	{
       		for(int j=0; j<Natoms; j++) {
        	//	cfg.type(j) = MLIP4PRMD_types[j];
			cfg.type(j) = types_[j];
		}
	}

        int counter = bead*Natoms*3;
	
	{
        	for(int i=0; i<Natoms; i++)
        		for(int a=0; a<3; a++)
                		cfg.pos(i,a) = _coords[counter++];
	}

        counter = bead*Natoms*3;
        double xi = *_xi;
        int info = 0;

        cfg.FromAtomicUnits();
	Configuration cfg_curr = cfg;
	cfg_curr.has_energy(false);	
        cfg_curr.has_forces(false);
	cfg_curr.features["xi"] = to_string(xi);

        try
        {
        	//MLIP_wrp->CalcEFS2(cfg, xi, bead, info);
		MLIP_wrp->CalcEFS(cfg);
		cfg_curr.features["MV_grade"] = cfg.features["MV_grade"];
		/*if (stod(cfg_curr.features["MV_grade"]) > MLIP_wrp->Threshold()) {
			std::string fnm;
            		if (bead < 10) {
                    		fnm = "Selected/selected00"+to_string(bead)+"_"+to_string(xi)+".cfg";
            		}
            		else if (bead >= 10 && bead < 100) {
                    		fnm = "Selected/selected0"+to_string(bead)+"_"+to_string(xi)+".cfg";
            		}
            		else {
                    		fnm = "Selected/selected"+to_string(bead)+"_"+to_string(xi)+".cfg";
            		}
		
            		std::ofstream ofs(fnm);
            		cfg_curr.Save(ofs);
            		ofs.close();
		}*/
		//std::cout << cfg_curr.features["MV_grade"] << std::endl;
		if (stod(cfg_curr.features["MV_grade"]) > MLIP_wrp->Threshold()) 
            		cfg_curr.AppendToFile("selected.cfg");
		if (stod(cfg_curr.features["MV_grade"]) > MLIP_wrp->ThresholdBreak())
			info = 2;
        }
        catch (MlipException& excp)
        {
                std::cout << excp.What() << std::endl;
                std::cout << "Calculation and selection terminated" << std::endl;
		cfg_curr.features["MV_grade"] = cfg.features["MV_grade"];
                /*if (stod(cfg_curr.features["MV_grade"]) > MLIP_wrp->Threshold()) {
                        std::string fnm;
                        if (bead < 10) {
                                fnm = "Selected/selected00"+to_string(bead)+"_"+to_string(xi)+".cfg";
                        }
                        else if (bead >= 10 && bead < 100) {
                                fnm = "Selected/selected0"+to_string(bead)+"_"+to_string(xi)+".cfg";
                        }
                        else {
                                fnm = "Selected/selected"+to_string(bead)+"_"+to_string(xi)+".cfg";
                        }
                
                        std::ofstream ofs(fnm);
                        cfg_curr.Save(ofs);
                        ofs.close();
                }*/
                if (stod(cfg_curr.features["MV_grade"]) > MLIP_wrp->Threshold())                
        		cfg_curr.AppendToFile("selected.cfg");
                if (stod(cfg_curr.features["MV_grade"]) > MLIP_wrp->ThresholdBreak())
                        info = 2;
                //exit(2048);
        }

        cfg.ToAtomicUnits();
        _V[bead] = cfg.energy;
        for(int i=0; i<Natoms; i++)
        	for(int a=0; a<3; a++)
                	_dVdq[counter++] = -cfg.force(i,a);

        *_info = info;

}

extern "C"
void save_cfgs_(char * output_fnm, int * natoms_, int * _bead, double * cart_1d_all, int * types_,
double * energy_all, double * forces_1d_all, double * _mindist, double * _xi)
{
	if (*_xi < -0.98 && *_xi > -1.02)
		output_fnm = "/scratch/mmm/statistics-100.txt";
        if (*_xi < 0.02 && *_xi > -0.02)
                output_fnm = "/scratch/mmm/statisticsa0.txt";
        if (*_xi < 0.14 && *_xi > 0.10)
                output_fnm = "/scratch/mmm/statistics12.txt";
        if (*_xi < 0.56 && *_xi > 0.52)
                output_fnm = "/scratch/mmm/statistics54.txt";


        std::ofstream ofs;
	ofs.open(output_fnm, std::ofstream::app);
	
	
        int bead = *_bead;
        bead = bead-1;
        int natoms = *natoms_;

                Configuration cfg;
                cfg.resize(natoms);
                cfg.has_energy(true);
                cfg.has_forces(true);
                cfg.lattice = Matrix3(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0);
                cfg.energy = energy_all[bead];
		int counter = bead*natoms*3;
                for (int i = 0; i < cfg.size(); i++) {
                        //cfg.type(i) = MLIP4PRMD_types[i];
                        cfg.type(i) = types_[i];
                        for (int a = 0; a < 3; a++) {
                                cfg.pos(i)[a] = cart_1d_all[counter];
                                cfg.force(i)[a] = forces_1d_all[counter];
				counter++;
                        }
                }
                cfg.FromAtomicUnits();

                cfg.features["mindist"] = *_mindist;
                double dist_h_h = 0;
                for (int a = 0; a < 3; a++) {
                        dist_h_h += (cfg.pos(2)[a] - cfg.pos(3)[a])*(cfg.pos(2)[a] - cfg.pos(3)[a]);
                }

                dist_h_h = sqrt(dist_h_h);
                cfg.features["dist_h_h"] = dist_h_h;

                double centroid1_2[3];
                double centroid3_4[3];

                for (int a = 0; a < 3; a++) {
                        centroid1_2[a] = (cfg.pos(0)[a] + cfg.pos(1)[a])/2;
                        centroid3_4[a] = (cfg.pos(2)[a] + cfg.pos(3)[a])/2;
                }

                double dist_between_centroids = 0;

                for (int a = 0; a < 3; a++) {
                        dist_between_centroids += (centroid1_2[a] - centroid3_4[a])*(centroid1_2[a] - centroid3_4[a]);
                }

                dist_between_centroids = sqrt(dist_between_centroids);
                cfg.features["dist_centroids"] = dist_between_centroids;
		ofs << *_mindist << " " << dist_h_h << " " << dist_between_centroids << " " << *_xi << " ";
		ofs << cfg.energy << " " << bead+1 << std::endl;
		//cfg.Save(ofs);

        ofs.close();
}


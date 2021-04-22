/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   This file contributors: Alexander Shapeev, Ivan Novikov
 */

#include "../../../src/mlip_wrapper.h"
#include <fstream>
#include <iostream>
#include <string>

MLIP_Wrapper *MLIP_wrp = nullptr;
std::vector<Configuration> comm_cfg;
Configuration cfg_curr;

extern "C" 
void mlip_rpmd_init_()
{
    std::ios_base::sync_with_stdio();
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
void mlip_rpmd_calc_new_(double *_coords, int * types_, double *_xi, double * _xi_current, int *_Natoms, int *_bead, double *_V, double *_dVdq, int *_info)
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
        cfg.has_stresses(false);
        cfg.lattice = Matrix3(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0);
    }

    {
        for(int j=0; j<Natoms; j++) 
            cfg.type(j) = types_[j];
    }

    int counter = bead*Natoms*3;
    {
        for(int i=0; i<Natoms; i++) 
            for(int a=0; a<3; a++)
                cfg.pos(i,a) = _coords[counter++];
    }
        
    counter = bead*Natoms*3;
    double xi = *_xi;
    double xi_current = *_xi_current;
    int info = 0;

    cfg.FromAtomicUnits();
    Configuration cfg_curr = cfg;
    cfg_curr.has_energy(false); 
    cfg_curr.has_forces(false);
    cfg_curr.has_stresses(false);
    cfg_curr.features["xi"] = to_string(xi);

    try
    {
        //MLIP_wrp->CalcEFS2(cfg, xi, bead, info);
        MLIP_wrp->CalcEFS(cfg);
        cfg_curr.features["MV_grade"] = cfg.features["MV_grade"];
        std::string grade_fnm;
        /*grade_fnm = "grade_for_xi_"+std::to_string(xi_current)+"_bead_"+std::to_string(bead)+".txt";
        std::ofstream ofs(grade_fnm);
        ofs << cfg.features["MV_grade"] << std::endl;
        ofs.close();*/

        if (stod(cfg_curr.features["MV_grade"]) > MLIP_wrp->Threshold()) {
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
        }
        //std::cout << cfg_curr.features["MV_grade"] << std::endl;
        //if (stod(cfg_curr.features["MV_grade"]) > MLIP_wrp->Threshold()) 
        //    cfg_curr.AppendToFile("selected.cfg");
        if (stod(cfg_curr.features["MV_grade"]) > MLIP_wrp->ThresholdBreak())
            info = 2;
    }
    catch (MlipException& excp)
    {
        std::cout << excp.What() << std::endl;
        std::cout << "Calculation and selection terminated" << std::endl;
        cfg_curr.features["MV_grade"] = cfg.features["MV_grade"];
        std::string grade_fnm;
        grade_fnm = "grade_for_xi_"+std::to_string(xi_current)+"_bead_"+std::to_string(bead)+".txt";
        std::ofstream ofs(grade_fnm);
        ofs << "1" << std::endl;
        ofs.close();
        if (stod(cfg_curr.features["MV_grade"]) > MLIP_wrp->Threshold()) {
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
        }
        //if (stod(cfg_curr.features["MV_grade"]) > MLIP_wrp->Threshold())
        //    cfg_curr.AppendToFile("selected.cfg");
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


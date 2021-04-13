#include "mtpr_plus_zbl.h"
#ifdef MLIP_MPI
#	include <mpi.h>
#endif

using namespace std;

void MTPplusZBL::CalcE(Configuration& cfg)
{
    ResetEFS(cfg);

    Configuration cfg_mtpr = cfg;
    Configuration cfg_zbl = cfg;

    mtpr->CalcEFS(cfg_mtpr);
    zbl->CalcEFS(cfg_zbl);

    cfg.energy = cfg_zbl.energy + cfg_mtpr.energy;
}

void MTPplusZBL::CalcEFS(Configuration& cfg)
{
    ResetEFS(cfg);

    Configuration cfg_mtpr = cfg;
    Configuration cfg_zbl = cfg;

    mtpr->CalcEFS(cfg_mtpr);
    zbl->CalcEFS(cfg_zbl);
    
    cfg.energy = cfg_mtpr.energy + cfg_zbl.energy;
	
    for (int i = 0; i < cfg.size(); i++) {
        for (int a = 0; a < 3; a++) 
            cfg.force(i)[a] = cfg_zbl.force(i)[a] + cfg_mtpr.force(i)[a];
    }

    cfg.stresses = cfg_zbl.stresses + cfg_mtpr.stresses;
}

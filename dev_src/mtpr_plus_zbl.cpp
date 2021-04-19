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

    p_mtpr->CalcEFS(cfg_mtpr);
    p_zbl->CalcEFS(cfg_zbl);

    cfg.energy = cfg_zbl.energy + cfg_mtpr.energy;
}

void MTPplusZBL::CalcEFS(Configuration& cfg)
{
    ResetEFS(cfg);

    Configuration cfg_mtpr = cfg;
    Configuration cfg_zbl = cfg;

    p_mtpr->CalcEFS(cfg_mtpr);
    p_zbl->CalcEFS(cfg_zbl);
    
    cfg.energy = cfg_mtpr.energy + cfg_zbl.energy;
	
    for (int i = 0; i < cfg.size(); i++) {
        for (int a = 0; a < 3; a++) 
            cfg.force(i)[a] = cfg_zbl.force(i)[a] + cfg_mtpr.force(i)[a];
    }

    cfg.stresses = cfg_zbl.stresses + cfg_mtpr.stresses;
}

void MTPplusZBL::AccumulateCombinationGrad(const Neighborhood& nbh,
				            std::vector<double>& out_grad_accumulator,
				            const double se_weight,
				            const Vector3* se_ders_weights)
{
    p_mtpr->AccumulateCombinationGrad(nbh, out_grad_accumulator, se_weight, se_ders_weights);
} 

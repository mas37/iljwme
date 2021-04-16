#ifndef MTP_PLUS_ZBL
#define MTP_PLUS_ZBL

#include "../src/pair_potentials.h"
#include "mtpr.h"

class MTPplusZBL : public AnyLocalMLIP
{

protected:
    std::ostream* p_logstrm;

public:
    void CalcE(Configuration&); 
    void CalcEFS(Configuration&); 														

    MLMTPR* p_mtpr = nullptr;
    ZBL* p_zbl = nullptr;

    MTPplusZBL(MLMTPR* _p_mtpr, ZBL* _p_zbl, std::ostream* _p_logstrm = nullptr)
    { 
        p_mtpr = _p_mtpr;
        p_zbl = _p_zbl;
        p_RadialBasis = p_mtpr->p_RadialBasis;
        p_logstrm = _p_logstrm;
    };
    
    int CoeffCount() //!< number of coefficients
    {
        return p_mtpr->CoeffCount();
    }
    
    double* Coeff() //!< coefficients themselves
    {
        return &p_mtpr->Coeff()[0];
    }
    
    void AccumulateCombinationGrad(const Neighborhood&, std::vector<double>&,
				    const double se_weight = 0.0,
				    const Vector3* se_ders_weights = nullptr) override;

    ~MTPplusZBL() {};
										
};

#endif // MTP_PLUS_ZBL

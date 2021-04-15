#ifndef MTP_PLUS_ZBL
#define MTP_PLUS_ZBL

#include "../src/pair_potentials.h"
#include "mtpr.h"

class MTPplusZBL : public AnyPotential
{

protected:
    std::ostream* p_logstrm;

public:
    void CalcE(Configuration&); 
    void CalcEFS(Configuration&); 														

    MLMTPR* p_mtpr;// = nullptr;
    ZBL* p_zbl;// = nullptr;

    MTPplusZBL(MLMTPR* _p_mtpr, ZBL* _p_zbl, std::ostream* _p_logstrm = nullptr)
    { 
        p_mtpr = _p_mtpr;
        p_zbl = _p_zbl;
        p_logstrm = _p_logstrm;
    };

    ~MTPplusZBL() {};
										
};

#endif // MTP_PLUS_ZBL

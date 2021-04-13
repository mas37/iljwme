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

    MLMTPR* mtpr;// = nullptr;
    ZBL* zbl;// = nullptr;

    MTPplusZBL(MLMTPR* _mtpr, ZBL* _zbl, std::ostream* _p_logstrm = nullptr)
    { 
        mtpr = _mtpr;
        zbl = _zbl;
        p_logstrm = _p_logstrm;
    };

    ~MTPplusZBL() {};
										
};

#endif // MTP_PLUS_ZBL

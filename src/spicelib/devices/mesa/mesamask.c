/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Thomas L. Quarles
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "ngspice/ifsim.h"
#include "mesadefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


/* ARGSUSED */
int
MESAmAsk(CKTcircuit *ckt, GENmodel *inst, int which, IFvalue *value)
{
    MESAmodel *here = (MESAmodel*)inst;

    NG_IGNORE(ckt);

    switch(which) {
        case MESA_MOD_VTO:
            value->rValue = here->MESAthreshold;
            return (OK);
        case MESA_MOD_VS:
            value->rValue = here->MESAvs;
            return (OK);
        case MESA_MOD_ALPHA:
            value->rValue = here->MESAalpha;
            return (OK);
        case MESA_MOD_BETA:
            value->rValue = here->MESAbeta;
            return (OK);
        case MESA_MOD_LAMBDA:
            value->rValue = here->MESAlambda;
            return (OK);
        case MESA_MOD_RG:
            value->rValue = here->MESAgateResist;
            return (OK);
        case MESA_MOD_RD:
            value->rValue = here->MESAdrainResist;
            return (OK);
        case MESA_MOD_RS:
            value->rValue = here->MESAsourceResist;
            return (OK);
        case MESA_MOD_RI:
            value->rValue = here->MESAri;
            return (OK);
        case MESA_MOD_RF:
            value->rValue = here->MESArf;
            return (OK);
        case MESA_MOD_RDI:
            value->rValue = here->MESArdi;
            return (OK);
        case MESA_MOD_RSI:
            value->rValue = here->MESArsi;
            return (OK);
        case MESA_MOD_PHIB:
            value->rValue = here->MESAphib;
            return (OK);
        case MESA_MOD_PHIB1:
            value->rValue = here->MESAphib1;
            return (OK);
        case MESA_MOD_ASTAR:
            value->rValue = here->MESAastar;
            return (OK);
        case MESA_MOD_GGR:
            value->rValue = here->MESAggr;
            return (OK);
        case MESA_MOD_DEL:
            value->rValue = here->MESAdel;
            return (OK);
        case MESA_MOD_XCHI:
            value->rValue = here->MESAxchi;
            return (OK);
        case MESA_MOD_N:
            value->rValue = here->MESAn;
            return (OK);
        case MESA_MOD_ETA:
            value->rValue = here->MESAeta;
            return (OK);
        case MESA_MOD_M:
            value->rValue = here->MESAm;
            return (OK);
        case MESA_MOD_MC:
            value->rValue = here->MESAmc;
            return (OK);
        case MESA_MOD_SIGMA0:
            value->rValue = here->MESAsigma0;
            return (OK);
        case MESA_MOD_VSIGMAT:
            value->rValue = here->MESAvsigmat;
            return (OK);
        case MESA_MOD_VSIGMA:
            value->rValue = here->MESAvsigma;
            return (OK);
        case MESA_MOD_MU:
            value->rValue = here->MESAmu;
            return (OK);
        case MESA_MOD_MU1:
            value->rValue = here->MESAmu1;
            return (OK);
        case MESA_MOD_MU2:
            value->rValue = here->MESAmu2;
            return (OK);
        case MESA_MOD_D:
            value->rValue = here->MESAd;
            return (OK);
        case MESA_MOD_ND:
            value->rValue = here->MESAnd;
            return (OK);
        case MESA_MOD_DELTA:
            value->rValue = here->MESAdelta;
            return (OK);
        case MESA_MOD_TC:
            value->rValue = here->MESAtc;
            return (OK);
        case MESA_MOD_TVTO:
            value->rValue = here->MESAtvto;
            return (OK);
        case MESA_MOD_TLAMBDA:
            value->rValue = here->MESAtlambda;
            return (OK);
        case MESA_MOD_TETA0:
            value->rValue = here->MESAteta0;
            return (OK);
        case MESA_MOD_TETA1:
            value->rValue = here->MESAteta1;
            return (OK);
        case MESA_MOD_TMU:
            value->rValue = here->MESAtmu;
            return (OK);
        case MESA_MOD_XTM0:
            value->rValue = here->MESAxtm0;
            return (OK);
        case MESA_MOD_XTM1:
            value->rValue = here->MESAxtm1;
            return (OK);
        case MESA_MOD_XTM2:
            value->rValue = here->MESAxtm2;
            return (OK);
        case MESA_MOD_KS:
            value->rValue = here->MESAks;
            return (OK);
        case MESA_MOD_VSG:
            value->rValue = here->MESAvsg;
            return (OK);
        case MESA_MOD_LAMBDAHF:
            value->rValue = here->MESAlambdahf;
            return (OK);
        case MESA_MOD_TF:
            value->rValue = here->MESAtf;
            return (OK);
        case MESA_MOD_FLO:
            value->rValue = here->MESAflo;
            return (OK);
        case MESA_MOD_DELFO:
            value->rValue = here->MESAdelfo;
            return (OK);
        case MESA_MOD_AG:
            value->rValue = here->MESAag;
            return (OK);
        case MESA_MOD_THETA:
            value->rValue = here->MESAtheta;
            return (OK);
        case MESA_MOD_TC1:
            value->rValue = here->MESAtc1;
            return (OK);
        case MESA_MOD_TC2:
            value->rValue = here->MESAtc2;
            return (OK);
        case MESA_MOD_ZETA:
            value->rValue = here->MESAzeta;
            return (OK);
        case MESA_MOD_DU:
            value->rValue = here->MESAdu;
            return (OK);
        
        
        
        case MESA_MOD_NDU:
            value->rValue = here->MESAndu;
            return (OK);
        case MESA_MOD_TH:
            value->rValue = here->MESAth;
            return (OK);
        case MESA_MOD_NDELTA:
            value->rValue = here->MESAndelta;
            return (OK);
        case MESA_MOD_LEVEL:
            value->rValue = here->MESAlevel;
            return (OK);
        case MESA_MOD_NMAX:
            value->rValue = here->MESAnmax;
            return (OK);
        case MESA_MOD_GAMMA:
            value->rValue = here->MESAgamma;
            return (OK);
        case MESA_MOD_EPSI:
            value->rValue = here->MESAepsi;
            return (OK);
        case MESA_MOD_CBS:
            value->rValue = here->MESAcbs;
            return (OK);
        case MESA_MOD_CAS:
            value->rValue = here->MESAcas;
            return (OK);  
        case MESA_MOD_TYPE:
            if (here->MESAtype == NMF)
                value->sValue = "nmf";
            else
                value->sValue = "pmf";
            return (OK);
        case MESA_MOD_DRAINCONDUCT:
            value->rValue = here->MESAdrainConduct;
            return (OK);
        case MESA_MOD_SOURCECONDUCT:
            value->rValue = here->MESAsourceConduct;
            return (OK);
        case MESA_MOD_VCRIT:
            value->rValue = here->MESAvcrit;
            return (OK); 
        default:
            return (E_BADPARM);
    }
    /* NOTREACHED */
}

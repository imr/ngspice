/**********
STAG version 2.7
Copyright 2000 owned by the United Kingdom Secretary of State for Defence
acting through the Defence Evaluation and Research Agency.
Developed by :     Jim Benson,
                   Department of Electronics and Computer Science,
                   University of Southampton,
                   United Kingdom.
With help from :   Nele D'Halleweyn, Ketan Mistry, Bill Redman-White, and Craig Easson.

Based on STAG version 2.1
Developed by :     Mike Lee,
With help from :   Bernard Tenbroek, Bill Redman-White, Mike Uren, Chris Edwards
                   and John Bunyan.
Acknowledgements : Rupert Howes and Pete Mole.
**********/

/********** 
Modified by Paolo Nenzi 2002
ngspice integration
**********/

#include "ngspice/ngspice.h"
#include "ngspice/const.h"
#include "ngspice/ifsim.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "soi3defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


/*ARGSUSED*/
int
SOI3mAsk(CKTcircuit *ckt, GENmodel *inst, int which, IFvalue *value)
{
    SOI3model *model = (SOI3model *)inst;

    NG_IGNORE(ckt);

    switch(which) {
        case SOI3_MOD_VTO:
            value->rValue = model->SOI3vt0;
            return(OK);
        case SOI3_MOD_VFBF:
            value->rValue = model->SOI3vfbF;
            return(OK);
        case SOI3_MOD_KP:
            value->rValue = model->SOI3transconductance;
            return(OK);
        case SOI3_MOD_GAMMA:
            value->rValue = model->SOI3gamma;
            return(OK);
        case SOI3_MOD_PHI:
            value->rValue = model->SOI3phi;
            return(OK);
        case SOI3_MOD_LAMBDA:
            value->rValue = model->SOI3lambda;
            return(OK);
        case SOI3_MOD_THETA:
            value->rValue = model->SOI3theta;
            return(OK);
        case SOI3_MOD_RD:
            value->rValue = model->SOI3drainResistance;
            return(OK);
        case SOI3_MOD_RS:
            value->rValue = model->SOI3sourceResistance;
            return(OK);
        case SOI3_MOD_CBD:
            value->rValue = model->SOI3capBD;
            return(OK);
        case SOI3_MOD_CBS:
            value->rValue = model->SOI3capBS;
            return(OK);                                  
        case SOI3_MOD_IS:
            value->rValue = model->SOI3jctSatCur;
            return(OK);
        case SOI3_MOD_IS1:
            value->rValue = model->SOI3jctSatCur1;
            return(OK);
        case SOI3_MOD_PB:
            value->rValue = model->SOI3bulkJctPotential;
            return(OK);
        case SOI3_MOD_CGFSO:
            value->rValue = model->SOI3frontGateSourceOverlapCapFactor;
            return(OK);
        case SOI3_MOD_CGFDO:
            value->rValue = model->SOI3frontGateDrainOverlapCapFactor;
            return(OK);
        case SOI3_MOD_CGFBO:
            value->rValue = model->SOI3frontGateBulkOverlapCapFactor;
            return(OK);
        case SOI3_MOD_CGBSO:
            value->rValue = model->SOI3backGateSourceOverlapCapAreaFactor;
            return(OK);
        case SOI3_MOD_CGBDO:
            value->rValue = model->SOI3backGateDrainOverlapCapAreaFactor;
            return(OK);
        case SOI3_MOD_CGBBO:
            value->rValue = model->SOI3backGateBulkOverlapCapAreaFactor;
            return(OK);
        case SOI3_MOD_RSH:
            value->rValue = model->SOI3sheetResistance;
            return(OK);
        case SOI3_MOD_CJSW:
            value->rValue = model->SOI3sideWallCapFactor;
            return(OK);
        case SOI3_MOD_MJSW:
            value->rValue = model->SOI3bulkJctSideGradingCoeff;
            return(OK);                         
        case SOI3_MOD_JS:
            value->rValue = model->SOI3jctSatCurDensity;
            return(OK);
        case SOI3_MOD_JS1:
            value->rValue = model->SOI3jctSatCurDensity1;
            return(OK);
        case SOI3_MOD_TOF:
            value->rValue = model->SOI3frontOxideThickness;
            return(OK);
        case SOI3_MOD_TOB:
            value->rValue = model->SOI3backOxideThickness;
            return(OK);
        case SOI3_MOD_TB:
            value->rValue = model->SOI3bodyThickness;
            return(OK);
        case SOI3_MOD_LD:
            value->rValue = model->SOI3latDiff;
            return(OK);
        case SOI3_MOD_U0:
            value->rValue = model->SOI3surfaceMobility;
            return(OK);
        case SOI3_MOD_FC:
            value->rValue = model->SOI3fwdCapDepCoeff;
            return(OK);
        case SOI3_MOD_KOX:
            value->rValue = model->SOI3oxideThermalConductivity;
            return(OK);
        case SOI3_MOD_SHSI:
            value->rValue = model->SOI3siliconSpecificHeat;
            return(OK);
        case SOI3_MOD_DSI:
            value->rValue = model->SOI3siliconDensity;
            return(OK);
        case SOI3_MOD_NSUB:
            value->rValue = model->SOI3substrateDoping;
            return(OK);
        case SOI3_MOD_TPG:
            value->iValue = model->SOI3gateType;
            return(OK);
        case SOI3_MOD_NQFF:
            value->rValue = model->SOI3frontFixedChargeDensity;
            return(OK);
        case SOI3_MOD_NQFB:
            value->rValue = model->SOI3backFixedChargeDensity;
            return(OK);
        case SOI3_MOD_NSSF:
            value->rValue = model->SOI3frontSurfaceStateDensity;
            return(OK);
        case SOI3_MOD_NSSB:
            value->rValue = model->SOI3backSurfaceStateDensity;
            return(OK);
        case SOI3_MOD_TNOM:
            value->rValue = model->SOI3tnom-CONSTCtoK;
            return(OK);
/* extra stuff for newer model - msll Jan96 */
        case SOI3_MOD_SIGMA:
            value->rValue = model->SOI3sigma;
            return(OK);
        case SOI3_MOD_CHIFB:
            value->rValue = model->SOI3chiFB;
            return(OK);
        case SOI3_MOD_CHIPHI:
            value->rValue = model->SOI3chiPHI;
            return(OK);
        case SOI3_MOD_DELTAW:
            value->rValue = model->SOI3deltaW;
            return(OK);
        case SOI3_MOD_DELTAL:
            value->rValue = model->SOI3deltaL;
            return(OK);
        case SOI3_MOD_VSAT:
            value->rValue = model->SOI3vsat;
            return(OK);
        case SOI3_MOD_K:
            value->rValue = model->SOI3k;
            return(OK);
        case SOI3_MOD_LX:
            value->rValue = model->SOI3lx;
            return(OK);
        case SOI3_MOD_VP:
            value->rValue = model->SOI3vp;
            return(OK);
        case SOI3_MOD_ETA:
            value->rValue = model->SOI3eta;
            return(OK);
        case SOI3_MOD_ALPHA0:
            value->rValue = model->SOI3alpha0;
            return(OK);
        case SOI3_MOD_BETA0:
            value->rValue = model->SOI3beta0;
            return(OK);
        case SOI3_MOD_LM:
            value->rValue = model->SOI3lm;
            return(OK);
        case SOI3_MOD_LM1:
            value->rValue = model->SOI3lm1;
            return(OK);
        case SOI3_MOD_LM2:
            value->rValue = model->SOI3lm2;
            return(OK);
        case SOI3_MOD_ETAD:
            value->rValue = model->SOI3etad;
            return(OK);
        case SOI3_MOD_ETAD1:
            value->rValue = model->SOI3etad1;
            return(OK);
        case SOI3_MOD_CHIBETA:
            value->rValue = model->SOI3chibeta;
            return(OK);
        case SOI3_MOD_VFBB:
            value->rValue = model->SOI3vfbB;
            return(OK);
        case SOI3_MOD_GAMMAB:
            value->rValue = model->SOI3gammaB;
            return(OK);
        case SOI3_MOD_CHID:
            value->rValue = model->SOI3chid;
            return(OK);
        case SOI3_MOD_CHID1:
            value->rValue = model->SOI3chid1;
            return(OK);
        case SOI3_MOD_DVT:
            value->iValue = model->SOI3dvt;
            return(OK);
        case SOI3_MOD_NLEV:
            value->iValue = model->SOI3nLev;
            return(OK);
        case SOI3_MOD_BETABJT:
            value->rValue = model->SOI3betaBJT;
            return(OK);
        case SOI3_MOD_TAUFBJT:
            value->rValue = model->SOI3tauFBJT;
            return(OK);
        case SOI3_MOD_TAURBJT:
            value->rValue = model->SOI3tauRBJT;
            return(OK);
        case SOI3_MOD_BETAEXP:
            value->rValue = model->SOI3betaEXP;
            return(OK);
        case SOI3_MOD_TAUEXP:
            value->rValue = model->SOI3tauEXP;
            return(OK);
        case SOI3_MOD_RSW:
            value->rValue = model->SOI3rsw;
            return(OK);
        case SOI3_MOD_RDW:
            value->rValue = model->SOI3rdw;
            return(OK);
        case SOI3_MOD_FMIN:
            value->rValue = model->SOI3minimumFeatureSize;
            return(OK);
        case SOI3_MOD_VTEX:
            value->rValue = model->SOI3vtex;
            return(OK);
        case SOI3_MOD_VDEX:
            value->rValue = model->SOI3vdex;
            return(OK);
        case SOI3_MOD_DELTA0:
            value->rValue = model->SOI3delta0;
            return(OK);
        case SOI3_MOD_CSF:
            value->rValue = model->SOI3satChargeShareFactor;
            return(OK);
        case SOI3_MOD_NPLUS:
            value->rValue = model->SOI3nplusDoping;
            return(OK);
        case SOI3_MOD_RTA:
            value->rValue = model->SOI3rta;
            return(OK);
        case SOI3_MOD_CTA:
            value->rValue = model->SOI3cta;
            return(OK);
	case SOI3_MOD_MEXP:
            value->rValue = model->SOI3mexp;
            return(OK);
        default:
            return(E_BADPARM);
    }
    /* NOTREACHED */
}

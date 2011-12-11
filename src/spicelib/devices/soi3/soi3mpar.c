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
#include "soi3defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
SOI3mParam(int param, IFvalue *value, GENmodel *inModel)
{
    register SOI3model *model = (SOI3model *)inModel;
    switch(param) {
        case SOI3_MOD_VTO:
            model->SOI3vt0 = value->rValue;
            model->SOI3vt0Given = TRUE;
            break;
        case SOI3_MOD_VFBF:
            model->SOI3vfbF = value->rValue;
            model->SOI3vfbFGiven = TRUE;
            break;
        case SOI3_MOD_KP:
            model->SOI3transconductance = value->rValue;
            model->SOI3transconductanceGiven = TRUE;
            break;
        case SOI3_MOD_GAMMA:
            model->SOI3gamma = value->rValue;
            model->SOI3gammaGiven = TRUE;
            break;
        case SOI3_MOD_PHI:
            model->SOI3phi = value->rValue;
            model->SOI3phiGiven = TRUE;
            break;
        case SOI3_MOD_LAMBDA:
            model->SOI3lambda = value->rValue;
            model->SOI3lambdaGiven = TRUE;
            break;
        case SOI3_MOD_THETA:
            model->SOI3theta = value->rValue;
            model->SOI3thetaGiven = TRUE;
            break;
        case SOI3_MOD_RD:
            model->SOI3drainResistance = value->rValue;
            model->SOI3drainResistanceGiven = TRUE;
            break;
        case SOI3_MOD_RS:
            model->SOI3sourceResistance = value->rValue;
            model->SOI3sourceResistanceGiven = TRUE;
            break;
        case SOI3_MOD_CBD:
            model->SOI3capBD = value->rValue;
            model->SOI3capBDGiven = TRUE;
            break;
        case SOI3_MOD_CBS:
            model->SOI3capBS = value->rValue;
            model->SOI3capBSGiven = TRUE;
            break;
        case SOI3_MOD_IS:
            model->SOI3jctSatCur = value->rValue;
            model->SOI3jctSatCurGiven = TRUE;
            break;
        case SOI3_MOD_IS1:
            model->SOI3jctSatCur1 = value->rValue;
            model->SOI3jctSatCur1Given = TRUE;
            break;
        case SOI3_MOD_PB:
            model->SOI3bulkJctPotential = value->rValue;
            model->SOI3bulkJctPotentialGiven = TRUE;
            break;
        case SOI3_MOD_CGFSO:
            model->SOI3frontGateSourceOverlapCapFactor = value->rValue;
            model->SOI3frontGateSourceOverlapCapFactorGiven = TRUE;
            break;
        case SOI3_MOD_CGFDO:
            model->SOI3frontGateDrainOverlapCapFactor = value->rValue;
            model->SOI3frontGateDrainOverlapCapFactorGiven = TRUE;
            break;
        case SOI3_MOD_CGFBO:
            model->SOI3frontGateBulkOverlapCapFactor = value->rValue;
            model->SOI3frontGateBulkOverlapCapFactorGiven = TRUE;
            break;
        case SOI3_MOD_CGBSO:
            model->SOI3backGateSourceOverlapCapAreaFactor = value->rValue;
            model->SOI3backGateSourceOverlapCapAreaFactorGiven = TRUE;
            break;
        case SOI3_MOD_CGBDO:
            model->SOI3backGateDrainOverlapCapAreaFactor = value->rValue;
            model->SOI3backGateDrainOverlapCapAreaFactorGiven = TRUE;
            break;
        case SOI3_MOD_CGBBO:
            model->SOI3backGateBulkOverlapCapAreaFactor = value->rValue;
            model->SOI3backGateBulkOverlapCapAreaFactorGiven = TRUE;
            break;
/*        case SOI3_MOD_CJ:
            model->SOI3bulkCapFactor = value->rValue;
            model->SOI3bulkCapFactorGiven = TRUE;
            break;
        case SOI3_MOD_MJ:
            model->SOI3bulkJctBotGradingCoeff = value->rValue;
            model->SOI3bulkJctBotGradingCoeffGiven = TRUE;
            break;                                              */
        case SOI3_MOD_RSH:
            model->SOI3sheetResistance = value->rValue;
            model->SOI3sheetResistanceGiven = TRUE;
            break;
        case SOI3_MOD_CJSW:
            model->SOI3sideWallCapFactor = value->rValue;
            model->SOI3sideWallCapFactorGiven = TRUE;
            break;
        case SOI3_MOD_MJSW:
            model->SOI3bulkJctSideGradingCoeff = value->rValue;
            model->SOI3bulkJctSideGradingCoeffGiven = TRUE;
            break;
        case SOI3_MOD_JS:
            model->SOI3jctSatCurDensity = value->rValue;
            model->SOI3jctSatCurDensityGiven = TRUE;
            break;
        case SOI3_MOD_JS1:
            model->SOI3jctSatCurDensity1 = value->rValue;
            model->SOI3jctSatCurDensity1Given = TRUE;
            break;
        case SOI3_MOD_TOF:
            model->SOI3frontOxideThickness = value->rValue;
            model->SOI3frontOxideThicknessGiven = TRUE;
            break;
        case SOI3_MOD_TOB:
            model->SOI3backOxideThickness = value->rValue;
            model->SOI3backOxideThicknessGiven = TRUE;
            break;
        case SOI3_MOD_TB:
            model->SOI3bodyThickness = value->rValue;
            model->SOI3bodyThicknessGiven = TRUE;
            break;
        case SOI3_MOD_LD:
            model->SOI3latDiff = value->rValue;
            model->SOI3latDiffGiven = TRUE;
            break;
        case SOI3_MOD_U0:
            model->SOI3surfaceMobility = value->rValue;
            model->SOI3surfaceMobilityGiven = TRUE;
            break;
        case SOI3_MOD_FC:
            model->SOI3fwdCapDepCoeff = value->rValue;
            model->SOI3fwdCapDepCoeffGiven = TRUE;
            break;
        case SOI3_MOD_NSOI3:
            if(value->iValue) {
                model->SOI3type = 1;
                model->SOI3typeGiven = TRUE;
            }
            break;
        case SOI3_MOD_PSOI3:
            if(value->iValue) {
                model->SOI3type = -1;
                model->SOI3typeGiven = TRUE;
            }
            break;
        case SOI3_MOD_KOX:
            model->SOI3oxideThermalConductivity = value->rValue;
            model->SOI3oxideThermalConductivityGiven = TRUE;
            break;
        case SOI3_MOD_SHSI:
            model->SOI3siliconSpecificHeat = value->rValue;
            model->SOI3siliconSpecificHeatGiven = TRUE;
            break;
        case SOI3_MOD_DSI:
            model->SOI3siliconDensity = value->rValue;
            model->SOI3siliconDensityGiven = TRUE;
            break;
        case SOI3_MOD_NSUB:
            model->SOI3substrateDoping = value->rValue;
            model->SOI3substrateDopingGiven = TRUE;
            break;
        case SOI3_MOD_TPG:
            model->SOI3gateType = value->iValue;
            model->SOI3gateTypeGiven = TRUE;
            break;
        case SOI3_MOD_NQFF:
            model->SOI3frontFixedChargeDensity = value->rValue;
            model->SOI3frontFixedChargeDensityGiven = TRUE;
            break;
        case SOI3_MOD_NQFB:
            model->SOI3backFixedChargeDensity = value->rValue;
            model->SOI3backFixedChargeDensityGiven = TRUE;
            break;
        case SOI3_MOD_NSSF:
            model->SOI3frontSurfaceStateDensity = value->rValue;
            model->SOI3frontSurfaceStateDensityGiven = TRUE;
            break;
        case SOI3_MOD_NSSB:
            model->SOI3backSurfaceStateDensity = value->rValue;
            model->SOI3backSurfaceStateDensityGiven = TRUE;
            break;
        case SOI3_MOD_TNOM:
            model->SOI3tnom = value->rValue+CONSTCtoK;
            model->SOI3tnomGiven = TRUE;
            break;
        case SOI3_MOD_KF:
            model->SOI3fNcoef = value->rValue;
            model->SOI3fNcoefGiven = TRUE;
            break;
        case SOI3_MOD_AF:
            model->SOI3fNexp = value->rValue;
            model->SOI3fNexpGiven = TRUE;
            break;
/* extra stuff for newer model - msll Jan96 */
        case SOI3_MOD_SIGMA:
            model->SOI3sigma = value->rValue;
            model->SOI3sigmaGiven = TRUE;
            break;
        case SOI3_MOD_CHIFB:
            model->SOI3chiFB = value->rValue;
            model->SOI3chiFBGiven = TRUE;
            break;
        case SOI3_MOD_CHIPHI:
            model->SOI3chiPHI = value->rValue;
            model->SOI3chiPHIGiven = TRUE;
            break;
        case SOI3_MOD_DELTAW:
            model->SOI3deltaW = value->rValue;
            model->SOI3deltaWGiven = TRUE;
            break;
        case SOI3_MOD_DELTAL:
            model->SOI3deltaL = value->rValue;
            model->SOI3deltaLGiven = TRUE;
            break;
        case SOI3_MOD_VSAT:
            model->SOI3vsat = value->rValue;
            model->SOI3vsatGiven = TRUE;
            break;
        case SOI3_MOD_K:
            model->SOI3k = value->rValue;
            model->SOI3kGiven = TRUE;
            break;
        case SOI3_MOD_LX:
            model->SOI3lx = value->rValue;
            model->SOI3lxGiven = TRUE;
            break;
        case SOI3_MOD_VP:
            model->SOI3vp = value->rValue;
            model->SOI3vpGiven = TRUE;
            break;
        case SOI3_MOD_ETA:
            model->SOI3eta = value->rValue;
            model->SOI3etaGiven = TRUE;
            break;
        case SOI3_MOD_ALPHA0:
            model->SOI3alpha0 = value->rValue;
            model->SOI3alpha0Given = TRUE;
            break;
        case SOI3_MOD_BETA0:
            model->SOI3beta0 = value->rValue;
            model->SOI3beta0Given = TRUE;
            break;
        case SOI3_MOD_LM:
            model->SOI3lm = value->rValue;
            model->SOI3lmGiven = TRUE;
            break;
        case SOI3_MOD_LM1:
            model->SOI3lm1 = value->rValue;
            model->SOI3lm1Given = TRUE;
            break;
        case SOI3_MOD_LM2:
            model->SOI3lm2 = value->rValue;
            model->SOI3lm2Given = TRUE;
            break;
        case SOI3_MOD_ETAD:
            model->SOI3etad = value->rValue;
            model->SOI3etadGiven = TRUE;
            break;
        case SOI3_MOD_ETAD1:
            model->SOI3etad1 = value->rValue;
            model->SOI3etad1Given = TRUE;
            break;
        case SOI3_MOD_CHIBETA:
            model->SOI3chibeta = value->rValue;
            model->SOI3chibetaGiven = TRUE;
            break;
        case SOI3_MOD_VFBB:
            model->SOI3vfbB = value->rValue;
            model->SOI3vfbBGiven = TRUE;
            break;
        case SOI3_MOD_GAMMAB:
            model->SOI3gammaB = value->rValue;
            model->SOI3gammaBGiven = TRUE;
            break;
        case SOI3_MOD_CHID:
            model->SOI3chid = value->rValue;
            model->SOI3chidGiven = TRUE;
            break;
        case SOI3_MOD_CHID1:
            model->SOI3chid1 = value->rValue;
            model->SOI3chid1Given = TRUE;
            break;
        case SOI3_MOD_DVT:
            model->SOI3dvt = value->iValue;
            model->SOI3dvtGiven = TRUE;
            break;
        case SOI3_MOD_NLEV:
            model->SOI3nLev = value->iValue;
            model->SOI3nLevGiven = TRUE;
            break;
        case SOI3_MOD_BETABJT:
            model->SOI3betaBJT = value->rValue;
            model->SOI3betaBJTGiven = TRUE;
            break;
        case SOI3_MOD_TAUFBJT:
            model->SOI3tauFBJT = value->rValue;
            model->SOI3tauFBJTGiven = TRUE;
            break;
        case SOI3_MOD_TAURBJT:
            model->SOI3tauRBJT = value->rValue;
            model->SOI3tauRBJTGiven = TRUE;
            break;
        case SOI3_MOD_BETAEXP:
            model->SOI3betaEXP = value->rValue;
            model->SOI3betaEXPGiven = TRUE;
            break;
        case SOI3_MOD_TAUEXP:
            model->SOI3tauEXP = value->rValue;
            model->SOI3tauEXPGiven = TRUE;
            break;
        case SOI3_MOD_RSW:
            model->SOI3rsw = value->rValue;
            model->SOI3rswGiven = TRUE;
            break;
        case SOI3_MOD_RDW:
            model->SOI3rdw = value->rValue;
            model->SOI3rdwGiven = TRUE;
            break;
        case SOI3_MOD_FMIN:
            model->SOI3minimumFeatureSize = value->rValue;
            model->SOI3minimumFeatureSizeGiven = TRUE;
            break;
        case SOI3_MOD_VTEX:
            model->SOI3vtex = value->rValue;
            model->SOI3vtexGiven = TRUE;
            break;
        case SOI3_MOD_VDEX:
            model->SOI3vdex = value->rValue;
            model->SOI3vdexGiven = TRUE;
            break;
        case SOI3_MOD_DELTA0:
            model->SOI3delta0 = value->rValue;
            model->SOI3delta0Given = TRUE;
            break;
        case SOI3_MOD_CSF:
            model->SOI3satChargeShareFactor = value->rValue;
            model->SOI3satChargeShareFactorGiven = TRUE;
            break;
        case SOI3_MOD_NPLUS:
            model->SOI3nplusDoping = value->rValue;
            model->SOI3nplusDopingGiven = TRUE;
            break;
        case SOI3_MOD_RTA:
            model->SOI3rta = value->rValue;
            model->SOI3rtaGiven = TRUE;
            break;
        case SOI3_MOD_CTA:
            model->SOI3cta = value->rValue;
            model->SOI3ctaGiven = TRUE;
            break;
	case SOI3_MOD_MEXP:
            model->SOI3mexp = value->rValue;
            model->SOI3mexpGiven = TRUE;
            break;

        default:
            return(E_BADPARM);
    }
    return(OK);
}

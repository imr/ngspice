/**********
Copyright 1993: T. Ytterdal, K. Lee, M. Shur and T. A. Fjeldly. All rights reserved.
Author: Trond Ytterdal
**********/

#include "ngspice/ngspice.h"
#include "ngspice/const.h"
#include "ngspice/ifsim.h"
#include "mesadefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
MESAmParam(int param, IFvalue *value, GENmodel *inModel)
{
    MESAmodel *model = (MESAmodel*)inModel;
    switch(param) {
        case MESA_MOD_VTO:
            model->MESAthresholdGiven = TRUE;
            model->MESAthreshold = value->rValue;
            break;
        case MESA_MOD_BETA:
            model->MESAbetaGiven = TRUE;
            model->MESAbeta = value->rValue;
            break;
        case MESA_MOD_VS:
            model->MESAvsGiven = TRUE;
            model->MESAvs = value->rValue;
            break;
        case MESA_MOD_LAMBDA:
            model->MESAlambdaGiven = TRUE;
            model->MESAlambda = value->rValue;
            break;
        case MESA_MOD_RD:
            model->MESAdrainResistGiven = TRUE;
            model->MESAdrainResist = value->rValue;
            break;
        case MESA_MOD_RS:
            model->MESAsourceResistGiven = TRUE;
            model->MESAsourceResist = value->rValue;
            break;
        case MESA_MOD_RG:
            model->MESAgateResistGiven = TRUE;
            model->MESAgateResist = value->rValue;
            break;
        case MESA_MOD_RI:
            model->MESAriGiven = TRUE;
            model->MESAri = value->rValue;
            break;
        case MESA_MOD_RF:
            model->MESArfGiven = TRUE;
            model->MESArf = value->rValue;
            break;
        case MESA_MOD_RDI:
            model->MESArdiGiven = TRUE;
            model->MESArdi = value->rValue;
            break;
        case MESA_MOD_RSI:
            model->MESArsiGiven = TRUE;
            model->MESArsi = value->rValue;
            break;
        case MESA_MOD_PHIB:
            model->MESAphibGiven = TRUE;
            model->MESAphib = value->rValue*CHARGE;
            break;
        case MESA_MOD_PHIB1:
            model->MESAphib1Given = TRUE;
            model->MESAphib1 = value->rValue*CHARGE;
            break;
        case MESA_MOD_ASTAR:
            model->MESAastarGiven = TRUE;
            model->MESAastar = value->rValue;
            break;
        case MESA_MOD_GGR:
            model->MESAggrGiven = TRUE;
            model->MESAggr = value->rValue;
            break;
        case MESA_MOD_DEL:
            model->MESAdelGiven = TRUE;
            model->MESAdel = value->rValue;
            break;
        case MESA_MOD_XCHI:
            model->MESAxchiGiven = TRUE;
            model->MESAxchi = value->rValue;
            break;
        case MESA_MOD_N:
            model->MESAnGiven = TRUE;
            model->MESAn = value->rValue;
            break;
        case MESA_MOD_ETA:
            model->MESAetaGiven = TRUE;
            model->MESAeta = value->rValue;
            break;
        case MESA_MOD_M:
            model->MESAmGiven = TRUE;
            model->MESAm = value->rValue;
            break;
        case MESA_MOD_MC:
            model->MESAmcGiven = TRUE;
            model->MESAmc = value->rValue;
            break;
        case MESA_MOD_ALPHA:
            model->MESAalphaGiven = TRUE;
            model->MESAalpha = value->rValue;
            break;
        case MESA_MOD_SIGMA0:
            model->MESAsigma0Given = TRUE;
            model->MESAsigma0 = value->rValue;
            break;
        case MESA_MOD_VSIGMAT:
            model->MESAvsigmatGiven = TRUE;
            model->MESAvsigmat = value->rValue;
            break;
        case MESA_MOD_VSIGMA:
            model->MESAvsigmaGiven = TRUE;
            model->MESAvsigma = value->rValue;
            break;
        case MESA_MOD_MU:
            model->MESAmuGiven = TRUE;
            model->MESAmu = value->rValue;
            break;
        case MESA_MOD_THETA:
            model->MESAthetaGiven = TRUE;
            model->MESAtheta = value->rValue;
            break;
        case MESA_MOD_MU1:
            model->MESAmu1Given = TRUE;
            model->MESAmu1 = value->rValue;
            break;
        case MESA_MOD_MU2:
            model->MESAmu2Given = TRUE;
            model->MESAmu2 = value->rValue;
            break;
        case MESA_MOD_D:
            model->MESAdGiven = TRUE;
            model->MESAd = value->rValue;
            break;
        case MESA_MOD_ND:
            model->MESAndGiven = TRUE;
            model->MESAnd = value->rValue;
            break;
        case MESA_MOD_DU:
            model->MESAduGiven = TRUE;
            model->MESAdu = value->rValue;
            break;
        case MESA_MOD_NDU:
            model->MESAnduGiven = TRUE;
            model->MESAndu = value->rValue;
            break;
        case MESA_MOD_TH:
            model->MESAthGiven = TRUE;
            model->MESAth = value->rValue;
            break;
        case MESA_MOD_NDELTA:
            model->MESAndeltaGiven = TRUE;
            model->MESAndelta = value->rValue;
            break;
        case MESA_MOD_DELTA:
            model->MESAdeltaGiven = TRUE;
            model->MESAdelta = value->rValue;
            break;
        case MESA_MOD_TC:
            model->MESAtcGiven = TRUE;
            model->MESAtc = value->rValue;
            break;
        case MESA_MOD_NMF:
            if(value->iValue) {
                model->MESAtype = NMF;
            }
            break;
        case MESA_MOD_PMF:
            if(value->iValue) {
                fprintf(stderr, "Only nmf model type supported, set to nmf\n");
                model->MESAtype = NMF;
            }
            break;
        case MESA_MOD_TVTO:
            model->MESAtvtoGiven = TRUE;
            model->MESAtvto = value->rValue;
            break;
        case MESA_MOD_TLAMBDA:
            model->MESAtlambdaGiven = TRUE;
            model->MESAtlambda = value->rValue+CONSTCtoK;
            break;
        case MESA_MOD_TETA0:
            model->MESAteta0Given = TRUE;
            model->MESAteta0 = value->rValue+CONSTCtoK;
            break;
        case MESA_MOD_TETA1:
            model->MESAteta1Given = TRUE;
            model->MESAteta1 = value->rValue+CONSTCtoK;
            break;
        case MESA_MOD_TMU:
            model->MESAtmuGiven = TRUE;
            model->MESAtmu = value->rValue+CONSTCtoK;
            break;
        case MESA_MOD_XTM0:
            model->MESAxtm0Given = TRUE;
            model->MESAxtm0 = value->rValue;
            break;
        case MESA_MOD_XTM1:
            model->MESAxtm1Given = TRUE;
            model->MESAxtm1 = value->rValue;
            break;
        case MESA_MOD_XTM2:
            model->MESAxtm2Given = TRUE;
            model->MESAxtm2 = value->rValue;
            break;
        case MESA_MOD_KS:
            model->MESAksGiven = TRUE;
            model->MESAks = value->rValue;
            break;
        case MESA_MOD_VSG:
            model->MESAvsgGiven = TRUE;
            model->MESAvsg = value->rValue;
            break;
        case MESA_MOD_LAMBDAHF:
            model->MESAlambdahfGiven = TRUE;
            model->MESAlambdahf = value->rValue;
            break;
        case MESA_MOD_TF:
            model->MESAtfGiven = TRUE;
            model->MESAtf = value->rValue+CONSTCtoK;
            break;
        case MESA_MOD_FLO:
            model->MESAfloGiven = TRUE;
            model->MESAflo = value->rValue;
            break;
        case MESA_MOD_DELFO:
            model->MESAdelfoGiven = TRUE;
            model->MESAdelfo = value->rValue;
            break;
        case MESA_MOD_AG:
            model->MESAagGiven = TRUE;
            model->MESAag = value->rValue;
            break;
        case MESA_MOD_TC1:
            model->MESAtc1Given = TRUE;
            model->MESAtc1 = value->rValue;
            break;
        case MESA_MOD_TC2:
            model->MESAtc2Given = TRUE;
            model->MESAtc2 = value->rValue;
            break;
        case MESA_MOD_ZETA:
            model->MESAzetaGiven = TRUE;
            model->MESAzeta = value->rValue;
            break;
        case MESA_MOD_LEVEL:
            model->MESAlevelGiven = TRUE;
            model->MESAlevel = value->rValue;
            break;
        case MESA_MOD_NMAX:
            model->MESAnmaxGiven = TRUE;
            model->MESAnmax = value->rValue;
            break;
        case MESA_MOD_GAMMA:
            model->MESAgammaGiven = TRUE;
            model->MESAgamma = value->rValue;
            break;
        case MESA_MOD_EPSI:
            model->MESAepsiGiven = TRUE;
            model->MESAepsi = value->rValue;
            break;
        case MESA_MOD_CBS:
            model->MESAcbsGiven = TRUE;
            model->MESAcbs = value->rValue;
            break;
        case MESA_MOD_CAS:
            model->MESAcasGiven = TRUE;
            model->MESAcas = value->rValue;
            break;
        default:
            return(E_BADPARM);
    }
    return(OK);
}

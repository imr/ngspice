/**********
Imported from MacSpice3f4 - Antony Wilson
Modified: Paolo Nenzi
**********/

#include "ngspice/ngspice.h"
#include "ngspice/const.h"
#include "ngspice/ifsim.h"
#include "hfetdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
HFETAmParam(int param, IFvalue *value, GENmodel *inModel)
{
    HFETAmodel *model = (HFETAmodel*)inModel;
    switch(param) 
    {
        case HFETA_MOD_VTO:
            model->HFETAthresholdGiven = TRUE;
            model->HFETAthreshold = value->rValue;
            break;
        case HFETA_MOD_LAMBDA:
            model->HFETAlambdaGiven = TRUE;
            model->HFETAlambda = value->rValue;
            break;
        case HFETA_MOD_RD:
            model->HFETArdGiven = TRUE;
            model->HFETArd = value->rValue;
            break;
        case HFETA_MOD_RS:
            model->HFETArsGiven = TRUE;
            model->HFETArs = value->rValue;
            break;
        case HFETA_MOD_RG:
            model->HFETArgGiven = TRUE;
            model->HFETArg = value->rValue;
            break;
        case HFETA_MOD_RDI:
            model->HFETArdiGiven = TRUE;
            model->HFETArdi = value->rValue;
            break;
        case HFETA_MOD_RSI:
            model->HFETArsiGiven = TRUE;
            model->HFETArsi = value->rValue;
            break;
        case HFETA_MOD_RGS:
            model->HFETArgsGiven = TRUE;
            model->HFETArgs = value->rValue;
            break;
        case HFETA_MOD_RGD:
            model->HFETArgdGiven = TRUE;
            model->HFETArgd = value->rValue;
            break;
        case HFETA_MOD_RI:
            model->HFETAriGiven = TRUE;
            model->HFETAri = value->rValue;
            break;
        case HFETA_MOD_RF:
            model->HFETArfGiven = TRUE;
            model->HFETArf = value->rValue;
            break;
        case HFETA_MOD_ETA:
            model->HFETAetaGiven = TRUE;
            model->HFETAeta = value->rValue;
            break;
        case HFETA_MOD_M:
            model->HFETAmGiven = TRUE;
            model->HFETAm = value->rValue;
            break;
        case HFETA_MOD_MC:
            model->HFETAmcGiven = TRUE;
            model->HFETAmc = value->rValue;
            break;
        case HFETA_MOD_GAMMA:
            model->HFETAgammaGiven = TRUE;
            model->HFETAgamma = value->rValue;
            break;
        case HFETA_MOD_SIGMA0:
            model->HFETAsigma0Given = TRUE;
            model->HFETAsigma0 = value->rValue;
            break;
        case HFETA_MOD_VSIGMAT:
            model->HFETAvsigmatGiven = TRUE;
            model->HFETAvsigmat = value->rValue;
            break;
        case HFETA_MOD_VSIGMA:
            model->HFETAvsigmaGiven = TRUE;
            model->HFETAvsigma = value->rValue;
            break;
        case HFETA_MOD_MU:
            model->HFETAmuGiven = TRUE;
            model->HFETAmu = value->rValue;
            break;
        case HFETA_MOD_DI:
            model->HFETAdiGiven = TRUE;
            model->HFETAdi = value->rValue;
            break;
        case HFETA_MOD_DELTA:
            model->HFETAdeltaGiven = TRUE;
            model->HFETAdelta = value->rValue;
            break;
        case HFETA_MOD_VS:
            model->HFETAvsGiven = TRUE;
            model->HFETAvs = value->rValue;
            break;
        case HFETA_MOD_NMAX:
            model->HFETAnmaxGiven = TRUE;
            model->HFETAnmax = value->rValue;
            break;
        case HFETA_MOD_DELTAD:
            model->HFETAdeltadGiven = TRUE;
            model->HFETAdeltad = value->rValue;
            break;
        case HFETA_MOD_JS1D:
            model->HFETAjs1dGiven = TRUE;
            model->HFETAjs1d = value->rValue;
            break;
        case HFETA_MOD_JS2D:
            model->HFETAjs2dGiven = TRUE;
            model->HFETAjs2d = value->rValue;
            break;
        case HFETA_MOD_JS1S:
            model->HFETAjs1sGiven = TRUE;
            model->HFETAjs1s = value->rValue;
            break;
        case HFETA_MOD_JS2S:
            model->HFETAjs2sGiven = TRUE;
            model->HFETAjs2s = value->rValue;
            break;
        case HFETA_MOD_M1D:
            model->HFETAm1dGiven = TRUE;
            model->HFETAm1d = value->rValue;
            break;
        case HFETA_MOD_M2D:
            model->HFETAm2dGiven = TRUE;
            model->HFETAm2d = value->rValue;
            break;
        case HFETA_MOD_M1S:
            model->HFETAm1sGiven = TRUE;
            model->HFETAm1s = value->rValue;
            break;
        case HFETA_MOD_M2S:
            model->HFETAm2sGiven = TRUE;
            model->HFETAm2s = value->rValue;
            break;
        case HFETA_MOD_EPSI:
            model->HFETAepsiGiven = TRUE;
            model->HFETAepsi = value->rValue;
            break;
        case HFETA_MOD_A1:
            model->HFETAa1Given = TRUE;
            model->HFETAa1 = value->rValue;
            break;
        case HFETA_MOD_A2:
            model->HFETAa2Given = TRUE;
            model->HFETAa2 = value->rValue;
            break;
        case HFETA_MOD_MV1:
            model->HFETAmv1Given = TRUE;
            model->HFETAmv1 = value->rValue;
            break;
        case HFETA_MOD_P:
            model->HFETApGiven = TRUE;
            model->HFETAp = value->rValue;
            break;
        case HFETA_MOD_KAPPA:
            model->HFETAkappaGiven = TRUE;
            model->HFETAkappa = value->rValue;
            break;
        case HFETA_MOD_DELF:
            model->HFETAdelfGiven = TRUE;
            model->HFETAdelf = value->rValue;
            break;
        case HFETA_MOD_FGDS:
            model->HFETAfgdsGiven = TRUE;
            model->HFETAfgds = value->rValue;
            break;
        case HFETA_MOD_TF:
            model->HFETAtfGiven = TRUE;
            model->HFETAtf = value->rValue+CONSTCtoK;
            break;
        case HFETA_MOD_CDS:
            model->HFETAcdsGiven = TRUE;
            model->HFETAcds = value->rValue;
            break;
        case HFETA_MOD_PHIB:
            model->HFETAphibGiven = TRUE;
            model->HFETAphib = value->rValue*CHARGE;
            break;
        case HFETA_MOD_TALPHA:
            model->HFETAtalphaGiven = TRUE;
            model->HFETAtalpha = value->rValue;
            break;
        case HFETA_MOD_MT1:
            model->HFETAmt1Given = TRUE;
            model->HFETAmt1 = value->rValue;
            break;
        case HFETA_MOD_MT2:
            model->HFETAmt2Given = TRUE;
            model->HFETAmt2 = value->rValue;
            break;
        case HFETA_MOD_CK1:
            model->HFETAck1Given = TRUE;
            model->HFETAck1 = value->rValue;
            break;
        case HFETA_MOD_CK2:
            model->HFETAck2Given = TRUE;
            model->HFETAck2 = value->rValue;
            break;
        case HFETA_MOD_CM1:
            model->HFETAcm1Given = TRUE;
            model->HFETAcm1 = value->rValue;
            break;
        case HFETA_MOD_CM2:
            model->HFETAcm2Given = TRUE;
            model->HFETAcm2 = value->rValue;
            break;
        case HFETA_MOD_CM3:
            model->HFETAcm3Given = TRUE;
            model->HFETAcm3 = value->rValue;
            break;
        case HFETA_MOD_ASTAR:
            model->HFETAastarGiven = TRUE;
            model->HFETAastar = value->rValue;
            break;
        case HFETA_MOD_ETA1:
            model->HFETAeta1Given = TRUE;
            model->HFETAeta1 = value->rValue;
            break;
        case HFETA_MOD_D1:
            model->HFETAd1Given = TRUE;
            model->HFETAd1 = value->rValue;
            break;
        case HFETA_MOD_VT1:
            model->HFETAvt1Given = TRUE;
            model->HFETAvt1 = value->rValue;
            break;
        case HFETA_MOD_ETA2:
            model->HFETAeta2Given = TRUE;
            model->HFETAeta2 = value->rValue;
            break;
        case HFETA_MOD_D2:
            model->HFETAd2Given = TRUE;
            model->HFETAd2 = value->rValue;
            break;
        case HFETA_MOD_VT2:
            model->HFETAvt2Given = TRUE;
            model->HFETAvt2 = value->rValue;
            break;
        case HFETA_MOD_GGR:
            model->HFETAggrGiven = TRUE;
            model->HFETAggr = value->rValue;
            break;
        case HFETA_MOD_DEL:
            model->HFETAdelGiven = TRUE;
            model->HFETAdel = value->rValue;
            break;
        case HFETA_MOD_GATEMOD:
            model->HFETAgatemodGiven = TRUE;
            model->HFETAgatemod = value->iValue;
            break;
        case HFETA_MOD_KLAMBDA:
           model->HFETAklambdaGiven = TRUE;
           KLAMBDA = value->rValue;
           break;
        case HFETA_MOD_KMU:
           model->HFETAkmuGiven = TRUE;
           KMU = value->rValue;
           break;
        case HFETA_MOD_KVTO:
           model->HFETAkvtoGiven = TRUE;
           KVTO = value->rValue;
           break;
        case HFETA_MOD_NHFET:
            if(value->iValue) {
                model->HFETAtype = NHFET;
            }
            break;
        case HFETA_MOD_PHFET:
            if(value->iValue) {
                model->HFETAtype = PHFET;
            }
            break;
        default:
            return(E_BADPARM);
    }
    return(OK);
}

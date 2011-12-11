/**********
Imported from MacSpice3f4 - Antony Wilson
Modified: Paolo Nenzi
**********/

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "hfet2defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int HFET2mParam(int param, IFvalue *value, GENmodel *inModel)
{
  
  HFET2model *model = (HFET2model*)inModel;
  switch(param) {
    case HFET2_MOD_CF:
      model->HFET2cfGiven = TRUE;
      CF = value->rValue;
      break;
    case HFET2_MOD_D1:
      model->HFET2d1Given = TRUE;
      D1 = value->rValue;
      break;
    case HFET2_MOD_D2:
      model->HFET2d2Given = TRUE;
      D2 = value->rValue;
      break;
    case HFET2_MOD_DEL:
      model->HFET2delGiven = TRUE;
      DEL = value->rValue;
      break;      
    case HFET2_MOD_DELTA:
      model->HFET2deltaGiven = TRUE;
      DELTA = value->rValue;
      break;      
    case HFET2_MOD_DELTAD:
      model->HFET2deltadGiven = TRUE;
      DELTAD = value->rValue;
      break;
    case HFET2_MOD_DI:
      model->HFET2diGiven = TRUE;
      DI = value->rValue;
      break;
    case HFET2_MOD_EPSI:
      model->HFET2epsiGiven = TRUE;
      EPSI = value->rValue;
      break;
    case HFET2_MOD_ETA:
      model->HFET2etaGiven = TRUE;
      ETA = value->rValue;
      break;
    case HFET2_MOD_ETA1:
      model->HFET2eta1Given = TRUE;
      ETA1 = value->rValue;
      break;
    case HFET2_MOD_ETA2:
      model->HFET2eta2Given = TRUE;
      ETA2 = value->rValue;
      break;
    case HFET2_MOD_GAMMA:
      model->HFET2gammaGiven = TRUE;
      GAMMA = value->rValue;
      break;
    case HFET2_MOD_GGR:
      model->HFET2ggrGiven = TRUE;
      GGR = value->rValue;
      break;
    case HFET2_MOD_JS:
      model->HFET2jsGiven = TRUE;
      JS = value->rValue;
      break;      
    case HFET2_MOD_KLAMBDA:
      model->HFET2klambdaGiven = TRUE;
      KLAMBDA = value->rValue;
      break;
    case HFET2_MOD_KMU:
      model->HFET2kmuGiven = TRUE;
      KMU = value->rValue;
      break;
    case HFET2_MOD_KNMAX:
      model->HFET2knmaxGiven = TRUE;
      KNMAX = value->rValue;
      break;
    case HFET2_MOD_KVTO:
      model->HFET2kvtoGiven = TRUE;
      KVTO = value->rValue;
      break;
    case HFET2_MOD_LAMBDA:
      model->HFET2lambdaGiven = TRUE;
      LAMBDA = value->rValue;
      break;
    case HFET2_MOD_M:
      model->HFET2mGiven = TRUE;
      M = value->rValue;
      break;
    case HFET2_MOD_MC:
      model->HFET2mcGiven = TRUE;
      MC = value->rValue;
      break;
    case HFET2_MOD_MU:
      model->HFET2muGiven = TRUE;
      MU = value->rValue;
      break;
    case HFET2_MOD_N:
      model->HFET2nGiven = TRUE;
      N = value->rValue;
      break;      
    case HFET2_MOD_NMAX:
      model->HFET2nmaxGiven = TRUE;
      NMAX = value->rValue;
      break;
    case HFET2_MOD_P:
      model->HFET2pGiven = TRUE;
      PP = value->rValue;
      break;
    case HFET2_MOD_RD:
      model->HFET2rdGiven = TRUE;
      RD = value->rValue;
      break;
    case HFET2_MOD_RDI:
      model->HFET2rdiGiven = TRUE;
      RDI = value->rValue;
      break;
    case HFET2_MOD_RS:
      model->HFET2rsGiven = TRUE;
      RS = value->rValue;
      break;
    case HFET2_MOD_RSI:
      model->HFET2rsiGiven = TRUE;
      RSI = value->rValue;
      break;
    case HFET2_MOD_SIGMA0:
      model->HFET2sigma0Given = TRUE;
      SIGMA0 = value->rValue;
      break;
    case HFET2_MOD_VS:
      model->HFET2vsGiven = TRUE;
      VS = value->rValue;
      break;
    case HFET2_MOD_VSIGMA:
      model->HFET2vsigmaGiven = TRUE;
      VSIGMA = value->rValue;
      break;
    case HFET2_MOD_VSIGMAT:
      model->HFET2vsigmatGiven = TRUE;
      VSIGMAT = value->rValue;
      break;
    case HFET2_MOD_VT1:
      model->HFET2vt1Given = TRUE;
      HFET2_VT1 = value->rValue;
      break;
    case HFET2_MOD_VT2:
      model->HFET2vt2Given = TRUE;
      VT2 = value->rValue;
      break;
    case HFET2_MOD_VTO:
      model->HFET2vtoGiven = TRUE;
      VTO = value->rValue;
      break;
    case HFET2_MOD_NHFET:
      if(value->iValue) {
        TYPE = NHFET;
      }
      break;
    case HFET2_MOD_PHFET:
      if(value->iValue) {
        TYPE = PHFET;
      }
      break;
    default:
      return(E_BADPARM);
  }
  return(OK);
  
}

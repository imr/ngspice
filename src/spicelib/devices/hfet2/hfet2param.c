/**********
Imported from MacSpice3f4 - Antony Wilson
Modified: Paolo Nenzi
**********/

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "hfet2defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int HFET2param(int param, IFvalue *value, GENinstance *inst, IFvalue *select)
{
  
  HFET2instance *here = (HFET2instance*)inst;

  NG_IGNORE(select);

  switch (param) {
    case HFET2_LENGTH:
      L = value->rValue;
      here->HFET2lengthGiven = TRUE;
      break;
    case HFET2_IC_VDS:
      here->HFET2icVDS = value->rValue;
      here->HFET2icVDSGiven = TRUE;
      break;
    case HFET2_IC_VGS:
      here->HFET2icVGS = value->rValue;
      here->HFET2icVGSGiven = TRUE;
      break;
    case HFET2_OFF:
      here->HFET2off = value->iValue;
      break;
    case HFET2_IC:
        /* FALLTHROUGH added to suppress GCC warning due to
         * -Wimplicit-fallthrough flag */
      switch (value->v.numValue) {
        case 2:
          here->HFET2icVGS = *(value->v.vec.rVec+1);
          here->HFET2icVGSGiven = TRUE;
          /* FALLTHROUGH */
        case 1:
          here->HFET2icVDS = *(value->v.vec.rVec);
          here->HFET2icVDSGiven = TRUE;
          break;
        default:
          return(E_BADPARM);
      }
      break;
    case HFET2_TEMP:
      TEMP = value->rValue + CONSTCtoK;
      here->HFET2tempGiven = TRUE;
      break;
    case HFET2_DTEMP:
      here->HFET2dtemp = value->rValue;
      here->HFET2dtempGiven = TRUE;
      break;
    case HFET2_WIDTH:
      W = value->rValue;
      here->HFET2widthGiven = TRUE;
      break;
    case HFET2_M:
      here->HFET2m = value->rValue;
      here->HFET2mGiven = TRUE;
      break;
    default:
      return(E_BADPARM);
  }
  return(OK);
  
}

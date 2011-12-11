/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

#include "ngspice/ngspice.h"
#include "ngspice/const.h"
#include "ngspice/ifsim.h"
#include "numd2def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
NUMD2param(int param, IFvalue *value, GENinstance *inInst, IFvalue *select)
{
  NUMD2instance *inst = (NUMD2instance *) inInst;

  NG_IGNORE(select);

  switch (param) {
  case NUMD2_WIDTH:
    inst->NUMD2width = value->rValue;
    inst->NUMD2widthGiven = TRUE;
    break;
  case NUMD2_AREA:
    inst->NUMD2area = value->rValue;
    inst->NUMD2areaGiven = TRUE;
    break;
  case NUMD2_OFF:
    inst->NUMD2off = TRUE;
    break;
  case NUMD2_IC_FILE:
    inst->NUMD2icFile = value->sValue;
    inst->NUMD2icFileGiven = TRUE;
    break;
  case NUMD2_PRINT:
    inst->NUMD2print = value->iValue;
    inst->NUMD2printGiven = TRUE;
    break;
  case NUMD2_TEMP:
    inst->NUMD2temp = value->rValue + CONSTCtoK;
    inst->NUMD2tempGiven = TRUE;
    break;
  default:
    return (E_BADPARM);
  }
  return (OK);
}

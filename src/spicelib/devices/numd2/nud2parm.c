/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

#include "ngspice.h"
#include "const.h"
#include "ifsim.h"
#include "numd2def.h"
#include "sperror.h"
#include "suffix.h"

int
NUMD2param(param, value, inInst, select)
  int param;
  IFvalue *value;
  GENinstance *inInst;
  IFvalue *select;
{
  NUMD2instance *inst = (NUMD2instance *) inInst;
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
    inst->NUMD2print = value->rValue;
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
